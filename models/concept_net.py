import torch.nn
import torch.optim
from concept_module import ConceptNetwork
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.data.tensor_specs import DiscreteBox
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymWrapper,
    NoopResetEnv,
    ParallelEnv,
    Resize,
    RewardClipping,
    RewardSum,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

from torchsummary import summary


def make_ppo_modules_pixels(proof_environment):

    # Define input shape
    pixels_shape = proof_environment.observation_spec["pixels"].shape
    visual_concepts_shape = proof_environment.observation_spec["visual_concepts"].shape
    visual_concepts_flatten = visual_concepts_shape[-1]

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec.space, DiscreteBox):
        num_outputs = proof_environment.action_spec.space.n
        distribution_class = OneHotCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": proof_environment.action_spec.space.low,
            "max": proof_environment.action_spec.space.high,
        }

    # Define input keys
    in_keys = ["pixels", "visual_concepts"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    # use silu here in the future?
    common_cnn = ConvNet(
        activation_class=torch.nn.ReLU,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    common_cnn_output = common_cnn(torch.ones(pixels_shape))

    common_mlp = MLP(
        in_features=common_cnn_output.shape[-1],
        activation_class=torch.nn.ReLU,
        activate_last_layer=True,
        out_features=visual_concepts_flatten,
        num_cells=[512, 512, 512],
    )
    common_mlp_output = common_mlp(common_cnn_output)

    # concatenating time and visual concepts
    concept_encode_input_shape = (
        common_mlp_output.shape[-2] * common_mlp_output.shape[-1]
    )

    commmon_features_len = 512

    concept_network = ConceptNetwork(
        concept_pred=torch.nn.Sequential(common_cnn, common_mlp),
        concept_encode=MLP(
            in_features=concept_encode_input_shape,
            out_features=commmon_features_len,
            num_cells=[],
        ),
        visual_concepts_shape=visual_concepts_shape,
    )

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=concept_network,
        in_keys=in_keys,
        out_keys=["concept_features", "common_features"],
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=commmon_features_len,
        out_features=num_outputs,
        activation_class=torch.nn.ReLU,
        num_cells=[],
    )
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU,
        in_features=commmon_features_len,
        out_features=1,
        num_cells=[],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


def make_ppo_modules_crafter(proof_environment):

    # Define input shape
    pixels_shape = proof_environment.observation_spec["pixels"].shape
    visual_concepts_shape = proof_environment.observation_spec["visual_concepts"].shape
    visual_concepts_flatten = visual_concepts_shape[-1]

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec.space, DiscreteBox):
        num_outputs = proof_environment.action_spec.space.n
        distribution_class = OneHotCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": proof_environment.action_spec.space.low,
            "max": proof_environment.action_spec.space.high,
        }

    # Define input keys
    in_keys = ["pixels", "visual_concepts"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    # use silu here in the future?
    common_cnn = ConvNet(
        activation_class=torch.nn.SiLU,
        num_cells=[32, 64, 64, 19],
        kernel_sizes=[8, 4, 4, 3],
        strides=[4, 2, 1, 1],
        aggregator_class=torch.nn.Identity,
        # norm_class=torch.nn.LayerNorm,
    )

    common_cnn_output = common_cnn(torch.ones(pixels_shape))

    # print(summary(common_cnn.to("cuda"), (1, 128, 128)))
    # exit()

    mlp_wrapper_io = (
        common_cnn_output.shape[-3]
        * common_cnn_output.shape[-2]
        * common_cnn_output.shape[-1]
    )
    common_cnn_wra = MLP(
        in_features=mlp_wrapper_io,
        out_features=mlp_wrapper_io,
        num_cells=[],
    )
    # common_cnn_wra = None

    # concatenating time and visual concepts
    concept_encode_input_shape = (
        common_cnn_output.shape[-4]
        * common_cnn_output.shape[-2]
        * common_cnn_output.shape[-1]
    )

    commmon_features_len = 512

    concept_network = ConceptNetwork(
        concept_pred=common_cnn,
        cnn_wrapper=common_cnn_wra,
        concept_encode=MLP(
            in_features=concept_encode_input_shape,
            out_features=commmon_features_len,
            num_cells=[
                commmon_features_len,
                commmon_features_len,
                commmon_features_len,
            ],
            norm_class=torch.nn.LayerNorm,
            norm_kwargs={"normalized_shape": commmon_features_len},
        ),
        visual_concepts_shape=visual_concepts_shape,
    )

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=concept_network,
        in_keys=in_keys,
        out_keys=["concept_features", "common_features"],
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=commmon_features_len,
        out_features=num_outputs,
        activation_class=torch.nn.SiLU,
        num_cells=[],
    )
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.SiLU,
        in_features=commmon_features_len,
        out_features=1,
        num_cells=[],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module
