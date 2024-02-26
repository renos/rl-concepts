from contrib.achievement_distillation.impala_cnn import ImpalaCNN
from contrib.achievement_distillation.mse_head import (
    CategoricalActionHead,
    ScaledMSEHead,
)
from contrib.achievement_distillation.torch_util import (
    FanInInitReLULayer,
    StackedFCFanInInitReLULayer,
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


import torch.nn
import torch.optim
from concept_module import ConceptNetwork, SimplifiedConceptNetwork
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.data.tensor_specs import DiscreteBox
from torchrl.envs import ExplorationType
import torchsummary


class CNNTorchRLWrapper(torch.nn.Module):
    def __init__(self, cnn):
        super(CNNTorchRLWrapper, self).__init__()
        self.cnn = cnn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, C, L, W = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = self.cnn(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out


def make_ppo_modules_crafter_impala(proof_environment):

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
    in_keys = ["pixels"]

    obs_shape = (3, 64, 64)
    dense_init_norm_kwargs = {"layer_norm": True}
    impala_kwargs = {
        "chans": [64, 128, 128],
        "outsize": 256,
        "nblock": 2,
        "post_pool_groups": 1,
        "init_norm_kwargs": {"batch_norm": False, "group_norm_groups": 1},
    }
    out_size = 256
    hid_size = 1024

    enc = ImpalaCNN(
        obs_shape,
        dense_init_norm_kwargs=dense_init_norm_kwargs,
        **impala_kwargs,
    )
    output_shape = enc(torch.ones(pixels_shape))

    outsize = impala_kwargs["outsize"]
    lin = FanInInitReLULayer(
        outsize,
        hid_size,
        layer_type="linear",
        **dense_init_norm_kwargs,
    )

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=torch.nn.Sequential(CNNTorchRLWrapper(enc), lin),
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    # policy_net = MLP(
    #     in_features=hid_size,
    #     out_features=num_outputs,
    #     activation_class=torch.nn.ReLU,
    #     num_cells=[],
    # )
    policy_net = CategoricalActionHead(
        insize=hid_size, num_actions=num_outputs, init_scale=0.1
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
    # value_net = MLP(
    #     activation_class=torch.nn.ReLU,
    #     in_features=hid_size,
    #     out_features=1,
    #     num_cells=[],
    # )
    value_net = ScaledMSEHead(
        insize=hid_size,
        outsize=1,
        init_scale=0.1,
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
        out_keys=["state_value", "denormalized_state_value"],
    )

    return common_module, policy_module, value_module


def make_ppo_modules_crafter_impala_achievements(proof_environment, model_params):

    # Define input shape
    pixels_shape = proof_environment.observation_spec["pixels"].shape
    status_concepts_shape = proof_environment.observation_spec["status_concepts"].shape
    status_concepts_len = status_concepts_shape[-1] * status_concepts_shape[-2]
    history_concepts_shape = proof_environment.observation_spec[
        "history_concepts"
    ].shape

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
    in_keys = ["pixels", "status_concepts", "history_concepts"]

    obs_shape = (3, 64, 64)
    dense_init_norm_kwargs = {"layer_norm": True}
    impala_kwargs = {
        "chans": [64, 128, 128],
        "outsize": 256,
        "nblock": 2,
        "post_pool_groups": 1,
        "init_norm_kwargs": {"batch_norm": False, "group_norm_groups": 1},
    }
    out_size = 256
    hid_size = 1024

    concepts_hidden_size = 64
    if "use_pixels" in model_params and model_params["use_pixels"]:
        enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        output_shape = enc(torch.ones(pixels_shape))
        outsize = impala_kwargs["outsize"]
    else:
        enc = None

    # enc = ImpalaCNN(
    #     obs_shape,
    #     dense_init_norm_kwargs=dense_init_norm_kwargs,
    #     **impala_kwargs,
    # )
    # output_shape = enc(torch.ones(pixels_shape))

    # torchsummary.summary(enc, pixels_shape[1:], device="cpu")
    # exit()

    # FanInInitReLULayer(
    #     outsize,
    #     hid_size,
    #     layer_type="linear",
    #     **dense_init_norm_kwargs,
    # )

    # concept_encode = FanInInitReLULayer(
    #     achievement_concepts_len,
    #     32,
    #     layer_type="linear",
    #     **dense_init_norm_kwargs,
    # )

    status_concept_encoder = StackedFCFanInInitReLULayer(
        status_concepts_len,
        concepts_hidden_size,
        concepts_hidden_size,
        model_params["status_n_layers"],
        **dense_init_norm_kwargs,
    )

    history_position_embedding = StackedFCFanInInitReLULayer(
        2,
        16,
        16,
        model_params["posenc_n_layers"],
        **dense_init_norm_kwargs,
    )

    history_concept_encoder = StackedFCFanInInitReLULayer(
        16 * history_concepts_shape[-2],
        concepts_hidden_size,
        concepts_hidden_size,
        model_params["history_n_layers"],
        **dense_init_norm_kwargs,
    )

    outsize = (
        impala_kwargs["outsize"] if enc is not None else 0
    ) + 2 * concepts_hidden_size

    joint_encoder = StackedFCFanInInitReLULayer(
        outsize,
        2 * outsize,
        hid_size,
        model_params["joint_n_layers"],
        **dense_init_norm_kwargs,
    )

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=SimplifiedConceptNetwork(
            CNNTorchRLWrapper(enc) if enc is not None else None,
            status_concept_encoder,
            history_position_embedding,
            history_concept_encoder,
            joint_encoder,
        ),
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    # policy_net = MLP(
    #     in_features=hid_size,
    #     out_features=num_outputs,
    #     activation_class=torch.nn.ReLU,
    #     num_cells=[],
    # )
    policy_net = CategoricalActionHead(
        insize=hid_size, num_actions=num_outputs, init_scale=0.1
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
    # value_net = MLP(
    #     activation_class=torch.nn.ReLU,
    #     in_features=hid_size,
    #     out_features=1,
    #     num_cells=[],
    # )
    value_net = ScaledMSEHead(
        insize=hid_size,
        outsize=1,
        init_scale=0.1,
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
        out_keys=["state_value", "denormalized_state_value"],
    )

    return common_module, policy_module, value_module
