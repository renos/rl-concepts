from contrib.achievement_distillation.impala_cnn import ImpalaCNN
from contrib.achievement_distillation.torch_util import FanInInitReLULayer
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
from concept_module import ConceptNetwork
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.data.tensor_specs import DiscreteBox
from torchrl.envs import ExplorationType

from models.fan_init_pyramid import OrthogonalInitLinear, PyramidModule
import torch
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


def make_ppo_modules_crafter_pyramid(proof_environment):

    # Define input shape
    visual_concepts_shape = proof_environment.observation_spec["visual_concepts"].shape

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
    in_keys = ["visual_concepts"]

    out_size = 1024

    enc = PyramidModule(N=2, M=2, inshape=(19, 9, 9))
    output_shape = enc(torch.ones(visual_concepts_shape))

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=CNNTorchRLWrapper(enc),
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    # policy_net = MLP(
    #     in_features=out_size,
    #     out_features=num_outputs,
    #     activation_class=torch.nn.SiLU,
    #     num_cells=[],
    # )
    # dense_init_norm_kwargs = {"layer_norm": True}
    # policy_net = FanInInitReLULayer(
    #     out_size,
    #     num_outputs,
    #     layer_type="linear",
    #     dense_init_norm_kwargs=dense_init_norm_kwargs,
    # )
    policy_net = OrthogonalInitLinear(out_size, num_outputs)

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
        in_features=out_size,
        out_features=1,
        num_cells=[],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module
