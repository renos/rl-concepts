import torch
from torch import nn

from contrib.concept_balanced_loss.cb_loss import CB_Loss

import torchmetrics
from torchrl.objectives.utils import (
    distance_loss,
)


class ConceptNetwork(nn.Module):
    def __init__(
        self, concept_pred, cnn_wrapper, concept_encode, visual_concepts_shape
    ):
        super(ConceptNetwork, self).__init__()
        # Initialize the concept prediction module
        self.concept_pred = concept_pred
        # Initialize the concept encoding module (MLP)
        self.concept_encode = concept_encode
        self.visual_concepts_shape = visual_concepts_shape
        if cnn_wrapper is not None:
            self.cnn_wrapper = cnn_wrapper
        else:
            self.cnn_wrapper = torch.nn.Identity()

    def forward(
        self,
        pixels,
        concepts=None,
    ):
        # Pass the input through the concept prediction module

        x = self.concept_pred(pixels)
        x_shape = x.shape
        x = self.cnn_wrapper(x.flatten(start_dim=-3))
        x = x.reshape(x_shape)
        _concept_features = x.permute(
            *range(x.dim() - 3), x.dim() - 2, x.dim() - 1, x.dim() - 3
        )
        *initial_dims, m3, m2, m1 = _concept_features.shape

        new_shape = (*initial_dims, m3 * m2, m1)
        _concept_features = _concept_features.reshape(new_shape)

        concept_features = _concept_features.detach().argmax(dim=-1)

        if concepts is not None:
            assert (
                concepts.shape == concept_features.shape
            ), f"{concepts.shape} != {concept_features.shape}"
            concept_features = concepts

        # Flatten the concept features time dimension
        *initial_dims, m2, m1 = concept_features.shape
        new_shape = (*initial_dims, m2 * m1)
        concept_features = concept_features.detach().reshape(new_shape)

        # Pass the concept features through the concept encoding module
        common_features = self.concept_encode(concept_features)

        return _concept_features, common_features


class SimplifiedConceptNetwork(nn.Module):
    def __init__(
        self,
        cnn,
        status_concept_encoder,
        history_position_embedding,
        history_concept_encoder,
        joint_encoder,
    ):
        """
        Initializes the SimplifiedConceptNetwork.

        Parameters:
        - cnn: A convolutional neural network (CNN) for processing the input pixels.
        - fc_net: A fully connected (FC) network that takes the concatenated output of the CNN and the concept vector.
        """
        super(SimplifiedConceptNetwork, self).__init__()
        self.cnn = cnn  # Initialize the CNN part of the network
        self.status_concept_encoder = status_concept_encoder
        self.history_position_embedding = history_position_embedding
        self.history_concept_encoder = history_concept_encoder
        self.joint_encoder = joint_encoder
        self.use_pixels = self.cnn is not None

    def forward(self, pixels, status, history):
        """
        Forward pass of the SimplifiedConceptNetwork.

        Parameters:
        - pixels: Input pixels, expected to be a tensor suitable for the CNN.
        - concept_vector: Concept vector to be concatenated with CNN output.

        Returns:
        - Output from the FC network after processing the combined CNN output and concept vector.
        """
        # Pass the input pixels through the CNN
        if self.use_pixels:
            cnn_output = self.cnn(pixels)

        # we don't do positional encoding on history
        status_embeddings = self.status_concept_encoder(
            torch.flatten(status, start_dim=-2)
        )

        history_position_embeddings = self.history_position_embedding(history)
        history_position_embeddings = history_position_embeddings.flatten(start_dim=-2)
        history_embeddings = self.history_concept_encoder(history_position_embeddings)

        # Concatenate the flattened CNN output with the concept vector
        if self.use_pixels:
            combined_input = torch.cat(
                (cnn_output, status_embeddings, history_embeddings), dim=-1
            )
        else:
            combined_input = torch.cat((status_embeddings, history_embeddings), dim=-1)

        # Pass the combined input through the FC network
        output = self.joint_encoder(combined_input)

        return output


import torch
from torch.nn.functional import mse_loss
from torchrl.objectives import ClipPPOLoss
import contextlib


class ClipPPOLossWithConceptLoss(ClipPPOLoss):
    """
    Extends ClipPPOLoss with the calculation of mean squared error loss
    between 'visual concepts' (ground truth) and 'concept_features' (model predictions).
    """

    def __init__(self, concept_coef=0.0, *args, **kwargs):
        super(ClipPPOLossWithConceptLoss, self).__init__(*args, **kwargs)

        self.loss_type = "focal_loss"
        self.concept_coef = concept_coef

    def calculate_concepts(self, tensordict):
        if "concept_features" in tensordict.keys():
            del tensordict["concept_features"]

        with (
            self.actor_network_params.to_module(self.actor_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            dist = self.actor_network.get_dist(tensordict)
        concept_features = tensordict["concept_features"]
        return concept_features

    def visual_concept_loss(self, tensordict):

        visual_concepts = tensordict.get("visual_concepts")
        concept_features = self.calculate_concepts(tensordict)

        # collapse time and batch dimensions

        visual_concepts = visual_concepts.flatten(start_dim=0, end_dim=1)
        concept_features = concept_features.flatten(start_dim=0, end_dim=1)

        # Flatten the last two dimensions of visual concepts
        # visual_concepts_flat = visual_concepts.flatten(start_dim=-2)

        # Calculate mean squared error loss
        vc_loss = 0
        accs = []
        for i in range(int(visual_concepts.shape[-1])):
            concept_target_amax = visual_concepts[:, i].long()
            concept_logits = concept_features[:, i, :]
            concept_logits_amax = torch.argmax(concept_logits, dim=1)

            n_samples_per_class = torch.bincount(
                concept_target_amax.cpu(), minlength=19
            )

            if self.loss_type == "focal_loss":
                loss = CB_Loss(
                    loss_type="focal_loss", samples_per_class=n_samples_per_class
                )
            else:
                loss = nn.CrossEntropyLoss()
            accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=19).to(
                concept_logits.device
            )

            acc = accuracy(concept_logits_amax, concept_target_amax)
            accs.append(acc)

            vc_loss += torch.mean(loss(concept_logits, concept_target_amax))

        average_acc = torch.mean(torch.stack(accs))
        # td_out["average_acc"] = average_acc
        # td_out.set("loss_visual_concept", vc_loss)
        # return td_out
        return average_acc, vc_loss / visual_concepts.shape[-1]

    def loss_critic(self, tensordict) -> torch.Tensor:
        # TODO: if the advantage is gathered by forward, this introduces an
        # overhead that we could easily reduce.
        if self.separate_losses:
            tensordict = tensordict.detach()
        try:
            target_return = tensordict.get(self.tensor_keys.value_target)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )

        with (
            self.critic_network_params.to_module(self.critic_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            state_value_td = self.critic_network(tensordict)

        try:
            state_value = state_value_td.get(self.tensor_keys.value)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                f"Make sure that the value_key passed to PPO is accurate."
            )

        with (
            self.critic_network_params.to_module(self.critic_network)
            if self.functional
            else contextlib.nullcontext()
        ):
            scaled_mse_head = self.critic_network._modules["module"][-1].module
            loss_value = scaled_mse_head.mse_loss(state_value, target_return)

        # loss_value = distance_loss(
        #     target_return,
        #     state_value,
        #     loss_function=self.loss_critic_type,
        # )

        return self.critic_coef * loss_value

    def forward(self, tensordict):
        # Call the original forward method of ClipPPOLoss to handle the usual loss calculations
        if self.concept_coef != 0.0:
            average_acc, vc_loss = self.visual_concept_loss(tensordict)
        td_out = super(ClipPPOLossWithConceptLoss, self).forward(tensordict)

        if self.concept_coef != 0.0:
            td_out["average_acc"] = average_acc
            td_out["loss_visual_concept"] = vc_loss

        return td_out
