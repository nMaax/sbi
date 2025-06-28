import torch
from torch import Tensor

from sbi.neural_nets.estimators.base import ConditionalVectorFieldEstimator


class MaskedConditionalVectorFieldEstimatorWrapper(ConditionalVectorFieldEstimator):
    def __init__(self, original_estimator, fixed_condition_mask, fixed_edge_mask):
        T, F = original_estimator.input_shape

        num_latent = int(torch.sum(fixed_condition_mask == 0).item())
        num_observed = int(torch.sum(fixed_condition_mask == 1).item())

        # Count number of latent and observed nodes
        self._new_input_shape = torch.Size((num_latent * F,))
        self._new_condition_shape = torch.Size((num_observed * F,))

        super().__init__(
            net=original_estimator.net,
            input_shape=self._new_input_shape,
            condition_shape=self._new_condition_shape,
            t_min=original_estimator.t_min,
            t_max=original_estimator.t_max,
        )

        self.SCORE_DEFINED = original_estimator.SCORE_DEFINED
        self.SDE_DEFINED = original_estimator.SDE_DEFINED
        self.MARGINALS_DEFINED = original_estimator.MARGINALS_DEFINED

        self._original_T = T
        self._original_F = F
        self._num_latent = num_latent
        self._num_observed = num_observed

        self._original_estimator = original_estimator

        # Ensure input_part and condition_part are on the same device
        device = next(original_estimator.net.parameters()).device
        self.register_buffer(
            "_fixed_condition_mask",
            fixed_condition_mask.to(device).clone().detach(),
        )
        self.register_buffer(
            "_fixed_edge_mask", fixed_edge_mask.to(device).clone().detach()
        )

        # Extract indices for latent (0) and observed (1) nodes
        # from the fixed_condition_mask
        self._latent_idx = (fixed_condition_mask == 0).nonzero(as_tuple=True)[0]
        self._observed_idx = (fixed_condition_mask == 1).nonzero(as_tuple=True)[0]

        # Get the mean/std for the latent nodes from the original estimator
        latent_mean_base_unflattened = original_estimator.mean_base[
            :, self._latent_idx, :
        ]
        latent_std_base_unflattened = original_estimator.std_base[
            :, self._latent_idx, :
        ]

        latent_mean_base_flattened = latent_mean_base_unflattened.flatten(start_dim=1)
        latent_std_base_flattened = latent_std_base_unflattened.flatten(start_dim=1)

        # Register these flattened buffers
        self.register_buffer("_mean_base", latent_mean_base_flattened.clone().detach())
        self.register_buffer("_std_base", latent_std_base_flattened.clone().detach())

    def forward(
        self, input: Tensor, condition: Tensor, time: Tensor, **kwargs
    ) -> Tensor:
        # Assemble full input from give input and condition
        # Take (B, T*F) and returns (B, T, F)
        full_inputs_tensor = self._assemble_full_inputs(input, condition)
        B = full_inputs_tensor.shape[0]
        expanded_cond_mask = self._fixed_condition_mask.unsqueeze(0).expand(B, -1)
        expanded_edge_mask = self._fixed_edge_mask.unsqueeze(0).expand(B, -1, -1)

        # Call the original masked estimator's forward method
        full_outputs = self._original_estimator.forward(
            input=full_inputs_tensor,
            time=time,
            condition_mask=expanded_cond_mask,
            edge_mask=expanded_edge_mask,
            **kwargs,
        )

        # Take B, T, F and return (B, num_latent*F) and (B, num_observed*F)
        latent_out, condition_out = self._disassemble_full_outputs(full_outputs)
        return latent_out

    def loss(
        self,
        input: Tensor,
        condition: Tensor,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError(
            "The loss method of the UnmaskedWrapper is not "
            "intended to be used directly. If you want to use "
            "this estimator for a different inference method, "
            "please use the original masked estimator "
            "or implement a suitable loss."
        )

    # -------------------------- ODE METHODS --------------------------

    def ode_fn(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        full_inputs_tensor = self._assemble_full_inputs(input, condition)  # (B, T, F)
        B = full_inputs_tensor.shape[0]
        expanded_cond_mask = self._fixed_condition_mask.unsqueeze(0).expand(B, -1)
        expanded_edge_mask = self._fixed_edge_mask.unsqueeze(0).expand(B, -1, -1)

        # original_estimator.ode_fn returns (B, T, F)
        full_outputs_ode = self._original_estimator.ode_fn(
            full_inputs_tensor,
            times,
            expanded_cond_mask,
            expanded_edge_mask,
        )
        # Disassemble and flatten the output
        latent_out, _ = self._disassemble_full_outputs(
            full_outputs_ode
        )  # Returns (B, num_latent*F)
        return latent_out

    # -------------------------- SDE METHODS --------------------------

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        # Assemble full input from give input and condition
        # input: (B, num_latent * F), condition: (B, num_observed * F)
        full_inputs_tensor = self._assemble_full_inputs(input, condition)

        # Call the original estimator's loss
        B = full_inputs_tensor.shape[0]
        expanded_cond_mask = self._fixed_condition_mask.unsqueeze(0).expand(B, -1)
        expanded_edge_mask = self._fixed_edge_mask.unsqueeze(0).expand(B, -1, -1)

        full_score_outputs = self._original_estimator.score(
            full_inputs_tensor,
            t,
            expanded_cond_mask,
            expanded_edge_mask,
        )

        # Take B, T, F and return (B, num_latent*F) and (B, num_observed*F)
        latent_score, _ = self._disassemble_full_outputs(full_score_outputs)
        # Returns (B, num_latent * F)
        return latent_score

    def mean_t_fn(self, times: Tensor) -> Tensor:
        return self._original_estimator.mean_t_fn(times)

    def std_fn(self, times: Tensor) -> Tensor:
        return self._original_estimator.std_fn(times)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return self._original_estimator.drift_fn(input, times)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return self._original_estimator.diffusion_fn(input, times)

    # ------------------------- UTILITIES ------------------------------

    def _assemble_full_inputs(self, input_part, condition_part):
        # Get batch shape and feature dimension
        B = input_part.shape[0]
        input_part_unflattened = input_part.reshape(
            B, self._num_latent, self._original_F
        )
        condition_part_unflattened = condition_part.reshape(
            -1, self._num_observed, self._original_F
        ).expand(B, self._num_observed, self._original_F)

        full_inputs = torch.zeros(
            B,
            self._original_T,
            self._original_F,
            dtype=input_part.dtype,
            device=input_part.device,
        )
        # Place unflattened parts into the correct positions
        full_inputs[:, self._latent_idx, :] = input_part_unflattened
        full_inputs[:, self._observed_idx, :] = condition_part_unflattened

        return full_inputs

    def _disassemble_full_outputs(self, full_outputs):
        latent_part_unflattened = full_outputs[
            :, self._latent_idx, :
        ]  # (B, num_latent, F)
        observed_part_unflattened = full_outputs[
            :, self._observed_idx, :
        ]  # (B, num_observed, F)

        latent_part = latent_part_unflattened.reshape(
            latent_part_unflattened.shape[0],
            -1,
            self._num_latent * self._original_F,
        )  # (B, ..., num_latent * F)
        observed_part = observed_part_unflattened.reshape(
            observed_part_unflattened.shape[0],
            -1,
            self._num_observed * self._original_F,
        )  # (B, ..., num_observed * F)

        return latent_part, observed_part
