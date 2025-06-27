# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Literal, Optional, Union

from torch import Tensor
from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.joints.vector_field_joint import (
    VectorFieldJoint,
)
from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.trainers.vfpe.base_vf_inference import (
    MaskedVectorFieldEstimatorBuilder,
    MaskedVectorFieldInference,
)
from sbi.neural_nets.estimators import MaskedConditionalVectorFieldEstimator
from sbi.neural_nets.factory import simformer_nn


class Simformer(MaskedVectorFieldInference):
    """Simformer as in Gloeckler et al. (2024)

    Instead of sampling only from Posterior or Likelihood, Simformer is able
    to sample from any arbitrary joint conditional distribution.

    NOTE: Simformer does not support multi-round inference yet.
        Such API is still provided for coherence with sbi, but unused.

    NOTE: Simformer does not support prior in the sense of other sbi methods.
        Such API is still provided for coherence with sbi, but unused.
        The base distribution of the diffusion process is always
        a standard Gaussian (at t=T), this acts as an implicit "prior" in
        the latent space of the diffusion.
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        vf_estimator: Union[
            str,
            MaskedVectorFieldEstimatorBuilder,
        ] = "simformer",
        sde_type: Literal["vp", "ve", "subvp"] = "ve",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ):
        r"""Initialize Simformer.

        Args:
            prior: Prior distribution. (Ignored, for compatibility only)
            vf_estimator: Neural network architecture for the
                vector field estimator. Can be a string (e.g., `'simformer_standard'`
                for a basic Simformer block or `'simformer_dit'` for a DiT-style block)
                or a callable that implements the `MaskedVectorFieldEstimatorBuilder`
                protocol. If a callable, `__call__` must accept `inputs`,
                `conditioning_mask`, and `edge_mask`, and return
                a `MaskedConditionalVectorFieldEstimator`.
            sde_type: Type of SDE to use. Must be one of ['vp', 've', 'subvp'].
                Only ve (variance exploding) is supported by now.
            device: Device to run the training on.
            logging_level: Logging level for the training. Can be an integer or a
                string.
            summary_writer: Tensorboard summary writer.
            show_progress_bars: Whether to show progress bars during training.
            kwargs: Additional keyword arguments passed to the default builder if
                `score_estimator` is a string.

        References:
            - Gloeckler, Deistler, Weilbach, Wood, Macke.
                "All-in-one simulation-based inference.", ICML 2024
        """
        super().__init__(
            prior=prior,
            masked_vector_field_estimator_builder=vf_estimator,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            sde_type=sde_type,
            **kwargs,
        )

    def _build_default_nn_fn(self, **kwargs) -> MaskedVectorFieldEstimatorBuilder:
        net_type = kwargs.pop("vector_field_estimator_builder", "simformer")
        return simformer_nn(model=net_type, **kwargs)

    def build_arbitrary_joint(
        self,
        conditional_mask: Tensor,
        edge_mask: Tensor,
        vector_field_estimator: Optional[MaskedConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "sde",
        **kwargs,
    ) -> VectorFieldJoint:
        r"""Build an arbitrary conditional joint distribution from
        the vector field estimator.

        Args:
            vector_field_estimator: The vector field estimator that the posterior is
                based on. If `None`, use the latest vector field estimator that was
                trained.
            prior: Prior distribution (unused).
            sample_with: Method to use for sampling from the posterior. Can only be
                'sde' (default).
            **kwargs: Additional keyword arguments passed to
                `VectorFieldBasedPotential`.

        Returns:
            Conditional distribution of latent nodes given the observed nodes
            and the edge structure.  With `.sample()` and `.log_prob()` methods.
        """
        return self._build_arbitrary_joint(
            conditional_mask,
            edge_mask,
            vector_field_estimator,
            prior,
            sample_with=sample_with,
            **kwargs,
        )

    def build_posterior(
        self,
        condition_mask: Tensor,
        edge_mask: Tensor,
        vector_field_estimator: Optional[MaskedConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "sde",
        **kwargs,
    ) -> VectorFieldPosterior:
        return self._build_posterior(
            condition_mask,
            edge_mask,
            masked_vector_field_estimator=vector_field_estimator,
            prior=prior,
            sample_with=sample_with,
            **kwargs,
        )

    def build_likelihood(
        self,
        vector_field_estimator: Optional[MaskedConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "sde",
    ):
        raise NotImplementedError
