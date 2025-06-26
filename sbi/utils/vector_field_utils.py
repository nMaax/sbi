from abc import ABC, abstractmethod

from torch import Tensor, nn


class VectorFieldNet(nn.Module, ABC):
    """Abstract base class for vector field estimation networks.

    Used by both flow matching and score matching approaches.
    """

    @abstractmethod
    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass to compute the vector field.

        Args:
            theta: Parameters (for FMPE) or state (for NPSE)
            x: Conditioning information
            t: Time parameter (scalar or batched)

        Returns:
            Vector field evaluation at the provided points
        """
        pass


class MaskedVectorFieldNet(nn.Module, ABC):
    """Abstract base class for vector field networks with masking support.

    Used by models that require conditioning and edge masking, such as Siformer.
    """

    @abstractmethod
    def forward(
        self, inputs: Tensor, t: Tensor, conditioning_mask: Tensor, edge_mask: Tensor
    ) -> Tensor:
        """Compute the vector field with support for conditioning and edge masks.

        Args:
            inputs: Input tensor containing parameters or states.
            t: Time parameter (scalar or batched).
            conditioning_mask: Mask indicating which parts of the input
                are conditioned (v. latent).
            edge_mask: Mask specifying dependencies between inputs
                (edges in the DAG).

        Returns:
            Tensor representing the vector field evaluated at the provided points.
        """
        pass
