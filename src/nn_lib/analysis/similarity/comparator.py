from typing import Iterable

import torch
from abc import ABC, abstractmethod


# TODO - think about how to handle methods, such as generalized CCA, that operate on N>=2
#  representational spaces at once rather than pairwise on x and y.


class Comparator(ABC):
    @abstractmethod
    def compare(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Quantify representational (dis)similarity between neural data x and y. Specific
        comparators are implemented by subclasses.
        """


class StreamingComparator(Comparator, ABC):
    def compare(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.streaming_compare([(x, y)])

    @abstractmethod
    def streaming_compare(
        self, xy_batches: Iterable[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Quantify representational (dis)similarity between batches of neural data x and y. The
        input `xy` will be iterated inside this function until some method-specific stopping
        condition is met such as maximum number of items or a variance threshold.

        Important: some StreamingComparator instances require multiple passes over the data. The
        iterable `xy` must therefore be *repeatable* in the sense that multiple calls to iter(xy)
        must return the same sequence of pairs of tensors. This is *not* the case for shuffled
        dataloaders. We also expect all batches to have the same first dimension.
        """
