from abc import ABC, abstractmethod

import torch

from nn_lib.analysis.similarity.utils import BatchIteratorFactory


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
        return self.streaming_compare(lambda: [(x, y)])

    @abstractmethod
    def streaming_compare(self, batch_iterator_factory: BatchIteratorFactory) -> torch.Tensor:
        """Quantify representational (dis)similarity between batches of neural data x and y. The
        input `xy` will be iterated inside this function until some method-specific stopping
        condition is met such as maximum number of items or a variance threshold.

        Important: some StreamingComparator instances require multiple passes over the data.
        Python does not provide a clean way to repeat the same iterator without storing
        everything in memory. We therefore adopt an 'iterator factory' pattern where the caller
        provides a (perhaps lambda) function which, when called with no arguments, produces
        something that we can iterate over.

        Subclasses requiring multiple passes over the data in the same order are responsible for
        calling assert_repeatable_iter_factory()
        """
