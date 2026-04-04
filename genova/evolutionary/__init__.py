"""Evolutionary conservation and phylogenetic analysis tools.

Modules:
    multi_species: Multi-species training with shared embedding space.
    conservation: Conservation-aware predictions and weighted losses.
    transfer_learning: Cross-species transfer and domain adaptation.
"""

from genova.evolutionary.multi_species import (
    MultiSpeciesEncoder,
    SpeciesConfig,
    SpeciesEmbedding,
    HomologousAlignmentLoader,
)
from genova.evolutionary.conservation import (
    ConservationScorer,
    ConservationWeightedLoss,
    EvolutionaryRateEstimator,
)
from genova.evolutionary.transfer_learning import (
    CrossSpeciesTransferLearner,
    GradientReversalLayer,
    SpeciesDiscriminator,
)

__all__ = [
    "MultiSpeciesEncoder",
    "SpeciesConfig",
    "SpeciesEmbedding",
    "HomologousAlignmentLoader",
    "ConservationScorer",
    "ConservationWeightedLoss",
    "EvolutionaryRateEstimator",
    "CrossSpeciesTransferLearner",
    "GradientReversalLayer",
    "SpeciesDiscriminator",
]
