from __future__ import annotations

import abc
from collections import defaultdict
from collections.abc import Sequence as SequenceAbc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aizynthfinder.chem import TreeMolecule
from aizynthfinder.context.stock import StockException
from aizynthfinder.reactiontree import ReactionTree
from aizynthfinder.search.mcts import MctsNode
from aizynthfinder.utils.exceptions import ScorerException
from aizynthfinder.context.scoring.scorers import Scorer, FractionInStockScorer, MaxTransformScorerer

if TYPE_CHECKING:
    from aizynthfinder.chem import FixedRetroReaction, Molecule, RetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.utils.type_utils import (
        Iterable,
        Optional,
        Sequence,
        StrDict,
        Tuple,
        TypeVar,
        Union,
    )

    _Scoreable = TypeVar("_Scoreable", MctsNode, ReactionTree)
    _Scoreables = Sequence[_Scoreable]
    _ScorerItemType = Union[_Scoreables, _Scoreable]


class PrecursorScorer(Scorer):
    """Class for scoring nodes based on the state score"""

    scorer_name = "precursor score"

    def __init__(
            self, config: Configuration, scaler_params: Optional[StrDict] = None,
            **kwargs) -> None:
        super().__init__(config, scaler_params)
        # This is necessary because config should not be optional for this scorer
        self._config: Configuration = config
        print(f"Initializing PrecursorScorer with {kwargs}")
        self._transform_scorer = MaxTransformScorerer(
            config,
            scaler_params={"name": "squash", "slope": -1, "yoffset": 0, "xoffset": 4},
        )
        self._in_stock_scorer = FractionInStockScorer(config)

    def _score(self, item: _Scoreable) -> float:
        in_stock_fraction = self._in_stock_scorer(item)
        max_transform = self._transform_scorer(item)
        # A scorer can return a list of float if the item is a list of trees/nodes,
        # but that is not the case here. However this is needed because of mypy
        assert isinstance(in_stock_fraction, float) and isinstance(max_transform, float)
        return 0.95 * in_stock_fraction + 0.05 * max_transform

    def _score_node(self, node: MctsNode) -> float:
        return self._score(node)

    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        return self._score(tree)
