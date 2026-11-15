"""Helpers shared by the tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def reordered_node_maps(
    node_maps: Sequence[npt.NDArray[np.integer]],
    node_order: npt.NDArray[np.intp],
    cell_order: npt.NDArray[np.intp],
) -> list[npt.NDArray[np.integer]]:
    """Put the node maps of a zone in the node numbering and cell order of a reference.

    Args:
        node_maps: The node map of each section of the zone, 0-based.
        node_order: Result of ``tecio.match_nodes`` for the zone.
        cell_order: Result of ``tecio.match_cells`` for the zone.

    Returns:
        The node maps, which compare exactly with those of the reference zone.
    """
    renumber = np.argsort(node_order)
    reordered = []
    offset = 0
    for node_map in node_maps:
        cells = cell_order[offset : offset + len(node_map)] - offset
        reordered.append(renumber[node_map][cells])
        offset += len(node_map)
    return reordered
