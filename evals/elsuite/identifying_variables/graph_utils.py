"""Utils for network graph related operations."""
from typing import Any, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np


def val_and_count_roots(
    nodes: List[str],
    np_rng: np.random.Generator,
    total_edges: Optional[int] = None,
    min_roots: Optional[int] = None,
) -> int:
    """
    Validates the parameters for the construction of a random forest via
    `gen_random_forest` and determines the min number of roots to use.

    A random forest following the constraints of `gen_random_forest` with
    N nodes will have
      - R <= N roots
      - E <= N - R edges
    If min_roots is not specified, then E <= N - 1, since R >= 1.
    """
    n_nodes = len(nodes)
    if min_roots is not None:
        assert min_roots <= n_nodes, "Total roots must be less than or equal to the number of nodes"
        if total_edges is not None:
            assert (
                0 <= total_edges <= n_nodes - min_roots
            ), "Total edges must be between 0 and the number of nodes minus the number of roots"
    else:
        if total_edges is None:
            min_roots = np_rng.integers(1, n_nodes + 1)
        else:
            assert (
                0 <= total_edges <= n_nodes - 1
            ), "Total edges must be between 0 and the number of nodes minus 1"
            # if total edges is specified, then we have an upper bound on R, R <= N - E
            max_roots = n_nodes - total_edges
            min_roots = np_rng.integers(1, max_roots + 1)

    return min_roots


def gen_random_forest_tree_size(
    nodes: List[str],
    tree_size: int,
    np_rng: Optional[np.random.Generator] = None,
) -> nx.DiGraph:
    """
    Builds a random forest, i.e. a Directed Acyclic Graph (DAG)
    with potentially more than one root.

    We enforce the following constraints for our purposes:
        1. No self connections
        2. No bi-directional connections
        3. No children with multiple parents
        4. At least one root node (no parents)
        5. No cycles

    We additionally allow the user to specify the size that at least one
    of the trees in the forest should be.

    Args:
        nodes: A list of node names to build the graph from
        tree_size: The number of nodes that at least one of the trees in the forest
        should have
        np_rng: A numpy random number generator
    """
    num_nodes = len(nodes)
    assert tree_size <= num_nodes, "Tree size must be less than or equal to the number of nodes"

    max_number_roots = num_nodes - tree_size + 1
    min_number_roots = 1  # 1 root is always reserved to the tree of size tree_size

    np_rng = np_rng or np.random.default_rng()

    num_roots = np_rng.integers(min_number_roots, max_number_roots + 1)
    roots = set(np_rng.choice(nodes, num_roots, replace=False).tolist())

    size_controlled_root = np_rng.choice(list(roots))
    size_controlled_tree_nodes = {size_controlled_root}

    shuffled_nodes = np_rng.permutation(nodes)

    graph_children = set()

    graph = nx.DiGraph()
    graph.add_nodes_from(shuffled_nodes)

    while len(size_controlled_tree_nodes) < tree_size:
        possible_children = [
            n for n in nodes if n not in size_controlled_tree_nodes and n not in roots
        ]
        child = np_rng.choice(possible_children)
        possible_parents = list(size_controlled_tree_nodes)
        parent = np_rng.choice(possible_parents)
        graph.add_edge(parent, child)
        size_controlled_tree_nodes.add(child)
        graph_children.add(child)

    remaining_nodes = set(nodes) - size_controlled_tree_nodes

    for node in remaining_nodes:
        possible_children = [
            n
            for n in remaining_nodes
            # avoid self connections
            if n != node and
            # avoid cycles and bi-directional conns -> ancestors can't be children
            n not in nx.ancestors(graph, node) and
            # avoid children with multiple parents
            n not in graph_children and
            # roots can't be children
            n not in roots
        ]
        num_edges = np_rng.integers(0, len(possible_children) + 1)
        children = np_rng.choice(possible_children, num_edges, replace=False).tolist()

        for child in children:
            graph.add_edge(node, child)
        graph_children.update(children)

    return graph


def gen_random_forest(
    nodes: List[str],
    total_edges: Optional[int] = None,
    min_roots: Optional[int] = None,
    np_rng: Optional[np.random.Generator] = None,
) -> nx.DiGraph:
    """
    Builds a random forest, i.e. a Directed Acyclic Graph (DAG)
    with potentially more than one root.

    We enforce the following constraints for our purposes:
        1. No self connections
        2. No bi-directional connections
        3. No children with multiple parents
        4. At least one root node (no parents)
        5. No cycles

    Args:
        nodes: A list of node names to build the graph from
        total_edges: The total number of edges in the graph. If None, will be random.
        min_roots: The minimum number of roots in the graph. If None, will be random.
    """
    np_rng = np_rng or np.random.default_rng()
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    min_roots = val_and_count_roots(nodes, np_rng, total_edges, min_roots)

    # the minimal set of roots, there may be more as we create the graph
    roots = set(np_rng.choice(nodes, min_roots, replace=False).tolist())

    graph_children = set()
    edge_count = 0

    shuffled_nodes = np_rng.permutation(nodes)

    for node in shuffled_nodes:
        possible_children = [
            n
            for n in nodes
            # avoid self connections
            if n != node and
            # avoid cycles and bi-directional conns -> ancestors can't be children
            n not in nx.ancestors(graph, node) and
            # avoid children with multiple parents
            n not in graph_children and
            # roots can't be children
            n not in roots
        ]

        if len(possible_children) == 0:
            continue

        if total_edges is not None:
            remaining_edges = total_edges - edge_count
            if remaining_edges <= 0:
                break
            num_edges = np_rng.integers(0, min(remaining_edges, len(possible_children)) + 1)
        else:
            num_edges = np_rng.integers(0, len(possible_children) + 1)

        children = np_rng.choice(possible_children, num_edges, replace=False).tolist()

        for child in children:
            graph.add_edge(node, child)
        graph_children.update(children)
        edge_count += num_edges

    if total_edges is not None and edge_count < total_edges:
        # If we didn't reach the total number of edges, try again
        return gen_random_forest(nodes, total_edges, min_roots, np_rng)

    return graph


def find_farthest_node(graph: nx.DiGraph, source: str) -> Tuple[str, int]:
    """
    Performs Breadth-First Search (BFS) to find the farthest node from the source node
    and the distance to that node. Distance is defined as the number of edges between
    the source node and the farthest node.
    """
    graph = graph.to_undirected()

    # Compute shortest path lengths from source to all other nodes
    path_lengths = nx.single_source_shortest_path_length(graph, source)

    # Find the farthest node
    farthest_node = max(path_lengths, key=path_lengths.get)
    max_distance = path_lengths[farthest_node]

    return farthest_node, max_distance


def find_graph_roots(graph: nx.DiGraph) -> Set[str]:
    """
    Finds the root nodes of a graph
    """
    return set([n for n, d in graph.in_degree() if d == 0])


def find_graph_trees(graph: nx.DiGraph) -> List[Set[str]]:
    """
    Finds the trees of a graph
    """
    return [{root, *nx.descendants(graph, root)} for root in find_graph_roots(graph)]


def find_connected_nodes_pair(
    graph: nx.DiGraph, np_rng: np.random.Generator
) -> Union[Tuple[Any, Any], None]:
    """
    Finds a pair of connected nodes in a graph
    If no such pair exists, returns None
    """
    connected_pair = tuple(np_rng.choice(list(graph.edges))) if graph.edges else None
    return connected_pair


def find_unconnected_nodes_pair(graph: nx.DiGraph) -> Union[Tuple[Any, Any], None]:
    """
    Finds a pair of unconnected nodes in a graph
    If no such pair exists, returns None
    """
    components = list(nx.connected_components(graph.to_undirected()))

    if len(components) > 1:
        return next(iter(components[0])), next(iter(components[1]))
    return None
