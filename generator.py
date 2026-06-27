import numpy as np
import networkx as nx


def create_random_graph(n_nodes=8, edge_prob=0.35, seed=42):

    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()

    for node in range(n_nodes):
        graph.add_node(node)


    for i in range(n_nodes - 1):
        cost = int(rng.integers(5, 25))
        graph.add_edge(i, i + 1, weight=cost)


    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and not graph.has_edge(i, j):
                if rng.random() < edge_prob:
                    cost = int(rng.integers(5, 30))
                    graph.add_edge(i, j, weight=cost)

    return graph


def create_true_od(n_nodes=8, max_demand=20, active_prob=0.4, seed=42):

    rng = np.random.default_rng(seed + 1)
    od = np.zeros((n_nodes, n_nodes), dtype=float)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and rng.random() < active_prob:
                od[i, j] = int(rng.integers(1, max_demand + 1))

    return od


def compute_shortest_paths(graph):

    paths = {}

    for source in graph.nodes:
        for target in graph.nodes:
            if source == target:
                continue

            try:
                path = nx.shortest_path(
                    graph,
                    source=source,
                    target=target,
                    weight="weight"
                )
                paths[(source, target)] = path
            except nx.NetworkXNoPath:
                paths[(source, target)] = None

    return paths


def generate_link_counts(graph, true_od, paths, noise_level=0, seed=42):

    rng = np.random.default_rng(seed + 2)

    edges = list(graph.edges())
    link_counts = {edge: 0.0 for edge in edges}

    n_nodes = true_od.shape[0]

    for i in range(n_nodes):
        for j in range(n_nodes):
            demand = true_od[i, j]

            if demand <= 0:
                continue

            path = paths.get((i, j))

            if path is None:
                continue

            for u, v in zip(path[:-1], path[1:]):
                link_counts[(u, v)] += demand


    if noise_level > 0:
        for edge in link_counts:
            noise = rng.integers(-noise_level, noise_level + 1)
            link_counts[edge] = max(0, link_counts[edge] + noise)

    return link_counts