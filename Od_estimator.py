import numpy as np
from scipy.optimize import lsq_linear

def build_assignment_matrix(graph, paths):


    edges = list(graph.edges())
    od_pairs = []

    for source in graph.nodes:
        for target in graph.nodes:
            if source != target:
                od_pairs.append((source, target))

    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}
    od_to_index = {pair: idx for idx, pair in enumerate(od_pairs)}

    P = np.zeros((len(edges), len(od_pairs)))

    for pair in od_pairs:
        path = paths.get(pair)

        if path is None:
            continue

        col = od_to_index[pair]

        for u, v in zip(path[:-1], path[1:]):
            edge = (u, v)

            if edge in edge_to_index:
                row = edge_to_index[edge]
                P[row, col] = 1

    return P, edges, od_pairs


def link_counts_to_vector(link_counts, edges):

    y = np.zeros(len(edges))

    for idx, edge in enumerate(edges):
        y[idx] = link_counts.get(edge, 0.0)

    return y


def estimate_od_matrix(P, link_count_vector, od_pairs, n_nodes):


    result = lsq_linear(
        P,
        link_count_vector,
        bounds=(0, np.inf),
        lsmr_tol="auto",
        verbose=0
     )

    od_vector = result.x

    est_od = np.zeros((n_nodes, n_nodes))

    for value, (source, target) in zip(od_vector, od_pairs):
        est_od[source, target] = value

    return est_od, od_vector, result