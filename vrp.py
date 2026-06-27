import numpy as np
import networkx as nx


def compute_demand_from_od(od_matrix):

    demand = np.sum(od_matrix, axis=0)
    demand = np.nan_to_num(demand, nan=0.0, posinf=0.0, neginf=0.0)
    demand = np.maximum(demand, 0.0)
    return demand


def build_shortest_distance_matrix(graph):

    nodes = list(graph.nodes())
    n = len(nodes)

    distance_matrix = np.full((n, n), np.inf)

    for i in nodes:
        distance_matrix[i][i] = 0

    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))

    for source, targets in lengths.items():
        for target, dist in targets.items():
            distance_matrix[source][target] = dist

    finite_values = distance_matrix[np.isfinite(distance_matrix)]
    big_cost = 9999 if len(finite_values) == 0 else np.max(finite_values) * 10
    distance_matrix[~np.isfinite(distance_matrix)] = big_cost

    return distance_matrix


def nearest_neighbor_vrp(distance_matrix, demands, capacity, depot=0):

    demands = np.array(demands, dtype=float)
    remaining = np.nan_to_num(demands.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    remaining = np.maximum(remaining, 0.0)
    remaining[depot] = 0.0

    routes = []
    route_loads = []

    safety_counter = 0
    max_iterations = 10000

    while np.sum(remaining) > 1e-6:
        safety_counter += 1

        if safety_counter > max_iterations:
            raise RuntimeError("VRP loop stopped. Check demands and capacity.")

        route = [depot]
        load = 0.0
        current = depot

        while load < capacity - 1e-6 and np.sum(remaining) > 1e-6:
            best_node = None
            best_dist = float("inf")

            for node in range(len(remaining)):
                if node == depot:
                    continue

                if remaining[node] <= 1e-6:
                    continue

                dist = distance_matrix[current][node]

                if dist < best_dist:
                    best_dist = dist
                    best_node = node

            if best_node is None:
                break

            available_capacity = capacity - load
            delivered_amount = min(remaining[best_node], available_capacity)

            route.append(best_node)
            load += delivered_amount
            remaining[best_node] -= delivered_amount
            current = best_node

            if load >= capacity - 1e-6:
                break

        route.append(depot)

        if len(route) <= 2 and load <= 1e-6:
            raise RuntimeError("No feasible VRP move found.")

        routes.append(route)
        route_loads.append(load)

    return routes, route_loads


def compute_route_cost(routes, distance_matrix):

    total_cost = 0.0

    for route in routes:
        for i in range(len(route) - 1):
            total_cost += distance_matrix[route[i]][route[i + 1]]

    return total_cost


def check_vrp_solution(routes, route_loads, capacity, depot=0):

    starts_and_ends_ok = all(route[0] == depot and route[-1] == depot for route in routes)
    capacity_ok = all(load <= capacity + 1e-6 for load in route_loads)

    return starts_and_ends_ok and capacity_ok


def two_opt_route(route, distance_matrix):

    if len(route) <= 4:
        return route

    best_route = route.copy()
    best_cost = compute_route_cost([best_route], distance_matrix)

    improved = True

    while improved:
        improved = False

        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue

                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                new_cost = compute_route_cost([new_route], distance_matrix)

                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True

    return best_route


def improve_routes_with_2opt(routes, distance_matrix):
    
    improved_routes = []

    for route in routes:
        improved_routes.append(two_opt_route(route, distance_matrix))

    return improved_routes