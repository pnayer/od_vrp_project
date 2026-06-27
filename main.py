import os
import traceback
import numpy as np

from generator import (
    create_random_graph,
    create_true_od,
    compute_shortest_paths,
    generate_link_counts,
)

from Od_estimator import (
    build_assignment_matrix,
    link_counts_to_vector,
    estimate_od_matrix,
)

from metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_relative_total_error,
    reproduce_link_counts,
    calculate_link_count_error,
)

from vrp import (
    compute_demand_from_od,
    build_shortest_distance_matrix,
    nearest_neighbor_vrp,
    compute_route_cost,
    check_vrp_solution,
    improve_routes_with_2opt,
)


def main():
    seed = 42
    n_nodes = 8
    capacity = 40
    depot = 0


    # 1. Data generation

    graph = create_random_graph(
        n_nodes=n_nodes,
        edge_prob=0.35,
        seed=seed
    )

    distance_matrix = build_shortest_distance_matrix(graph)

    true_od = create_true_od(
        n_nodes=n_nodes,
        max_demand=20,
        active_prob=0.4,
        seed=seed
    )

    paths = compute_shortest_paths(graph)

    link_counts = generate_link_counts(
        graph=graph,
        true_od=true_od,
        paths=paths,
        noise_level=0,
        seed=seed
    )


    # 2. OD estimation

    P, edges, od_pairs = build_assignment_matrix(graph, paths)
    link_count_vector = link_counts_to_vector(link_counts, edges)

    est_od, est_od_vector, estimation_result = estimate_od_matrix(
        P=P,
        link_count_vector=link_count_vector,
        od_pairs=od_pairs,
        n_nodes=n_nodes
    )

    est_od = np.nan_to_num(est_od, nan=0.0, posinf=0.0, neginf=0.0)
    true_od = np.nan_to_num(true_od, nan=0.0, posinf=0.0, neginf=0.0)
    est_od_vector = np.nan_to_num(est_od_vector, nan=0.0, posinf=0.0, neginf=0.0)


    # 3. Metrics

    od_mae = calculate_mae(true_od, est_od)
    od_rmse = calculate_rmse(true_od, est_od)
    relative_total_error = calculate_relative_total_error(true_od, est_od)

    reproduced_counts = reproduce_link_counts(P, est_od_vector)

    link_mae, link_rmse = calculate_link_count_error(
        observed_counts=link_count_vector,
        reproduced_counts=reproduced_counts
    )


    # 4. VRP

    est_demand = compute_demand_from_od(est_od)
    true_demand = compute_demand_from_od(true_od)

    est_routes, est_loads = nearest_neighbor_vrp(
        distance_matrix=distance_matrix,
        demands=est_demand,
        capacity=capacity,
        depot=depot
    )

    true_routes, true_loads = nearest_neighbor_vrp(
        distance_matrix=distance_matrix,
        demands=true_demand,
        capacity=capacity,
        depot=depot
    )

    est_routes_before_2opt = [route.copy() for route in est_routes]
    true_routes_before_2opt = [route.copy() for route in true_routes]

    est_cost_before_2opt = compute_route_cost(est_routes_before_2opt, distance_matrix)
    true_cost_before_2opt = compute_route_cost(true_routes_before_2opt, distance_matrix)

    est_routes = improve_routes_with_2opt(est_routes, distance_matrix)
    true_routes = improve_routes_with_2opt(true_routes, distance_matrix)

    est_cost_after_2opt = compute_route_cost(est_routes, distance_matrix)
    true_cost_after_2opt = compute_route_cost(true_routes, distance_matrix)

    est_valid = check_vrp_solution(est_routes, est_loads, capacity, depot=depot)
    true_valid = check_vrp_solution(true_routes, true_loads, capacity, depot=depot)


    # 5. Console output

    print("========== Graph Edges ==========")
    for u, v, data in graph.edges(data=True):
        print(f"{u} -> {v}, cost = {data['weight']}")

    print("\n========== True OD Matrix ==========")
    print(true_od)

    print("\n========== Assignment Matrix P ==========")
    print("P shape:", P.shape)
    print("Number of edges:", len(edges))
    print("Number of OD pairs:", len(od_pairs))

    print("\n========== Estimated OD Matrix ==========")
    print(est_od)

    print("\n========== Estimation Info ==========")
    print("Optimization success:", estimation_result.success)
    print("Cost:", estimation_result.cost)
    print("Number of iterations:", estimation_result.nit)

    print("\n========== Evaluation Metrics ==========")
    print("OD MAE:", od_mae)
    print("OD RMSE:", od_rmse)
    print("Relative Total Error:", relative_total_error)
    print("Link Count MAE:", link_mae)
    print("Link Count RMSE:", link_rmse)

    print("\n========== VRP RESULTS ==========")
    print("Vehicle capacity:", capacity)
    print("Estimated demand:", est_demand)
    print("True demand:", true_demand)

    print("\nEstimated OD VRP cost before 2-opt:", est_cost_before_2opt)
    print("Estimated OD VRP cost after 2-opt:", est_cost_after_2opt)

    print("\nTrue OD VRP cost before 2-opt:", true_cost_before_2opt)
    print("True OD VRP cost after 2-opt:", true_cost_after_2opt)

    print("\nCost difference after 2-opt:", abs(est_cost_after_2opt - true_cost_after_2opt))

    print("\nEstimated VRP valid:", est_valid)
    print("True VRP valid:", true_valid)

    print("\nEstimated Routes after 2-opt:")
    for route, load in zip(est_routes, est_loads):
        print(route, "load =", round(load, 2))

    print("\nTrue Routes after 2-opt:")
    for route, load in zip(true_routes, true_loads):
        print(route, "load =", round(load, 2))


    # 6. Save results to txt

    results_path = os.path.join(os.getcwd(), "results.txt")

    with open(results_path, "w", encoding="utf-8") as file:
        file.write("OD Estimation Metrics\n")
        file.write("=====================\n")
        file.write(f"OD MAE: {od_mae}\n")
        file.write(f"OD RMSE: {od_rmse}\n")
        file.write(f"Relative Total Error: {relative_total_error}\n")
        file.write(f"Link Count MAE: {link_mae}\n")
        file.write(f"Link Count RMSE: {link_rmse}\n\n")

        file.write("VRP Results\n")
        file.write("===========\n")
        file.write(f"Vehicle capacity: {capacity}\n")
        file.write(f"Estimated OD VRP cost before 2-opt: {est_cost_before_2opt}\n")
        file.write(f"Estimated OD VRP cost after 2-opt: {est_cost_after_2opt}\n")
        file.write(f"True OD VRP cost before 2-opt: {true_cost_before_2opt}\n")
        file.write(f"True OD VRP cost after 2-opt: {true_cost_after_2opt}\n")
        file.write(f"Cost difference after 2-opt: {abs(est_cost_after_2opt - true_cost_after_2opt)}\n")
        file.write(f"Estimated VRP valid: {est_valid}\n")
        file.write(f"True VRP valid: {true_valid}\n\n")

        file.write("Estimated Routes after 2-opt\n")
        file.write("============================\n")
        for route, load in zip(est_routes, est_loads):
            file.write(f"{route}, load={round(load, 2)}\n")

        file.write("\nTrue Routes after 2-opt\n")
        file.write("=======================\n")
        for route, load in zip(true_routes, true_loads):
            file.write(f"{route}, load={round(load, 2)}\n")

    print("\nResults saved to:", results_path)



if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nERROR OCCURRED:")
        traceback.print_exc()