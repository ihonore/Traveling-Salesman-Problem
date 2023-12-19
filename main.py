# import networkx as nx
# import random
# import matplotlib.pyplot as plt
# import numpy as np
# from itertools import permutations
# import time

# def generate_random_graph(num_cities, max_distance):
#     G = nx.Graph()
#     G.add_nodes_from([chr(ord('A') + i) for i in range(num_cities)])
#     for i in range(num_cities):
#         for j in range(i + 1, num_cities):
#             distance = random.randint(1, max_distance)
#             G.add_edge(chr(ord('A') + i), chr(ord('A') + j), weight=distance)
#     return G

# def is_graph_connected(graph):
#     connected = nx.is_connected(graph)
#     return connected

# def calculate_shortest_paths(graph):
#     shortest_paths = dict(nx.all_pairs_dijkstra_path_length(graph))
#     return shortest_paths

# def nearest_neighbor_tour(graph):
#     tour = []
#     current_city = list(graph.nodes())[0]
    
#     while len(tour) < len(graph.nodes()) - 1:
#         tour.append(current_city)
#         neighbors = list(graph.neighbors(current_city))
#         neighbors = [city for city in neighbors if city not in tour]

#         if not neighbors:
#             break  # No unvisited neighbors

#         nearest_neighbor = min(neighbors, key=lambda city: graph[current_city][city]['weight'])
#         current_city = nearest_neighbor

#     tour.append(tour[0])
#     tour_length = sum(graph[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1))

#     return tour, tour_length

# def brute_force_tour(graph):
#     cities = list(graph.nodes())
#     shortest_tour = None
#     shortest_length = float('inf')

#     for perm in permutations(cities):
#         perm += (perm[0],)
#         tour_length = sum(graph[perm[i]][perm[i + 1]]['weight'] for i in range(len(perm) - 1))

#         if tour_length < shortest_length:
#             shortest_length = tour_length
#             shortest_tour = perm

#     return shortest_tour, shortest_length

# def display_results(graph, shortest_paths, nn_tour, bf_tour):
#     # Draw the graph
#     pos = nx.spring_layout(graph)
#     nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black')
#     edge_labels = nx.get_edge_attributes(graph, 'weight')
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
#     plt.title("Graph Connectivity and Shortest Paths")

#     # Display connectivity status
#     is_connected = is_graph_connected(graph)
#     print(f"\nGraph is connected: {is_connected}")

#     # Display shortest paths
#     print("\nShortest Paths:")
#     for source, paths in shortest_paths.items():
#         for target, length in paths.items():
#             print(f"Shortest path from {source} to {target}: {length}")

#     # Display Nearest Neighbor Tour
#     print(f"\nNearest Neighbor Tour: {nn_tour}")
#     print(f"Nearest Neighbor Tour Length: {nearest_neighbor_length:.2f}")

#     # Display Brute Force Tour
#     print(f"\nBrute Force Tour: {bf_tour}")
#     print(f"Brute Force Tour Length: {brute_force_length:.2f}")

#     # Display adjacency matrix
#     adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
#     print("\nAdjacency Matrix:")
#     print(adjacency_matrix)

#     # Keep the plot window open until it's manually closed
#     plt.show()

# # Example usage
# num_cities_range = range(4,7)
# max_distance = 10

# for num_cities in num_cities_range:
#     print(f"\nNumber of cities: {num_cities}")

#     complete_graph = generate_random_graph(num_cities, max_distance)

#     # Calculate shortest paths
#     shortest_paths = calculate_shortest_paths(complete_graph)

#     # Nearest Neighbor Tour
#     start_time = time.time()
#     nn_tour, nearest_neighbor_length = nearest_neighbor_tour(complete_graph)
#     nn_execution_time = time.time() - start_time

#     # Brute Force Tour
#     start_time = time.time()
#     bf_tour, brute_force_length = brute_force_tour(complete_graph)
#     bf_execution_time = time.time() - start_time

#     # Display results
#     display_results(complete_graph, shortest_paths, nn_tour, bf_tour)
#     print(f"\nNearest Neighbor Execution Time: {nn_execution_time:.6f} seconds")
#     print(f"Brute Force Execution Time: {bf_execution_time:.6f} seconds")


import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations
import time

def generate_random_graph(num_cities, max_distance):
    G = nx.Graph()
    G.add_nodes_from([chr(ord('A') + i) for i in range(num_cities)])
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = random.randint(1, max_distance)
            G.add_edge(chr(ord('A') + i), chr(ord('A') + j), weight=distance)
    return G

def is_graph_connected(graph):
    connected = nx.is_connected(graph)
    return connected

def calculate_shortest_paths(graph):
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(graph))
    return shortest_paths

def nearest_neighbor_tour(graph):
    tour = []
    current_city = list(graph.nodes())[0]
    
    while len(tour) < len(graph.nodes()) - 1:
        tour.append(current_city)
        neighbors = list(graph.neighbors(current_city))
        neighbors = [city for city in neighbors if city not in tour]

        if not neighbors:
            break  # No unvisited neighbors

        nearest_neighbor = min(neighbors, key=lambda city: graph[current_city][city]['weight'])
        current_city = nearest_neighbor

    tour.append(tour[0])
    tour_length = sum(graph[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1))

    return tour, tour_length, "O(n^2)"

def brute_force_tour(graph):
    cities = list(graph.nodes())
    shortest_tour = None
    shortest_length = float('inf')

    for perm in permutations(cities):
        perm += (perm[0],)
        tour_length = sum(graph[perm[i]][perm[i + 1]]['weight'] for i in range(len(perm) - 1))

        if tour_length < shortest_length:
            shortest_length = tour_length
            shortest_tour = perm

    return shortest_tour, shortest_length, "O(n!)"

def display_results(graph, shortest_paths, nn_tour, nn_length, nn_time_complexity, bf_tour, bf_length, bf_time_complexity):
    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black')
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title("Graph Connectivity and Shortest Paths")

    # Display connectivity status
    is_connected = is_graph_connected(graph)
    print(f"\nGraph is connected: {is_connected}")

    # Display shortest paths
    print("\nShortest Paths:")
    for source, paths in shortest_paths.items():
        for target, length in paths.items():
            print(f"Shortest path from {source} to {target}: {length}")

    # Display Nearest Neighbor Tour
    print(f"\nNearest Neighbor Tour: {nn_tour}")
    print(f"Nearest Neighbor Tour Length: {nn_length:.2f}")
    print(f"Nearest Neighbor Time Complexity: {nn_time_complexity}")

    # Display Brute Force Tour
    print(f"\nBrute Force Tour: {bf_tour}")
    print(f"Brute Force Tour Length: {bf_length:.2f}")
    print(f"Brute Force Time Complexity: {bf_time_complexity}")

    # Display adjacency matrix
    adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
    print("\nAdjacency Matrix:")
    print(adjacency_matrix)

    # Keep the plot window open until it's manually closed
    plt.show()

# Example usage
num_cities_range = range(4, 7)
max_distance = 10

for num_cities in num_cities_range:
    print(f"\nNumber of cities: {num_cities}")

    complete_graph = generate_random_graph(num_cities, max_distance)

    # Calculate shortest paths
    shortest_paths = calculate_shortest_paths(complete_graph)

    # Nearest Neighbor Tour
    start_time = time.time()
    nn_tour, nearest_neighbor_length, nn_time_complexity = nearest_neighbor_tour(complete_graph)
    nn_execution_time = time.time() - start_time

    # Brute Force Tour
    start_time = time.time()
    bf_tour, brute_force_length, bf_time_complexity = brute_force_tour(complete_graph)
    bf_execution_time = time.time() - start_time

    # Display results
    display_results(complete_graph, shortest_paths, nn_tour, nearest_neighbor_length, nn_time_complexity, bf_tour, brute_force_length, bf_time_complexity)
    print(f"\nNearest Neighbor Execution Time: {nn_execution_time:.6f} seconds")
    print(f"Brute Force Execution Time: {bf_execution_time:.6f} seconds")
