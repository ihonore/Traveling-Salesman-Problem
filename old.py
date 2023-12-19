#====================================

# import networkx as nx
# import random
# import matplotlib.pyplot as plt
# import numpy as np


# # def generate_random_graph(num_cities, max_distance):
# #     G = nx.Graph()

# #     # Add nodes representing cities
# #     G.add_nodes_from(range(1, num_cities + 1))

# #     # Add random edges with distances
# #     for i in range(1, num_cities + 1):
# #         for j in range(i + 1, num_cities + 1):
# #             distance = random.randint(1, max_distance)
# #             G.add_edge(i, j, weight=distance)

# #     return G

# def generate_random_graph(num_cities, max_distance):
#     G = nx.Graph()

#     # Add nodes representing cities with letter labels
#     G.add_nodes_from([chr(ord('A') + i) for i in range(num_cities)])

#     # Add random edges with distances
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

# def display_results(graph, shortest_paths):
#     # Draw the graph
#     pos = nx.spring_layout(graph)  # You can use different layout algorithms
#     nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black')
#     edge_labels = nx.get_edge_attributes(graph, 'weight')
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

#     # Display connectivity status
#     is_connected = is_graph_connected(graph)
#     print(f"\nGraph is connected: {is_connected}")

#     # Display shortest paths
#     plt.title("Graph Connectivity and Shortest Paths")

#     # Specify the matplotlib backend explicitly
#     plt.show(block=False)  # Use block=False to continue the script execution without blocking the console
#     plt.pause(0.1)  # Pause for a short time to allow the plot to be displayed

#     # Display adjacency matrix
#     adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
#     print("\nAdjacency Matrix:")
#     print(adjacency_matrix)

#     # Keep the plot window open until it's manually closed
#     plt.show()



# # Example usage
# num_cities = 5
# max_distance = 10

# graph = generate_random_graph(num_cities, max_distance)
# is_connected = is_graph_connected(graph)
# shortest_paths = calculate_shortest_paths(graph)

# display_results(graph, shortest_paths)

# =====================================

# import networkx as nx
# import random
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.spatial.distance import squareform, pdist
# from scipy.optimize import linear_sum_assignment
# import time

# # def generate_random_graph(num_cities, max_distance):
# #     G = nx.Graph()

# #     # Add nodes representing cities
# #     G.add_nodes_from(range(1, num_cities + 1))

# #     # Add random edges with distances
# #     for i in range(1, num_cities + 1):
# #         for j in range(i + 1, num_cities + 1):
# #             distance = random.randint(1, max_distance)
# #             G.add_edge(i, j, weight=distance)

# #     return G

# def generate_random_graph(num_cities, max_distance):
#     G = nx.Graph()

#     # Add nodes representing cities with letter labels
#     G.add_nodes_from([chr(ord('A') + i) for i in range(num_cities)])

#     # Add random edges with distances
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

# def nearest_neighbor(graph, start_city):
#     tour = [start_city]
#     current_city = start_city

#     while len(tour) < len(graph.nodes):
#         # Find the nearest unvisited city
#         nearest_city = min(set(graph.nodes) - set(tour), key=lambda city: graph[current_city][city]['weight'])
#         tour.append(nearest_city)
#         current_city = nearest_city

#     # Return to the starting city to complete the cycle
#     tour.append(start_city)

#     return tour

# def optimal_tsp_solution(graph):
#     # Create a mapping of original node labels to integer indices
#     node_mapping = {node: index for index, node in enumerate(graph.nodes)}

#     # Ensure that the nodes are labeled from 0 to n-1 for indexing
#     graph = nx.relabel_nodes(graph, node_mapping, copy=True)

#     print("Updated Node Labels and Indices:")
#     print({node: index for index, node in enumerate(graph.nodes)})

#     # Use node labels for indexing
#     distances = np.array([[graph[node1][node2]['weight'] for node2 in graph.nodes] for node1 in graph.nodes])

#     row_ind, col_ind = linear_sum_assignment(distances)

#     # Use node labels to reconstruct the optimal tour
#     optimal_tour = [node for node in np.argsort(row_ind)]

#     return optimal_tour





# def compare_tsp_solutions(graph, start_city):
#     # Nearest Neighbor
#     start_time = time.time()
#     nn_tour = nearest_neighbor(graph, start_city)
#     nn_duration = time.time() - start_time
#     nn_length = sum(graph[nn_tour[i]][nn_tour[i + 1]]['weight'] for i in range(len(nn_tour) - 1))

#     # Optimal TSP Solution
#     start_time = time.time()
#     optimal_tour = optimal_tsp_solution(graph)
#     optimal_duration = time.time() - start_time
#     optimal_length = sum(graph[optimal_tour[i]][optimal_tour[i + 1]]['weight'] for i in range(len(optimal_tour) - 1))

#     print("Nearest Neighbor Solution:")
#     print(f"Tour: {nn_tour}")
#     print(f"Length: {nn_length}")
#     print(f"Execution Time: {nn_duration} seconds\n")

#     print("Optimal TSP Solution:")
#     print(f"Tour: {optimal_tour}")
#     print(f"Length: {optimal_length}")
#     print(f"Execution Time: {optimal_duration} seconds")

# # Example usage
# num_cities = 10
# max_distance = 20

# graph = generate_random_graph(num_cities, max_distance)
# is_connected = is_graph_connected(graph)
# shortest_paths = calculate_shortest_paths(graph)

# compare_tsp_solutions(graph, start_city='A')