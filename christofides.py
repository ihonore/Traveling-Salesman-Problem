import networkx as nx
import random
import matplotlib.pyplot as plt
import timeit
from itertools import permutations

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

    return tour, tour_length

def measure_length_christofides(graph):
    # Step 1: Find a minimum spanning tree
    mst = nx.minimum_spanning_tree(graph)

    # Step 2: Find minimum-weight perfect matching on odd-degree vertices
    odd_vertices = [node for node, degree in mst.degree() if degree % 2 != 0]
    subgraph_odd = graph.subgraph(odd_vertices)
    min_weight_matching = nx.algorithms.matching.max_weight_matching(subgraph_odd)

    # Step 3: Combine the matching and the spanning tree to form a multigraph
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(min_weight_matching)

    # Step 4: Find an Eulerian circuit in the multigraph
    eulerian_circuit = list(nx.eulerian_circuit(multigraph))

    # Step 5: Create a Hamiltonian circuit from the Eulerian circuit
    visited = set()
    hamiltonian_circuit = [eulerian_circuit[0][0]]
    for edge in eulerian_circuit:
        if edge[1] not in visited:
            hamiltonian_circuit.append(edge[1])
            visited.add(edge[1])

    # Step 6: Calculate the tour length
    tour_length = sum(graph[hamiltonian_circuit[i]][hamiltonian_circuit[i + 1]]['weight'] for i in range(len(hamiltonian_circuit) - 1))

    return hamiltonian_circuit, tour_length

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

    return shortest_tour, shortest_length


def display_results(graph, shortest_paths, nn_tour, bf_tour, cf_tour):
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
    print(f"Nearest Neighbor Tour Length: {nearest_neighbor_length:.2f}")

    # Display Brute Force Tour
    print(f"\nBrute Force Tour: {bf_tour}")
    print(f"Brute Force Tour Length: {brute_force_length:.2f}")

    # Display Christofides Tour
    print(f"\nChristofides Tour: {cf_tour}")
    print(f"Christofides Tour Length: {christofides_length:.2f}")

    # Display adjacency matrix
    adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
    print("\nAdjacency Matrix:")
    print(adjacency_matrix)

    # Keep the plot window open until it's manually closed
    plt.show()

# Example usage
num_cities_range = range(3, 13)
max_distance = 10

for num_cities in num_cities_range:
    print(f"\nNumber of cities: {num_cities}")

    complete_graph = generate_random_graph(num_cities, max_distance)

    # Calculate shortest paths
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(complete_graph))

    # Nearest Neighbor Tour
    start_time = timeit.default_timer()
    nn_tour, nearest_neighbor_length = nearest_neighbor_tour(complete_graph)
    nn_execution_time = timeit.default_timer() - start_time


    # Brute Force Tour
    start_time = timeit.default_timer()
    bf_tour, brute_force_length = brute_force_tour(complete_graph)
    bf_execution_time = timeit.default_timer() - start_time

    # Christofides Tour
    start_time = timeit.default_timer()
    cf_tour, christofides_length = measure_length_christofides(complete_graph)
    cf_execution_time = timeit.default_timer() - start_time

    # Display results
    display_results(complete_graph, shortest_paths, nn_tour, bf_tour, cf_tour)
    print(f"\nNearest Neighbor Execution Time: {nn_execution_time:.6f} seconds")
    print(f"Brute Force Execution Time: {bf_execution_time:.6f} seconds")
    print(f"Christofides Execution Time: {cf_execution_time:.6f} seconds")
