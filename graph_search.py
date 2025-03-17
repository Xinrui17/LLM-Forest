import pandas as pd
import networkx as nx
import numpy as np
import random
import json
from collections import Counter
import time
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_neighbors_from_txt(file_path):
    round_neighbors = []
    current_round_neighbors = {}
    current_round = None
    
    with open(file_path, 'r') as file:
        for line in file:
            if 'Neighbors found in Round' in line:
                if current_round_neighbors:
                    round_neighbors.append(current_round_neighbors)
                    current_round_neighbors = {}
                current_round = int(line.strip().split()[-1][:-1])  # Extract round number
            elif line.strip():
                parts = line.split(":")
                patient_id = int(parts[0].split()[1])
                neighbors = eval(parts[1].strip())
                current_round_neighbors[patient_id] = neighbors
        # Append the last round
        if current_round_neighbors:
            round_neighbors.append(current_round_neighbors)
    
    return round_neighbors


def dynamic_binning(data, feature, num_bins=10, min_patients_per_bin=5):
    """
    Dynamically bins a feature to ensure each bin has at least min_patients_per_bin patients.
    Handles cases where distinct values are fewer than the number of bins.
    """
    unique_values = sorted(data[feature].dropna().unique())
    if len(unique_values) <= num_bins:
        bins_assigned = pd.cut(data[feature], bins=len(unique_values), labels=False, include_lowest=True)
        bin_edges = unique_values + [unique_values[-1] + 1]  # Extend to cover the last value
        return bins_assigned, bin_edges

    _, bin_edges = pd.cut(data[feature], bins=num_bins, retbins=True, labels=False, include_lowest=True)
    bins_assigned = pd.cut(data[feature], bins=bin_edges, labels=False, include_lowest=True)
    bin_counts = bins_assigned.value_counts().sort_index()

    # Adjust bins dynamically
    max_iterations = 100
    iterations = 0

    while bin_counts.min() < min_patients_per_bin:
        if iterations >= max_iterations:
            print(f"Warning: Max iterations reached for {feature}.")
            break

        new_bin_edges = [bin_edges[0]]  # Always include the first bin edge

        i = 0
        while i < len(bin_counts) - 1:  # Iterate through bins
            current_count = bin_counts.iloc[i]
            next_count = bin_counts.iloc[i + 1]

            if current_count < min_patients_per_bin:
                merged_count = current_count + next_count
                if merged_count >= min_patients_per_bin:
                    bin_counts.iloc[i + 1] = merged_count
                    bin_counts.iloc[i] = 0  # Mark the current bin as merged
                    new_bin_edges.append(bin_edges[i + 2])  # Skip the next edge
                    i += 2
                else:
                    bin_counts.iloc[i + 1] = merged_count
                    bin_counts.iloc[i] = 0
                    i += 1
            else:
                new_bin_edges.append(bin_edges[i + 1])
                i += 1

        if bin_counts.iloc[-1] < min_patients_per_bin:
            bin_counts.iloc[-2] += bin_counts.iloc[-1]
            bin_counts.iloc[-1] = 0
            new_bin_edges = new_bin_edges[:-1]  # Remove the second-to-last edge

        new_bin_edges.append(bin_edges[-1])
        new_bin_edges = sorted(set(new_bin_edges))
        bin_edges = new_bin_edges
        bins_assigned = pd.cut(data[feature], bins=bin_edges, labels=False, include_lowest=True)
        bin_counts = bins_assigned.value_counts().sort_index()
        iterations += 1

    return bins_assigned, bin_edges


def create_bipartite_graph_for_binned(binned_data, data, feature, bin_edges):
    G = nx.Graph()
    patients = data.index.tolist()
    bin_edges = np.array(bin_edges)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_nodes = [f'{feature}_bin_{i}' for i in range(len(bin_centers))]

    G.add_nodes_from(patients, bipartite=0)
    G.add_nodes_from(bin_nodes, bipartite=1)

    # Connect each patient to their respective bin
    for patient in patients:
        bin_index = binned_data.at[patient, feature]
        if pd.notna(bin_index):
            bin_node = f'{feature}_bin_{bin_index}'
            bin_center = bin_centers[int(bin_index)]
            value = data.at[patient, feature]

            # Calculate the relative distance
            if bin_center != 0:
                distance_to_center = abs(value - bin_center) / bin_center
            else:
                distance_to_center = abs(value - bin_center)

            # Calculate weight
            weight = 1 / (2 + distance_to_center)
            G.add_edge(patient, bin_node, weight=weight)

    return G


def bin_continuous_features(data, num_bins=10, min_patients_per_bin=4):
    """
    Bins all continuous features dynamically in the dataset.
    Handles cases with fewer distinct values than the number of bins.
    """
    binned_data = data.copy()
    bin_edges_dict = {}

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            bins_assigned, bin_edges = dynamic_binning(data, col, num_bins=num_bins, min_patients_per_bin=min_patients_per_bin)
            binned_data[col] = bins_assigned
            bin_edges_dict[col] = bin_edges

    return binned_data, bin_edges_dict

    
def group_graphs_randomly(graphs, group_size=3):
    remaining_features = list(graphs.keys())
    graph_groups = []

    while len(remaining_features) >= group_size:
        group = random.sample(remaining_features, group_size)
        graph_groups.append(group)
        for g in group:
            remaining_features.remove(g)

    # Handle any remaining graphs (less than group_size)
    if len(remaining_features) > 0:
        graph_groups.append(remaining_features)

    return graph_groups

# Function to randomly pair graphs for merging
def pair_graphs_randomly(graphs):
    remaining_features = list(graphs.keys())
    graph_pairs = []

    while len(remaining_features) > 1:
        pair = random.sample(remaining_features, 2)
        graph_pairs.append(pair)
        remaining_features.remove(pair[0])
        remaining_features.remove(pair[1])

    if len(remaining_features) == 1:
        graph_pairs.append([remaining_features[0]])

    return graph_pairs

def merge_three_graphs_by_patient_connections(G1, G2, G3, threshold=3):
    merged_graph = nx.Graph()

    patients = [n for n, d in G1.nodes(data=True) if d.get('bipartite') == 0]
    values1 = [n for n, d in G1.nodes(data=True) if d.get('bipartite') == 1]
    values2 = [n for n, d in G2.nodes(data=True) if d.get('bipartite') == 1]
    values3 = [n for n, d in G3.nodes(data=True) if d.get('bipartite') == 1]

    # Calculate common patients between every pair of value nodes across three graphs
    value_pairs_common_patients = []
    for v1 in values1:
        for v2 in values2:
            for v3 in values3:
                common_patients = set(G1.neighbors(v1)) & set(G2.neighbors(v2)) & set(G3.neighbors(v3))
                if len(common_patients) >= threshold:
                    value_pairs_common_patients.append((v1, v2, v3, len(common_patients)))

    value_pairs_common_patients.sort(key=lambda x: x[3], reverse=True)

    # Merge nodes based on common patient connections
    merged_values = set()
    for v1, v2, v3, common_count in value_pairs_common_patients:
        if v1 in merged_values or v2 in merged_values or v3 in merged_values:
            continue  # Skip if any value has already been merged

        merged_graph.add_node((v1, v2, v3), bipartite=1)  
        merged_values.add(v1)
        merged_values.add(v2)
        merged_values.add(v3)

        # Add edges between patients and merged value nodes
    for patient in patients:
        merged_graph.add_node(patient, bipartite=0)
        for v1, v2, v3, _ in value_pairs_common_patients:
            weight1 = G1[patient][v1]['weight'] if G1.has_edge(patient, v1) else 0
            weight2 = G2[patient][v2]['weight'] if G2.has_edge(patient, v2) else 0
            weight3 = G3[patient][v3]['weight'] if G3.has_edge(patient, v3) else 0
            if weight1 > 0 or weight2 > 0 or weight3 > 0:
                if merged_graph.has_node((v1, v2, v3)):
                    merged_graph.add_edge(patient, (v1, v2, v3), weight=weight1 + weight2 + weight3)

    # Add remaining unmerged value nodes to the graph
    for v in set(values1 + values2 + values3) - merged_values:
        bipartite_value = 1 if v in values1 or v in values2 or v in values3 else 0  # Determine if it's a patient or value node
        merged_graph.add_node(v, bipartite=bipartite_value)
        for patient in G1.neighbors(v) if v in values1 else (G2.neighbors(v) if v in values2 else G3.neighbors(v)):
            weight = G1[patient][v]['weight'] if G1.has_edge(patient, v) else (G2[patient][v]['weight'] if G2.has_edge(patient, v) else G3[patient][v]['weight'])
            merged_graph.add_edge(patient, v, weight=weight)

    return merged_graph

# Function to merge two graphs by computing common patients between value nodes
def merge_two_graphs_by_patient_connections(G1, G2, threshold=3):
    merged_graph = nx.Graph()

    patients = [n for n, d in G1.nodes(data=True) if d.get('bipartite') == 0]
    values1 = [n for n, d in G1.nodes(data=True) if d.get('bipartite') == 1]
    values2 = [n for n, d in G2.nodes(data=True) if d.get('bipartite') == 1]
    
    # Calculate common patients between every pair of value nodes across three graphs
    value_pairs_common_patients = []
    for v1 in values1:
        for v2 in values2:

            common_patients = set(G1.neighbors(v1)) & set(G2.neighbors(v2))
            if len(common_patients) >= threshold:
                value_pairs_common_patients.append((v1, v2, len(common_patients)))

    # Sort value pairs by the number of common patients (descending order)
    value_pairs_common_patients.sort(key=lambda x: x[3], reverse=True)

    # Merge nodes based on common patient connections
    merged_values = set()
    for v1, v2, common_count in value_pairs_common_patients:
        if v1 in merged_values or v2 in merged_values:
            continue  # Skip if any value has already been merged

        merged_graph.add_node((v1, v2), bipartite=1)  
        merged_values.add(v1)
        merged_values.add(v2)

        # Add edges between patients and merged value nodes
    for patient in patients:
        merged_graph.add_node(patient, bipartite=0)
        for v1, v2, _ in value_pairs_common_patients:
            weight1 = G1[patient][v1]['weight'] if G1.has_edge(patient, v1) else 0
            weight2 = G2[patient][v2]['weight'] if G2.has_edge(patient, v2) else 0
            if weight1 > 0 or weight2 > 0:
                if merged_graph.has_node((v1, v2)):
                    merged_graph.add_edge(patient, (v1, v2), weight=weight1 + weight2)

    # Add remaining unmerged value nodes to the graph
    for v in set(values1 + values2 ) - merged_values:
        bipartite_value = 1 if v in values1 or v in values2 else 0  # Determine if it's a patient or value node
        merged_graph.add_node(v, bipartite=bipartite_value)
        for patient in G1.neighbors(v) if v in values1 else G2.neighbors(v):
            weight = G1[patient][v]['weight'] if G1.has_edge(patient, v) else G2[patient][v]['weight']
            merged_graph.add_edge(patient, v, weight=weight)

    return merged_graph


def merge_and_process_graphs(graphs, group_size=2):
    graph_groups = group_graphs_randomly(graphs, group_size=group_size)
    merged_graphs = []

    for group in graph_groups:
        if len(group) == 3:
            g1, g2, g3 = group
            merged_graph = merge_three_graphs_by_patient_connections(graphs[g1], graphs[g2], graphs[g3], threshold=10)
            merged_graphs.append(merged_graph)
        elif len(group) == 2:
            g1, g2 = group
            merged_graph = merge_two_graphs_by_patient_connections(graphs[g1], graphs[g2], threshold=5)
            merged_graphs.append(merged_graph)
        else:
            merged_graphs.append(graphs[group[0]])

    return merged_graphs

def list_depth(lst):
    if isinstance(lst, list):
        return 1 + max(list_depth(item) for item in lst) if lst else 1
    return 0

def flatten_node(node):
    if isinstance(node, tuple):
        return '_'.join(map(str, node))
    return node

def weighted_random_walk(graph, start_node, steps=5):
    if start_node not in graph:
        raise ValueError(f"The node {start_node} is not present in the graph.")
    
    current_node = start_node
    for _ in range(steps):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break
        weights = np.array([graph[current_node][neighbor]['weight'] for neighbor in neighbors])
        sigmoid_weights = sigmoid(weights)
        probabilities = sigmoid_weights / sigmoid_weights.sum()
        neighbors_flat = [flatten_node(neighbor) for neighbor in neighbors]
        next_node_flat = np.random.choice(neighbors_flat, p=probabilities)
        next_node = neighbors[neighbors_flat.index(next_node_flat)]
        current_node = next_node
        yield current_node

# Function to get top neighbors by weight using random walk
def top_neighbors_by_weight(graph, start_node, steps=10, max_neighbors=10, num_walks=50, feature_value_count=1):
    neighbors_weights = Counter()
    for _ in range(num_walks):
        for neighbor in weighted_random_walk(graph, start_node, steps):
            neighbors_weights[neighbor] += 1

    sorted_neighbors = neighbors_weights.most_common(max_neighbors)
    top_patient_nodes_with_weights = {
        neighbor: neighbors_weights[neighbor] for neighbor, _ in sorted_neighbors if graph.nodes[neighbor].get('bipartite') == 0
    }
    
    return top_patient_nodes_with_weights

def save_neighbors_to_txt(all_neighbors, file_path='house_neighbors_list.txt'):
    with open(file_path, 'w') as file:
        for round_num, neighbors in enumerate(all_neighbors, 1):
            file.write(f"Neighbors found in Round {round_num}:\n")
            for patient, neighbor_dict in neighbors.items():
                file.write(f"Patient {patient}: {list(neighbor_dict.keys())}\n")
            file.write("\n")

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def update_patient_records_with_neighbors(patients_data, final_neighbors, round_num):
    for patient in patients_data:
        patient_id = patient['user']
        if patient_id in final_neighbors:
            neighbor_records = []
            i = 1
            for neighbor_id in final_neighbors[patient_id].keys():
                neighbor_data = next((item for item in patients_data if item['user'] == neighbor_id), None)
                if neighbor_data:
                    neighbor_record = neighbor_data.get('description', 'No record available')
                    neighbor_records.append(f"Similar house records {i}: {neighbor_record}")
                    i += 1

            original_record = patient.get('description', 'No original record available')
            neighbors_record = ' '.join(neighbor_records)
            updated_record = (
                f"the information of the similar houses for house {patient_id} are : {neighbors_record}\n"
                f"Only infer the missing values in house {patient_id}'s records: {original_record} "
                # f"Provide only the imputation results using the following succinct output format in JSON without any extra explanations: \"feature name\": \"inferred value\""
            )
            new_round_num = round_num+3
            patient[f'UpdatedRecords_Round_{round_num}'] = updated_record
            
def find_neighbors_and_update_json(data, patients_data, graphs, dataset, data_path, rounds, group_size, num_neighbors):
    all_neighbors = []  # To store neighbors from all rounds
   
    previous_neighbors = {patient: set() for patient in data.index.tolist()} 
    for round_num in range(1, rounds + 1):
        print(f"Starting round {round_num}")
        current_neighbors = {}  
        merged_graphs = merge_and_process_graphs(graphs, group_size=group_size)

        for patient in data.index.tolist():
            aggregated_neighbors = Counter()
            for merged_graph in merged_graphs:
                feature_value_count = len(set(n for n, d in merged_graph.nodes(data=True) if d.get('bipartite') == 1))
                neighbors = top_neighbors_by_weight(merged_graph, patient, steps=2, max_neighbors=100, num_walks=100, feature_value_count=feature_value_count)
                
                for neighbor, weight in neighbors.items():
                    aggregated_neighbors[neighbor] += weight * feature_value_count
            sorted_aggregated_neighbors = aggregated_neighbors.most_common(num_neighbors)  # Get more neighbors to account for filtering
            filtered_neighbors = sorted_aggregated_neighbors[:num_neighbors]
            
            current_neighbors[patient] = {neighbor: weight for neighbor, weight in filtered_neighbors}
            
            previous_neighbors[patient].update(current_neighbors[patient].keys())

        all_neighbors.append(current_neighbors)

        update_patient_records_with_neighbors(patients_data, current_neighbors, round_num)
        json_file_path = os.path.join(data_path, f'{dataset}_neighbors_{round_num}.json')
        save_json_data(patients_data, json_file_path)

    return all_neighbors


def neighbor_search(args):
    graphs = {}
    e = 0.001
    data = pd.read_csv(os.path.join(args.data_path, args.dataset, '.csv'))
    binned_data, bin_edges_dict = bin_continuous_features(data, num_bins=args.num_neighbors)

    graphs = {}
    for feature in binned_data.columns:
        if feature in bin_edges_dict:
            graphs[feature] = create_bipartite_graph_for_binned(binned_data, data, feature, bin_edges_dict[feature])

    original_file_path = os.path.join(args.data_path, f'{args.dataset}_descriptions.json')  # Replace with your actual file path
    patients_data = load_json_data(original_file_path)

    all_neighbors = find_neighbors_and_update_json(data, patients_data, graphs, dataset = args.dataset, data_path=args.data_path, rounds=args.num_round, group_size=args.group_size)

    save_neighbors_to_txt(all_neighbors, file_path=os.path.join(args.data_path, f'{args.dataset}_neighbors.txt'))