from utils import check_similarity
import numpy as np
from tqdm import tqdm


def graph_deep_merging(nodes, edges, similarity_model, threshold=0.8):
    # Compute pairwise similarities and group similar nodes
    node_names = list(nodes.keys())
    num_nodes = len(node_names)
    similarity_matrix = np.zeros((num_nodes, num_nodes))

    for i in tqdm(range(num_nodes)):
        for j in range(i+1, num_nodes):
            node_i = [node_names[i], nodes[node_names[i]]['type'], nodes[node_names[i]]['content']]
            node_j = [node_names[j], nodes[node_names[j]]['type'], nodes[node_names[j]]['content']]
            sim_score = similarity_model.advanced_similarity(node_i, node_j, threshold=threshold)
            similarity_matrix[i, j] = sim_score
            similarity_matrix[j, i] = sim_score

    # Union-Find data structure for grouping
    parent = {node_name: node_name for node_name in node_names}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]  # Path compression
            node = parent[node]
        return node

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            parent[root2] = root1  # Merge the sets

    # Perform union operations based on similarity
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if similarity_matrix[i, j] >= threshold:
                node_i = node_names[i]
                node_j = node_names[j]
                union(node_i, node_j)

    # Group nodes by their root parent
    groups = {}
    for node_name in node_names:
        root = find(node_name)
        if root not in groups:
            groups[root] = []
        groups[root].append(node_name)

    # Merge nodes within each group
    node_mapping = {}
    for group_nodes in groups.values():
        representative_name = group_nodes[0]  # Choose the first node as representative
        for node_name in group_nodes:
            node_mapping[node_name] = representative_name

    # Merge nodes
    merged_nodes = {}
    for original_name, node_data in nodes.items():
        representative_name = node_mapping[original_name]
        if representative_name not in merged_nodes:
            merged_nodes[representative_name] = {
                'type': node_data['type'],
                'content': node_data['content']
            }
        else:
            # Merge content
            existing_content = merged_nodes[representative_name]['content']
            new_content = node_data['content']
            if new_content not in existing_content:
                merged_nodes[representative_name]['content'] += '\n' + new_content
            # Handle type conflicts if necessary

    merged_edges = {}
    for edge_key, edge_label in edges.items():
        source, target = edge_key.split('<|>')
        new_source = node_mapping.get(source, source)
        new_target = node_mapping.get(target, target)
        if new_source != new_target:
            new_edge_key = f'{new_source}<|>{new_target}'
            # Merge edges if necessary
            if new_edge_key in merged_edges:
                existing_label = merged_edges[new_edge_key]
                if edge_label not in existing_label:
                    merged_edges[new_edge_key] += '\n' + edge_label
            else:
                merged_edges[new_edge_key] = edge_label
    
    return merged_edges, merged_nodes


def graph_init_merging(groups, similarity_model, threshold=0.8):
    """
    Merges node and edge information from multiple groups into final_node and final_edge dictionaries.
    """
    final_node = {}
    final_edge = {}

    for group_id, cur_group in groups.items():
        node_dict = cur_group['node']
        edge_dict = cur_group['edge']

        # Process nodes
        for node_id, node_info in node_dict.items():
            node_type = node_info['type']
            node_content = node_info['node_content']
            
            if node_id not in final_node:
                # Add new node with its content
                final_node[node_id] = {'type': node_type, 'content': [node_content]}
            else:
                # Check if the new content is dissimilar to existing contents
                if check_similarity(final_node[node_id]['content'], node_content, similarity_model, threshold=threshold):
                    final_node[node_id]['content'].append(node_content)

        # Process edges
        for key, edge_info in edge_dict.items():
            source_node = edge_info['source']
            target_node = edge_info['target']
            edge_description = edge_info['relationship']
            edge_key = f"{source_node}<|>{target_node}"
            reverse_edge_key = f"{target_node}<|>{source_node}"

            if edge_key not in final_edge and reverse_edge_key not in final_edge:
                # Add new edge with its description
                final_edge[edge_key] = [edge_description]
            elif edge_key in final_edge:
                # Check if the new description is dissimilar to existing descriptions
                if check_similarity(final_edge[edge_key], edge_description, similarity_model, threshold=threshold):
                    final_edge[edge_key].append(edge_description)
            elif reverse_edge_key in final_edge:
                # Check if the new description is dissimilar to existing descriptions in reverse edge
                if check_similarity(final_edge[reverse_edge_key], edge_description, similarity_model, threshold=threshold):
                    final_edge[reverse_edge_key].append(edge_description)

    for node in final_node:
        final_node[node]['content'] = '\n'.join(final_node[node]['content'])
    
    for edge in final_edge:
        final_edge[edge] = '\n'.join(final_edge[edge])
        
    return final_node, final_edge