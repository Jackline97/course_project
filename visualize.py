from pyvis.network import Network
import networkx as nx
from IPython.display import display, IFrame
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize(nodes, edges, filename='graph2.html'):
    # Create a NetworkX graph
    G = nx.DiGraph()

    # Step 1: Extract unique node types
    unique_types = set()
    for node_attr in nodes.values():
        node_type = node_attr.get('type', 'default')
        unique_types.add(node_type)

    # Step 2: Assign colors to node types
    num_types = len(unique_types)
    # Generate a list of colors using a colormap
    cmap = cm.get_cmap('hsv', num_types)
    color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    
    # Create a mapping from node types to colors
    type_color_map = dict(zip(sorted(unique_types), color_list))

    # Step 3: Add nodes with attributes
    for node_name, attr in nodes.items():
        node_type = attr.get('type', 'default')
        color = type_color_map.get(node_type, 'gray')
        # Set 'label' and 'title' for better display in Pyvis
        G.add_node(
            node_name,
            type=node_type,
            label=node_name,
            title=attr.get('content', ''),
            color=color
        )

    # Step 4: Add edges
    for edge_key in edges.keys():
        try:
            source, target = edge_key.split('<|>')
            G.add_edge(source, target)
        except ValueError:
            print(f"Edge key '{edge_key}' is not in the expected 'source<|>target' format.")

    # Step 5: Create a pyvis Network
    net = Network(height='750px', width='100%', notebook=True, directed=True)

    # Manually add nodes and edges to the pyvis Network to include attributes
    for node, data in G.nodes(data=True):
        net.add_node(node, **data)

    for source, target in G.edges():
        net.add_edge(source, target)

    # Step 6: Customize the appearance
    net.repulsion(node_distance=200, central_gravity=0.3)
    net.toggle_physics(True)

    # Step 7: Generate and display the network
    net.show(filename)

    # Display in Jupyter Notebook
    display(IFrame(filename, width='100%', height='750px'))