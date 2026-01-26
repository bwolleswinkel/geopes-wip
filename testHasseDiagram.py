"""Script to test the creation of Hasse diagrams from polytopes"""

import numpy as np
import cdd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


# FROM: Google Gemini | 2026/01/25 [untested/unverified]
def get_hasse_diagram(V):
    """Compute the Hasse diagram for a polytope given by its vertices V.
    Returns:
        G: NetworkX DiGraph
        adj_matrix: numpy array (adjacency matrix, ordered)
        ordered_nodes: list of nodes in order (empty set, vertices, edges, ..., facets, polytope)
    """
    # Compute H-representation (A, b) using cdd
    mat_v = cdd.matrix_from_array(np.column_stack((np.ones(V.shape[0]), V)), rep_type=cdd.RepType.GENERATOR)
    poly = cdd.polyhedron_from_matrix(mat_v)

    # Get Vertex-Facet Incidence
    facet_to_verts = list(cdd.copy_incidence(poly))
    num_vertices = len(V)

    # 3. Generate all faces (subsets of vertices that are faces)
    all_faces = set()
    all_faces.add(frozenset())  # Empty set
    all_faces.add(frozenset(range(num_vertices)))  # Polytope itself
    facet_sets = [frozenset(facet) for facet in facet_to_verts]
    all_faces.update(facet_sets)
    current_level = set(facet_sets)
    while current_level:
        next_level = set()
        for f1 in current_level:
            for f2 in facet_sets:
                inter = f1.intersection(f2)
                if inter and inter not in all_faces:
                    next_level.add(inter)
        all_faces.update(next_level)
        current_level = next_level
    for v in range(num_vertices):
        all_faces.add(frozenset([v]))

    # Helper: geometric dimension
    def face_geom_dim(face, V):
        if len(face) == 0:
            return -1  # empty set
        if len(face) == 1:
            return 0   # vertex
        pts = V[list(face)]
        return np.linalg.matrix_rank(pts - pts[0])

    # Order nodes: empty set, vertices, edges, ..., facets, polytope
    node_geom_dim = {node: face_geom_dim(node, V) for node in all_faces}
    # Group by dimension
    nodes_by_geom_dim = {}
    for node, gdim in node_geom_dim.items():
        nodes_by_geom_dim.setdefault(gdim, []).append(node)
    ordered_nodes = []
    for gdim in sorted(nodes_by_geom_dim):
        if gdim == 0:
            # Vertices: sort by V order
            # Each vertex is a frozenset with one element
            vertex_nodes = [frozenset([i]) for i in range(V.shape[0])]
            ordered_nodes.extend(vertex_nodes)
        else:
            ordered_nodes.extend(sorted(nodes_by_geom_dim[gdim], key=lambda x: tuple(sorted(x))))

    # Build graph
    G = nx.DiGraph()
    for face in ordered_nodes:
        G.add_node(face, size=len(face))
    for i, face_a in enumerate(ordered_nodes):
        for j, face_b in enumerate(ordered_nodes):
            if i == j: continue
            if face_b.issubset(face_a):
                is_direct = True
                for face_c in ordered_nodes:
                    if face_c != face_a and face_c != face_b:
                        if face_b.issubset(face_c) and face_c.issubset(face_a):
                            is_direct = False
                            break
                if is_direct:
                    G.add_edge(face_b, face_a)

    # Build adjacency matrix
    # FIXME: This should probably be a boolean matrix
    adj_matrix = np.zeros((len(ordered_nodes), len(ordered_nodes)), dtype=int)
    node_idx = {node: idx for idx, node in enumerate(ordered_nodes)}
    for u, v in G.edges:
        adj_matrix[node_idx[u], node_idx[v]] = 1

    return G, adj_matrix, ordered_nodes

# --- Example Usage & Plotting ---
# Example: A simple 2D square
# V_sq = np.array([[0,0], [1,0], [1,2], [0,1]])
# Example: A 'house' shape
V_sq = np.array([[0,0], [1,0], [1,2], [0,1], [0.5, 2]])
# Example: the 3D cube
V_sq = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0],
                 [0,0,1], [1,0,1], [1,1,1], [0,1,1]])
# Example: The unnamed 3-polytope from "Basic Properties of Convex Polytopes," Henk Martin
V_sq = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0],
                 [0,0,1], [1,1,1]])



# FROM: GitHub Copilot GPT-4.1 | 2026/01/25
G, adj_matrix, ordered_nodes = get_hasse_diagram(V_sq)


# Print nodes grouped by dimension (by geometric dimension)
print("Hasse diagram nodes by geometric dimension:")
def face_geom_dim(face, V):
    if len(face) == 0:
        return -1  # empty set
    if len(face) == 1:
        return 0   # vertex
    pts = V[list(face)]
    return np.linalg.matrix_rank(pts - pts[0])
node_geom_dim = {node: face_geom_dim(node, V_sq) for node in G.nodes}
nodes_by_geom_dim = {}
for node, gdim in node_geom_dim.items():
    nodes_by_geom_dim.setdefault(gdim, []).append(node)
for gdim in sorted(nodes_by_geom_dim):
    print(f"Dimension {gdim}: {['{' + ','.join(str(v) for v in sorted(face)) + '}' for face in nodes_by_geom_dim[gdim]]}")

# Print edges
print("\nHasse diagram edges (covering relation):")
for u, v in G.edges:
    print(f"{'{' + ','.join(str(x) for x in sorted(u)) + '}'} -> {'{' + ','.join(str(x) for x in sorted(v)) + '}'}")

# Print adjacency matrix with node labels
print("\nAdjacency matrix (rows/cols in order):")
for idx, node in enumerate(ordered_nodes):
    label = "{" + ",".join(str(v) for v in sorted(node)) + "}"
    print(f"Row {idx}: {label}")
print(adj_matrix)

# --- Corrected plotting: group by true geometric dimension ---
def face_geom_dim(face, V):
    if len(face) == 0:
        return -1  # empty set
    if len(face) == 1:
        return 0   # vertex
    pts = V[list(face)]
    return np.linalg.matrix_rank(pts - pts[0])

# Build mapping: node -> geometric dimension
node_geom_dim = {node: face_geom_dim(node, V_sq) for node in G.nodes}
nodes_by_geom_dim = {}
for node, gdim in node_geom_dim.items():
    nodes_by_geom_dim.setdefault(gdim, []).append(node)

max_geom_dim = max(nodes_by_geom_dim.keys())
row_gap = 1.0
col_gap = 1.0
pos = {}
for idx, gdim in enumerate(sorted(nodes_by_geom_dim)):
    if gdim == 0:
        # Vertices: order by V
        nodes = [frozenset([i]) for i in range(V_sq.shape[0])]
    else:
        nodes = sorted(nodes_by_geom_dim[gdim], key=lambda x: tuple(sorted(x)))
    n_nodes = len(nodes)
    x_start = -((n_nodes - 1) * col_gap) / 2 if n_nodes > 1 else 0
    for i, node in enumerate(nodes):
        pos[node] = (x_start + i * col_gap, idx * row_gap)

plt.figure(figsize=(2.5 * max(len(nodes_by_geom_dim.get(d, [])) for d in nodes_by_geom_dim), 2.5 * (max_geom_dim + 2)))

# Draw all nodes as before
nx.draw(G, pos,
    node_color='black', node_size=50,
    arrows=False, with_labels=False,
    )
# Draw only the vertex nodes (white fill, black border, black text, larger size)
vertex_nodes = [frozenset([i]) for i in range(V_sq.shape[0])]
nx.draw_networkx_nodes(G, pos,
    nodelist=vertex_nodes,
    node_color='white',
    node_size=300,
    edgecolors='black',
    linewidths=1
)
for node in vertex_nodes:
    idx = list(node)[0]
    x, y = pos[node]
    plt.text(x, y, rf'${idx}$', fontsize=11, ha='center', va='center', color='black')
# Add only two labels: empty set and polytope, with better placement
for node in G.nodes:
    if len(node) == 0:
        x, y = pos[node]
        plt.text(x, y - 0.05, r'$\varnothing$', fontsize=14, ha='center', va='top')
    elif set(node) == set(range(V_sq.shape[0])):
        x, y = pos[node]
        plt.text(x, y + 0.05, r'$P$', fontsize=14, ha='center', va='bottom')
plt.title("Hasse Diagram of Polytope Face Lattice (by geometric dimension)")
plt.axis('off')
plt.show()

