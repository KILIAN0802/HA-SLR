import numpy as np


# =========================================================
# Helper functions (spatial strategy)
# =========================================================
def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    neighbor_link = [(i, j) for (i, j) in inward + outward]

    # Build undirected adjacency for hop distance
    A_tmp = np.zeros((num_node, num_node))
    for i, j in neighbor_link:
        A_tmp[j, i] = 1
        A_tmp[i, j] = 1

    # Choose center node (Hip Center) as index 45 (0-based) for this layout
    center = 45

    hop_dis = get_hop_distance(num_node, neighbor_link, max_hop=num_node)
    dist_center = hop_dis[center, :]

    # A0: self connections
    A0 = np.zeros((num_node, num_node))
    for i, j in self_link:
        A0[i, j] = 1
    A0 = normalize_digraph(A0)

    # A1: centripetal (j closer to center than i)
    A1 = np.zeros((num_node, num_node))
    for i, j in neighbor_link:
        if dist_center[i] < dist_center[j]:
            A1[i, j] = 1
    A1 = normalize_digraph(A1)

    # A2: centrifugal (j farther from center than i)
    A2 = np.zeros((num_node, num_node))
    for i, j in neighbor_link:
        if dist_center[i] > dist_center[j]:
            A2[i, j] = 1
    A2 = normalize_digraph(A2)

    A = np.stack((A0, A1, A2))
    return A



class Graph:
    """Graph layout for 46 Mediapipe keypoints"""

    def __init__(self, layout='vsl_layout', strategy='spatial', max_hop=1, dilation=1, labeling_mode=None):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = 46

        self.layout = layout
        self.strategy = strategy
        self.labeling_mode = labeling_mode

        # Define inward edges
        self.inward = self._get_inward_edges()
        self.outward = [(j, i) for (i, j) in self.inward]
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor = self.inward + self.outward

        # adjacency
        self.A = self.get_adjacency_matrix()

    def _get_inward_edges(self):
        inward = []
        # ---- LEFT HAND (0-20) ----
        left_offset = 0
        left_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        inward += [(i + left_offset, j + left_offset) for (i, j) in left_edges]

        # ---- RIGHT HAND (21-41) ----
        right_offset = 21
        inward += [(i + right_offset, j + right_offset) for (i, j) in left_edges]

        # ---- BODY (42-45) ----
        # 42:Nose, 43:L-Sho, 44:R-Sho, 45:Hip-Cen
        body_edges = [
            (42, 43), (42, 44),
            (43, 45), (44, 45),
        ]
        inward += body_edges

        # ---- CONNECTIONS (Body -> Hands) ----
        inward += [(43, 0), (44, 21)]

        return inward

    def get_adjacency_matrix(self):
        if self.strategy == 'spatial':
            return get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError("Only 'spatial' strategy is supported")

