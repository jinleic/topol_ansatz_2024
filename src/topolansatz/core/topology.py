import networkx as nx
from itertools import combinations
from typing import List, Tuple, Dict, Set

class TopologyHandler:
    def __init__(self, coupling_map: List[Tuple[int, int]]):
        """Initialize with a coupling map from Qiskit backend.

        Args:
            coupling_map: List of tuples representing qubit connections
                         e.g. [(0,1), (1,2), (2,3)]
        """
        # Create graph from coupling map
        self.graph = nx.Graph()
        self.graph.add_edges_from(coupling_map)

        # Cache for storing computed subgraphs
        self._subgraph_cache = {}

    def get_connected_subgraphs(self, n_qubits: int) -> List[nx.Graph]:
        """Find all connected subgraphs of size n_qubits.

        Args:
            n_qubits: Number of qubits in subgraph (3-6)

        Returns:
            List of subgraphs with their original qubit indices
        """
        if n_qubits in self._subgraph_cache:
            return self._subgraph_cache[n_qubits]

        if not 3 <= n_qubits <= 6:
            raise ValueError("Subgraph size must be between 3 and 6 qubits")

        subgraphs = []
        # Get all possible combinations of n_qubits vertices
        for nodes in combinations(self.graph.nodes(), n_qubits):
            subgraph = self.graph.subgraph(nodes)
            # Only keep if subgraph is connected
            if nx.is_connected(subgraph):
                subgraphs.append(subgraph)

        self._subgraph_cache[n_qubits] = subgraphs
        return subgraphs

    def normalize_indices(self, subgraph: nx.Graph) -> Tuple[nx.Graph, Dict[int, int]]:
        """Create a normalized version of subgraph with sequential indices 0...n-1.

        Args:
            subgraph: Subgraph with original qubit indices

        Returns:
            Tuple of (normalized subgraph, mapping of new->original indices)
        """
        # Create mapping of original to sequential indices
        nodes = list(subgraph.nodes())
        index_map = {i: nodes[i] for i in range(len(nodes))}

        # Create new graph with normalized indices
        normalized = nx.Graph()
        normalized.add_nodes_from(range(len(nodes)))

        # Add edges using normalized indices
        edges = [(nodes.index(u), nodes.index(v)) for u, v in subgraph.edges()]
        normalized.add_edges_from(edges)

        return normalized, index_map

    def combine_subgraphs(self, subgraphs: List[nx.Graph]) -> nx.Graph:
        """Combine multiple subgraphs into a single larger graph.

        Args:
            subgraphs: List of subgraphs to combine

        Returns:
            Combined graph maintaining original qubit indices
        """
        combined = nx.Graph()
        for subgraph in subgraphs:
            combined.add_edges_from(subgraph.edges())
        return combined

    def verify_placement(self, original_indices: Set[int]) -> bool:
        """Verify that a set of qubit indices forms a valid connected subgraph.

        Args:
            original_indices: Set of qubit indices to verify

        Returns:
            True if indices form a valid connected subgraph
        """
        subgraph = self.graph.subgraph(original_indices)
        return nx.is_connected(subgraph)

# Example usage:
if __name__ == "__main__":
    # Example coupling map from IBM backend
    coupling_map = [(0,1), (1,2), (2,3), (3,4), (4,5), (1,4)]

    handler = TopologyHandler(coupling_map)

    # Get all 4-qubit connected subgraphs
    subgraphs = handler.get_connected_subgraphs(4)

    # Get normalized version of first subgraph
    if subgraphs:
        norm_graph, index_map = handler.normalize_indices(subgraphs[0])
        print(f"Original to normalized mapping: {index_map}")

        # Verify placement is valid
        original_indices = set(index_map.values())
        is_valid = handler.verify_placement(original_indices)
        print(f"Placement is valid: {is_valid}")
