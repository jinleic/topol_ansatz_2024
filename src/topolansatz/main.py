# src/topolansatz/main.py
from typing import List, Dict, Optional
import networkx as nx
from qiskit import QuantumCircuit
from .topology import TopologyHandler
from .circuits import CircuitBuilder
from .evaluator import CircuitEvaluator
from .stitching import CircuitStitcher

class TopolAnsatz:
    def __init__(self, coupling_map: List[tuple]):
        """Initialize TopolAnsatz with quantum hardware topology.
        
        Args:
            coupling_map: List of tuples representing qubit connections
        """
        # Create graph from coupling map
        self.graph = nx.Graph()
        self.graph.add_edges_from(coupling_map)
        
        self.topology = TopologyHandler(coupling_map)
        self.evaluator = CircuitEvaluator()
        # Pass both evaluator and topology graph to stitcher
        self.stitcher = CircuitStitcher(self.evaluator, self.graph)


    def generate_ansatz(self,
                    n_qubits: int,
                    depth: int = 2,
                    n_subcircuits: Optional[int] = None) -> QuantumCircuit:
        """Generate topology-aware ansatz circuit."""
        # 1. Find subgraphs that fit topology
        subgraphs = []
        subgraph_mappings = []  # Keep track of mappings
        remaining_qubits = n_qubits
        
        while remaining_qubits > 0:
            size = min(remaining_qubits, 6)  # Max 6 qubits per subcircuit
            new_subgraphs = self.topology.get_connected_subgraphs(size)
            if not new_subgraphs:
                size -= 1
                continue
            subgraph = new_subgraphs[0]  # Take first valid subgraph
            # Get normalized graph and mapping
            norm_graph, mapping = self.topology.normalize_indices(subgraph)
            subgraphs.append(norm_graph)
            subgraph_mappings.append(mapping)  # Store mapping
            remaining_qubits -= size
                
        # 2. Generate subcircuits for each subgraph
        subcircuits = []
        for i, graph in enumerate(subgraphs):
            builder = CircuitBuilder(graph, f"sub_{i}")
            subcircuits.append(builder.create_circuit(depth))
                
        # 3. Find possible connection points
        connections = self.stitcher.find_connection_points(subgraph_mappings)
        
        # 4. Optimize stitching between subcircuits
        n_connections = min(len(connections), n_qubits // 2)
        ansatz = self.stitcher.optimize_stitching(
            subcircuits=subcircuits,
            subgraph_mappings=subgraph_mappings,  # Pass mappings, not connections
            n_connections=n_connections
        )
            
        return ansatz



    def evaluate_ansatz(self, ansatz: QuantumCircuit) -> Dict[str, float]:
        """Evaluate quality metrics for ansatz."""
        return self.evaluator.get_circuit_metrics(ansatz)



    def generate_partitioned_ansatz(self,
                                total_qubits: int = 10,
                                partition_sizes: List[int] = None,
                                depth: int = 2) -> QuantumCircuit:
        """Generate ansatz with specified partition sizes.
        
        Args:
            total_qubits: Total number of qubits in the system
            partition_sizes: List of integers specifying size of each partition
            depth: Depth of each subcircuit
        
        Returns:
            Combined quantum circuit
        """
        if partition_sizes is None:
            # Default partitioning for 10 qubits: 4+4+2
            partition_sizes = [4, 4, 2]
        
        if sum(partition_sizes) != total_qubits:
            raise ValueError(f"Partition sizes {partition_sizes} must sum to total_qubits {total_qubits}")
        
        # Store subcircuits and their mappings
        subcircuits = []
        subgraph_mappings = []
        used_qubits = set()
        
        # Generate subcircuits for each partition
        for i, size in enumerate(partition_sizes):
            # Find valid subgraph of required size that doesn't overlap
            valid_subgraph = None
            for subgraph in self.topology.get_connected_subgraphs(size):
                subgraph_qubits = set(subgraph.nodes())
                if not subgraph_qubits.intersection(used_qubits):
                    valid_subgraph = subgraph
                    used_qubits.update(subgraph_qubits)
                    break
            
            if valid_subgraph is None:
                raise ValueError(f"Could not find non-overlapping subgraph of size {size}")
            
            # Normalize and build circuit
            norm_graph, mapping = self.topology.normalize_indices(valid_subgraph)
            builder = CircuitBuilder(norm_graph, f"sub_{i}")
            subcircuit = builder.create_circuit(depth)
            
            subcircuits.append(subcircuit)
            subgraph_mappings.append(mapping)
        
        # Find stitching points between adjacent subcircuits
        stitching_points = []
        for i in range(len(subcircuits)-1):
            map1 = subgraph_mappings[i]
            map2 = subgraph_mappings[i+1]
            connections = self.stitcher.find_connection_points([map1, map2])
            if connections:
                stitching_points.extend(connections[:2])  # Take up to 2 connections between each pair
        
        # Generate final stitched circuit
        return self.stitcher.optimize_stitching(
            subcircuits=subcircuits,
            subgraph_mappings=subgraph_mappings,
            n_connections=len(stitching_points)
        )

def main():
    # Example usage
    # IBM Quantum Falcon Processor coupling map
    coupling_map = [
        (0,1), (1,2), (2,3), (3,4),
        (0,5), (1,6), (2,7), (3,8), (4,9),
        (5,6), (6,7), (7,8), (8,9)
    ]
    
    # Create TopolAnsatz instance
    topol = TopolAnsatz(coupling_map)
    
    # Generate 8-qubit ansatz
    ansatz = topol.generate_ansatz(n_qubits=8, depth=3)
    
    # Evaluate ansatz quality
    metrics = topol.evaluate_ansatz(ansatz)
    print("Ansatz metrics:", metrics)
    
if __name__ == "__main__":
    main()