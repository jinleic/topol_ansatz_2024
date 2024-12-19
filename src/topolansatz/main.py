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
                                  partition_sizes: List[int],
                                  depth: int = 2,
                                  specific_subgraphs: List[set] = None) -> QuantumCircuit:
        """Generate partitioned ansatz with stitching between partitions.
        
        Args:
            partition_sizes: List of partition sizes (must sum to 10)
            depth: Depth of each subcircuit
            specific_subgraphs: List of sets, each containing qubit indices for a connected subgraph
            
        Returns:
            Combined quantum circuit with stitching
        """
        if specific_subgraphs:
            # Validate provided subgraphs
            for subgraph in specific_subgraphs:
                if not nx.is_connected(self.topology.graph.subgraph(subgraph)):
                    raise ValueError(f"Subgraph {subgraph} is not connected")
            
            used_qubits = set().union(*specific_subgraphs)
            if len(used_qubits) != 10:
                raise ValueError(f"Subgraphs must use all 10 qubits, got {len(used_qubits)}")
            
            # Use the provided subgraphs
            subgraphs = [self.topology.graph.subgraph(sg) for sg in specific_subgraphs]
            
        else:
            # Original logic for finding subgraphs
            subgraphs = []
            used_qubits = set()
            
            for size in partition_sizes:
                valid_subgraph = None
                for sg in self.topology.get_connected_subgraphs(size):
                    sg_qubits = set(sg.nodes())
                    if not sg_qubits.intersection(used_qubits):
                        valid_subgraph = sg
                        used_qubits.update(sg_qubits)
                        break
                
                if valid_subgraph is None:
                    raise ValueError(f"Could not find valid {size}-qubit subgraph")
                subgraphs.append(valid_subgraph)
        
        # Generate subcircuits
        subcircuits = []
        subgraph_mappings = []
        
        for i, subgraph in enumerate(subgraphs):
            norm_graph, mapping = self.topology.normalize_indices(subgraph)
            builder = CircuitBuilder(norm_graph, f"sub_{i}")
            subcircuit = builder.create_circuit(depth)
            subcircuits.append(subcircuit)
            subgraph_mappings.append(mapping)
        
        # Find stitching points
        connections = []
        for i in range(len(subgraphs) - 1):
            map1 = subgraph_mappings[i]
            map2 = subgraph_mappings[i+1]
            conn_points = self.stitcher.find_connection_points([map1, map2])
            if conn_points:
                connections.extend(conn_points[:2])  # Take up to 2 connections between partitions
        
        # Generate final stitched circuit
        final_circuit = self.stitcher.stitch_circuits(
            subcircuits=subcircuits,
            subgraph_mappings=subgraph_mappings,
            connections=connections
        )
        
        return final_circuit


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