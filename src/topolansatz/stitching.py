from qiskit import QuantumCircuit
import networkx as nx
from typing import List, Dict, Tuple, Optional
from .evaluator import CircuitEvaluator  # Change to relative import

class CircuitStitcher:
    def __init__(self, evaluator: CircuitEvaluator, topology: nx.Graph):
        """Initialize circuit stitcher with evaluator and topology info."""
        self.evaluator = evaluator
        self.topology = topology

    def find_connection_points(self, 
                             subgraph_mappings: List[Dict[int, int]]) -> List[Tuple[int, int]]:
        """Find valid connection points between subcircuits based on topology.
        
        Args:
            subgraph_mappings: List of mappings from local to global qubit indices
            
        Returns:
            List of valid qubit pairs that can be connected based on topology
        """
        connection_points = []
        
        # Get all possible pairs between subcircuits
        for i, map1 in enumerate(subgraph_mappings[:-1]):
            for map2 in subgraph_mappings[i+1:]:
                # Check each qubit pair
                for local1, global1 in map1.items():
                    for local2, global2 in map2.items():
                        # Verify connection exists in topology
                        if (global1, global2) in self.topology.edges() or \
                           (global2, global1) in self.topology.edges():
                            connection_points.append((global1, global2))
                            
        return connection_points

    def apply_stitching_gate(self,
                          circuit: QuantumCircuit,
                          qubits: Tuple[int, int],
                          stitch_type: str = 'cnot') -> None:
        """Apply stitching gate between qubits."""
        q1, q2 = qubits
        
        if stitch_type == 'cnot':
            circuit.cx(q1, q2)
        elif stitch_type == 'swap':
            circuit.swap(q1, q2)
        elif stitch_type == 'cphase':
            circuit.cz(q1, q2)
        else:
            raise ValueError(f"Unknown stitch type: {stitch_type}")

    def stitch_circuits(self,
                       subcircuits: List[QuantumCircuit],
                       subgraph_mappings: List[Dict[int, int]],
                       connections: List[Tuple[int, int]], 
                       stitch_type: str = 'cnot') -> QuantumCircuit:
        """Concatenate subcircuits and add stitching connections.
        
        Args:
            subcircuits: List of subcircuits to combine
            subgraph_mappings: Mappings from local to global qubit indices
            connections: List of qubit pairs to connect
            stitch_type: Type of stitching gate to use
            
        Returns:
            Combined quantum circuit
        """
        # Get total number of qubits needed
        all_qubits = set()
        for mapping in subgraph_mappings:
            all_qubits.update(mapping.values())
        n_qubits = max(all_qubits) + 1
        
        # Create combined circuit
        combined = QuantumCircuit(n_qubits)
        
        # Add each subcircuit using correct qubit mapping
        for qc, mapping in zip(subcircuits, subgraph_mappings):
            # Create instruction from subcircuit
            inst = qc.to_instruction()
            # Map local qubit indices to global indices
            global_qubits = [mapping[i] for i in range(qc.num_qubits)]
            # Apply to mapped qubits
            combined.append(inst, global_qubits)
            
        # Add stitching connections that respect topology
        for q1, q2 in connections:
            if (q1, q2) in self.topology.edges() or \
               (q2, q1) in self.topology.edges():
                self.apply_stitching_gate(combined, (q1, q2), stitch_type)
            
        return combined

    def optimize_stitching(self,
                        subcircuits: List[QuantumCircuit],
                        subgraph_mappings: List[Dict[int, int]],
                        n_connections: int = 2) -> QuantumCircuit:
        """Find optimal stitching pattern for subcircuits."""
        best_circuit = None
        best_score = float('inf')
        
        # Get valid connection points based on topology
        possible_connections = self.find_connection_points(subgraph_mappings)
        
        # Try different combinations of connections
        for stitch_type in ['cnot', 'swap']:
            from itertools import combinations
            for conn_subset in combinations(possible_connections, n_connections):
                circuit = self.stitch_circuits(
                    subcircuits=subcircuits,
                    subgraph_mappings=subgraph_mappings,
                    connections=conn_subset,
                    stitch_type=stitch_type
                )
                
                metrics = self.evaluator.get_circuit_metrics(circuit)
                
                # Score based on balance of expressivity and entanglement
                score = metrics['expressivity'] / (metrics['entanglement'] + 1e-10)
                
                if score < best_score:
                    best_score = score
                    best_circuit = circuit
                        
        return best_circuit if best_circuit is not None else subcircuits[0]  # Fallback to first subcircuit if no stitching possible