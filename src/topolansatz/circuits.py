from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import qiskit_aer
import networkx as nx
from typing import Dict, List, Tuple
import numpy as np
import random

def normalize_subgraph(subgraph: nx.Graph) -> Tuple[nx.Graph, Dict[int, int]]:
    """Create normalized version of subgraph (indices 0...n-1) and mapping.
    
    Args:
        subgraph: Graph with arbitrary qubit indices
        
    Returns:
        Tuple of (normalized graph with sequential indices, mapping from new to original indices)
    """
    # Create mapping of original to sequential indices
    orig_nodes = list(subgraph.nodes())
    index_map = {i: orig_nodes[i] for i in range(len(orig_nodes))}
    
    # Create new graph with normalized indices
    normalized = nx.Graph()
    normalized.add_nodes_from(range(len(orig_nodes)))
    
    # Add edges using normalized indices
    edges = [(orig_nodes.index(u), orig_nodes.index(v)) 
             for u, v in subgraph.edges()]
    normalized.add_edges_from(edges)
    
    return normalized, index_map

class CircuitBuilder:
    def __init__(self, normalized_graph: nx.Graph, circuit_id: str = ""):
        """Initialize with a normalized graph and optional circuit identifier.
        
        Args:
            normalized_graph: Graph with sequential qubit indices
            circuit_id: Identifier to make parameter names unique
        """
        self.graph = normalized_graph
        self.n_qubits = len(normalized_graph.nodes())
        self.circuit_id = circuit_id
        
        # Define available gates
        self.fixed_single_gates = ['h', 't', 's', 'x']
        self.param_single_gates = ['rx', 'ry', 'rz']
        
    def _get_param_name(self, param_idx: int, gate_type: str) -> str:
        """Generate unique parameter name."""
        if self.circuit_id:
            return f"θ_{self.circuit_id}_{gate_type}_{param_idx}"
        return f"θ_{gate_type}_{param_idx}"
        
    def _apply_random_single_gate(self, qc: QuantumCircuit, qubit: int, param_idx: int) -> int:
        """Apply a random single qubit gate - either fixed or parameterized."""
        if random.random() < 0.5:  # 50% chance for parameterized gate
            gate_type = random.choice(self.param_single_gates)
            theta = Parameter(self._get_param_name(param_idx, gate_type))
            if gate_type == 'rx':
                qc.rx(theta, qubit)
            elif gate_type == 'ry':
                qc.ry(theta, qubit)
            else:
                qc.rz(theta, qubit)
            return param_idx + 1
        else:
            gate_type = random.choice(self.fixed_single_gates)
            if gate_type == 'h':
                qc.h(qubit)
            elif gate_type == 't':
                qc.t(qubit)
            elif gate_type == 's':
                qc.s(qubit)
            else:
                qc.x(qubit)
            return param_idx
        
    def _apply_random_two_gate(self, qc: QuantumCircuit, qubit1: int, qubit2: int):
        """Apply a random two qubit gate - either CX or SWAP."""
        if random.random() < 0.7:  # 70% chance for CX
            qc.cx(qubit1, qubit2)
        else:  # 30% chance for SWAP
            qc.swap(qubit1, qubit2)
            
    def create_circuit(self, depth: int = 2) -> QuantumCircuit:
        """Create a circuit with mix of random and parameterized gates."""
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        
        for _ in range(depth):
            # Single qubit gates
            for i in range(self.n_qubits):
                param_idx = self._apply_random_single_gate(qc, i, param_idx)
            
            # Two-qubit gates following topology
            for i, j in self.graph.edges():
                self._apply_random_two_gate(qc, i, j)
        
        return qc

def combine_circuits(subcircuits: List[QuantumCircuit], 
                    index_maps: List[Dict[int, int]], 
                    total_qubits: int) -> QuantumCircuit:
    """Combine multiple subcircuits into larger circuit using original indices.
    
    Args:
        subcircuits: List of quantum circuits to combine
        index_maps: List of mappings from normalized to original indices
        total_qubits: Total number of qubits in combined circuit
        
    Returns:
        Combined quantum circuit
    """
    combined = QuantumCircuit(total_qubits)
    
    # Add each subcircuit
    for qc, index_map in zip(subcircuits, index_maps):
        # Create gate from subcircuit
        gate = qc.to_gate(label=f"SubCircuit-{len(index_map)}q")
        # Get qubit indices for this subcircuit
        qubits = [index_map[i] for i in range(len(index_map))]
        # Add to combined circuit
        combined.append(gate, qubits)
        
    return combined