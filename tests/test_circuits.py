import sys
import os
import pytest
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
import qiskit

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.topolansatz.core.circuits import CircuitBuilder, normalize_subgraph, combine_circuits

@pytest.fixture
def simple_graph():
    """Create a simple 3-qubit graph with line topology."""
    g = nx.Graph()
    g.add_edges_from([(0,1), (1,2)])
    return g

@pytest.fixture
def two_subgraphs():
    """Create two 3-qubit subgraphs."""
    g1 = nx.Graph()
    g1.add_edges_from([(0,1), (1,2)])
    
    g2 = nx.Graph()
    g2.add_edges_from([(0,1), (1,2)])
    
    return g1, g2

@pytest.fixture
def builder(simple_graph):
    """Create a CircuitBuilder instance."""
    return CircuitBuilder(simple_graph, circuit_id="test")

def test_circuit_builder_initialization(builder):
    assert builder.n_qubits == 3
    assert len(builder.graph.edges()) == 2
    assert builder.circuit_id == "test"

def test_create_circuit_basic(builder):
    """Test basic circuit creation."""
    qc = builder.create_circuit(depth=1)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 3
    
def test_parameter_uniqueness(two_subgraphs):
    """Test that parameters are unique across multiple circuits."""
    builder1 = CircuitBuilder(two_subgraphs[0], circuit_id="A")
    builder2 = CircuitBuilder(two_subgraphs[1], circuit_id="B")
    
    qc1 = builder1.create_circuit(depth=2)
    qc2 = builder2.create_circuit(depth=2)
    
    # Get all parameter names
    params1 = {str(param) for param in qc1.parameters}
    params2 = {str(param) for param in qc2.parameters}
    
    # Check for overlap
    assert not params1.intersection(params2), "Parameters should be unique between circuits"

def test_circuit_topology_constraints(builder):
    """Test that created circuits respect topology constraints."""
    qc = builder.create_circuit(depth=2)
    
    # Get all qubit pairs used in two-qubit gates
    two_qubit_pairs = set()
    for inst, qargs, _ in qc.data:
        if len(qargs) == 2:  # Two-qubit gate
            # pair = tuple(sorted(gate.index for gate in qargs))
            pair = tuple([qc.find_bit(qarg)[0] for qarg in qargs])
            two_qubit_pairs.add(pair)
    
    # Check all pairs are in the graph edges
    graph_edges = {tuple(sorted(edge)) for edge in builder.graph.edges()}
    assert two_qubit_pairs.issubset(graph_edges), "Circuit contains gates between unconnected qubits"

def test_normalize_subgraph():
    """Test subgraph normalization."""
    # Create a subgraph with non-sequential indices
    g = nx.Graph()
    g.add_edges_from([(3,4), (4,5)])
    
    normalized, index_map = normalize_subgraph(g)
    
    # Check normalized graph has sequential indices
    assert set(normalized.nodes()) == {0, 1, 2}
    assert len(normalized.edges()) == len(g.edges())
    
    # Check mapping
    assert len(index_map) == 3
    assert all(0 <= i <= 2 for i in index_map.keys())
    assert all(3 <= i <= 5 for i in index_map.values())

def test_combine_circuits(two_subgraphs):
    """Test circuit combination."""
    builder1 = CircuitBuilder(two_subgraphs[0], circuit_id="A")
    builder2 = CircuitBuilder(two_subgraphs[1], circuit_id="B")
    
    qc1 = builder1.create_circuit(depth=1)
    qc2 = builder2.create_circuit(depth=1)
    
    # Create mappings
    map1 = {0: 0, 1: 1, 2: 2}  # First subgraph uses qubits 0,1,2
    map2 = {0: 3, 1: 4, 2: 5}  # Second subgraph uses qubits 3,4,5
    
    # Combine circuits
    combined = combine_circuits(
        subcircuits=[qc1, qc2],
        index_maps=[map1, map2],
        total_qubits=6
    )
    
    assert combined.num_qubits == 6
    # Each subcircuit becomes a gate instruction
    instructions = combined.data
    custom_gates = [inst for inst in instructions 
                   if isinstance(inst.operation, qiskit.circuit.gate.Gate) and 
                   not inst.operation.name in ['cx', 'h', 't', 's', 'x', 'rx', 'ry', 'rz', 'swap']]
    assert len(custom_gates) == 2

def test_random_gate_distribution(builder):
    """Test the distribution of random gates."""
    # Create multiple circuits and check gate distribution
    n_samples = 100
    param_gate_count = 0
    fixed_gate_count = 0
    
    for _ in range(n_samples):
        qc = builder.create_circuit(depth=1)
        for inst, _, _ in qc.data:
            if inst.params:  # Parameterized gate
                param_gate_count += 1
            else:  # Fixed gate
                if inst.num_qubits == 1:
                    fixed_gate_count += 1
    
    # Check rough distribution (allowing for randomness)
    total_gates = param_gate_count + fixed_gate_count
    param_ratio = param_gate_count / total_gates
    assert 0.4 <= param_ratio <= 0.6, "Gate distribution significantly deviates from expected 50/50 split"

def test_circuit_depth(builder):
    """Test that circuit depth parameter works correctly."""
    depths = [1, 2, 3]
    for d in depths:
        qc = builder.create_circuit(depth=d)
        # Circuit depth might be larger than requested due to two-qubit gates
        assert qc.depth() >= d, f"Circuit depth less than requested depth {d}"

if __name__ == "__main__":
    pytest.main([__file__])