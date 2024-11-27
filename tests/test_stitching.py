import pytest
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from src.topolansatz.core.evaluator import CircuitEvaluator
from src.topolansatz.core.circuits import CircuitBuilder
from src.topolansatz.optimization.stitching import CircuitStitcher

@pytest.fixture
def evaluator():
    return CircuitEvaluator()

@pytest.fixture
def topology():
    """Create sample topology for testing."""
    graph = nx.Graph()
    # Create a 2x3 grid topology
    graph.add_edges_from([
        (0,1), (1,2),
        (3,4), (4,5),
        (0,3), (1,4), (2,5)
    ])
    return graph

@pytest.fixture
def stitcher(evaluator, topology):
    return CircuitStitcher(evaluator, topology)

@pytest.fixture
def sample_subgraphs():
    """Create two simple subgraphs with mappings for testing."""
    g1 = nx.Graph()
    g1.add_edges_from([(0,1), (1,2)])
    map1 = {0: 0, 1: 1, 2: 2}  # Local to global mapping
    
    g2 = nx.Graph()
    g2.add_edges_from([(0,1), (1,2)])
    map2 = {0: 3, 1: 4, 2: 5}  # Local to global mapping
    
    return [g1, g2], [map1, map2]

@pytest.fixture
def sample_subcircuits():
    """Create simple test circuits with parameters."""
    # Define parameters
    θ1 = Parameter('θ1')
    θ2 = Parameter('θ2')
    
    # First test circuit with a Hadamard, CNOT, and RX gate
    qc1 = QuantumCircuit(3)
    qc1.h(0)          
    qc1.cx(0, 1)      
    qc1.rx(θ1, 2)     
    
    # Second test circuit with a Hadamard, CNOT, and RY gate
    qc2 = QuantumCircuit(3)
    qc2.h(0)          
    qc2.cx(1, 2)      
    qc2.ry(θ2, 0)     
    
    return [qc1, qc2]

def test_find_connection_points(stitcher, sample_subgraphs):
    """Test finding valid connection points between subgraphs."""
    subgraphs, mappings = sample_subgraphs
    connections = stitcher.find_connection_points(mappings)
    
    # Should only find connections that exist in topology
    assert len(connections) > 0
    
    # Check valid connection pairs against topology
    for q1, q2 in connections:
        assert (q1, q2) in stitcher.topology.edges() or \
               (q2, q1) in stitcher.topology.edges()

def test_apply_stitching_gate(stitcher):
    """Test application of different types of stitching gates."""
    qc = QuantumCircuit(2)
    
    # Test CNOT stitching
    stitcher.apply_stitching_gate(qc, (0,1), 'cnot')
    assert qc.count_ops()['cx'] == 1
    
    # Test SWAP stitching
    qc_swap = QuantumCircuit(2)
    stitcher.apply_stitching_gate(qc_swap, (0,1), 'swap')
    assert qc_swap.count_ops()['swap'] == 1
    
    # Test invalid type
    qc_invalid = QuantumCircuit(2)
    with pytest.raises(ValueError):
        stitcher.apply_stitching_gate(qc_invalid, (0,1), 'invalid_type')
        
    # Test cphase gate
    qc_cphase = QuantumCircuit(2)
    stitcher.apply_stitching_gate(qc_cphase, (0,1), 'cphase')
    assert qc_cphase.count_ops()['cz'] == 1

def test_stitch_circuits(stitcher, sample_subcircuits, sample_subgraphs):
    """Test combining circuits with stitching."""
    _, mappings = sample_subgraphs
    connections = [(0,3), (2,5)]  # Valid topology connections
    
    stitched = stitcher.stitch_circuits(
        sample_subcircuits,
        mappings,
        connections,
        'cnot'
    )
    
    # Check circuit properties
    assert stitched.num_qubits == 6  # Total qubits from mappings
    # assert len(stitched.data) > sum(len(qc.data) for qc in sample_subcircuits)
    assert 'cx' in stitched.count_ops()
    
    # Verify connections respect topology
    for op in stitched.data:
        if op.operation.name == 'cx':
            q1, q2 = [stitched.find_bit(qb).index for qb in op.qubits]
            assert (q1, q2) in stitcher.topology.edges() or \
                   (q2, q1) in stitcher.topology.edges()

def test_optimize_stitching(stitcher, sample_subcircuits, sample_subgraphs):
    """Test stitching optimization with topology constraints."""
    _, mappings = sample_subgraphs
    optimized = stitcher.optimize_stitching(
        sample_subcircuits,
        mappings,
        n_connections=2
    )
    
    # Check optimization results
    metrics = stitcher.evaluator.get_circuit_metrics(optimized)
    assert metrics['entanglement'] > 0
    assert metrics['expressivity'] >= 0
    
    # Verify topology constraints
    for op in optimized.data:
        if op.operation.name in ['cx', 'swap']:
            q1, q2 = [optimized.find_bit(qb).index for qb in op.qubits]
            assert (q1, q2) in stitcher.topology.edges() or \
                   (q2, q1) in stitcher.topology.edges()

def test_stitching_improvement(stitcher, sample_subcircuits, sample_subgraphs):
    """Test that topology-aware stitching improves circuit properties."""
    _, mappings = sample_subgraphs
    
    # Get metrics before stitching
    combined = QuantumCircuit(6)
    for qc, mapping in zip(sample_subcircuits, mappings):
        global_qubits = [mapping[i] for i in range(qc.num_qubits)]
        combined.append(qc.to_instruction(), global_qubits)
    before_metrics = stitcher.evaluator.get_circuit_metrics(combined)
    
    # Get metrics after stitching
    connections = [(0,3), (2,5)]  # Valid topology connections
    stitched = stitcher.stitch_circuits(sample_subcircuits, mappings, connections)
    after_metrics = stitcher.evaluator.get_circuit_metrics(stitched)
    
    # Entanglement should improve with stitching
    assert after_metrics['entanglement'] >= before_metrics['entanglement']