import numpy as np
from qiskit import QuantumCircuit
import pytest
from qiskit.circuit import Parameter
from src.topolansatz.core.evaluator import CircuitEvaluator

def create_bell_state() -> QuantumCircuit:
    """Create a Bell state preparation circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def create_fixed_circuit() -> QuantumCircuit:
    """Create a fixed (non-parameterized) circuit."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    return qc

def create_parameterized_circuit(depth: int) -> QuantumCircuit:
    """Create a parameterized circuit with a given depth."""
    qc = QuantumCircuit(2)
    parameters = []

    for d in range(depth):
        theta_0 = Parameter(f'θ_{d}_0')
        theta_1 = Parameter(f'θ_{d}_1')
        parameters.append((theta_0, theta_1))

        qc.rx(theta_0, 0)
        qc.ry(theta_1, 1)
        qc.cx(0, 1)

    return qc

@pytest.fixture
def evaluator():
    return CircuitEvaluator()

def test_bell_state_entanglement(evaluator):
    """Test that Bell state has maximum entanglement."""
    bell = create_bell_state()
    mw_score = evaluator.evaluate_entanglement(bell)
    assert np.isclose(mw_score, 1.0, atol=0.1), f"Bell state MW score = {mw_score}, expected ~1.0"

def test_fixed_circuit_expressivity(evaluator):
    """Test that fixed circuit has zero expressivity."""
    fixed = create_fixed_circuit()
    expr_score = evaluator.evaluate_expressivity(fixed)
    assert np.isclose(expr_score, 0.0, atol=0.1), f"Fixed circuit expressivity = {expr_score}, expected ~0.0"

def test_expressivity_depth_correlation(evaluator):
    """Test that expressivity increases with circuit depth."""
    depths = [1, 3, 5]
    scores = []
    
    for d in depths:
        circuit = create_parameterized_circuit(d)
        score = evaluator.evaluate_expressivity(circuit)
        scores.append(score)
    
    # Check scores are increasing
    assert all(scores[i] < scores[i+1] for i in range(len(scores)-1)), \
        f"Expressivity should increase with depth. Scores: {scores}"

def test_metrics_consistency(evaluator):
    """Test that circuit metrics are consistent."""
    circuit = create_parameterized_circuit(2)
    metrics = evaluator.get_circuit_metrics(circuit)
    
    assert all(k in metrics for k in ['expressivity', 'entanglement', 'depth', 'n_parameters', 'n_cnot']), \
        "Missing metrics in output"
    assert metrics['depth'] == 4  # 2 layers * (2 single + 1 CNOT) gates
    assert metrics['n_parameters'] == 4  # 2 parameters per layer

def test_entanglement_distribution(evaluator):
    """Test distribution of entanglement scores."""
    scores = []
    n_trials = 50
    
    circuit = create_parameterized_circuit(2)
    for _ in range(n_trials):
        score = evaluator.evaluate_entanglement(circuit)
        scores.append(score)
    
    mean = np.mean(scores)
    std = np.std(scores)
    
    # Check reasonable bounds
    assert 0 <= mean <= 1, f"Mean entanglement {mean} outside [0,1]"
    assert std < 0.2, f"Entanglement std {std} too large"

if __name__ == "__main__":
    pytest.main([__file__])