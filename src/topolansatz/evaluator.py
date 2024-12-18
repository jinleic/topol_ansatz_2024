from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import partial_trace, Statevector
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import rel_entr
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def _compute_fidelity_batch(args: Tuple[QuantumCircuit, int]) -> List[float]:
    """Compute batch of fidelities for parallel processing.
    
    Args:
        args: Tuple of (circuit, batch_size)
    Returns:
        List of fidelity values
    """
    circuit, batch_size = args
    fidelities = []
    for _ in range(batch_size):
        params1 = {p: np.random.uniform(0, 2*np.pi) 
                  for p in circuit.parameters}
        params2 = {p: np.random.uniform(0, 2*np.pi) 
                  for p in circuit.parameters}
        
        state1 = Statevector.from_instruction(circuit.assign_parameters(params1))
        state2 = Statevector.from_instruction(circuit.assign_parameters(params2))
        
        # Use probability distributions for better numerical stability
        dist1 = state1.probabilities()
        dist2 = state2.probabilities()
        fid = np.sum(np.sqrt(dist1) * np.sqrt(dist2))
        fidelities.append(fid)
    return fidelities

class CircuitEvaluator:
    def __init__(self, n_bins: int = 75):
        """Initialize evaluator with quantum simulator.
        
        Args:
            n_bins: Number of bins for histogram (default 75 from reference)
        """
        self.simulator = Aer.get_backend('statevector_simulator')
        
        # Pre-compute bins and Haar distribution
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.haar_hist = self._compute_haar_distribution(n_bins)
        
    def _compute_haar_distribution(self, n_bins: int) -> np.ndarray:
        """Compute theoretical Haar random distribution."""
        def P_haar(l, u, N=2):
            return (1-l)**(N-1) - (1-u)**(N-1)
            
        hist = np.array([P_haar(self.bins[i], self.bins[i+1]) 
                        for i in range(n_bins)])
        return hist / hist.sum()
        
    def evaluate_expressivity(self, 
                            circuit: QuantumCircuit,
                            n_samples: int = 2000,
                            n_processes: Optional[int] = None) -> float:
        """Evaluate circuit expressivity using KL divergence."""
        if len(circuit.parameters) == 0:
            return 0.0  # Fixed circuits have no expressivity
            
        # Configure parallel processing
        if n_processes is None:
            n_processes = min(multiprocessing.cpu_count(), 8)
            
        batch_size = n_samples // n_processes
        
        # Create argument tuples for parallel processing
        args = [(circuit, batch_size) for _ in range(n_processes)]
        
        # Run parallel computation
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(executor.map(_compute_fidelity_batch, args))
            
        # Flatten results
        fidelities = [f for batch in results for f in batch]
            
        # Compute histogram and KL divergence
        hist, _ = np.histogram(fidelities, bins=self.bins, density=True)
        hist = hist / hist.sum()  # Normalize
        
        return np.sum(rel_entr(hist, self.haar_hist))
    
    def _compute_mw_measure(self, state: Statevector, n_qubits: int) -> float:
        """Compute Meyer-Wallach measure for a given state."""
        traces = [partial_trace(state, [j for j in range(n_qubits) if j != i]) 
                 for i in range(n_qubits)]
        
        purity_sum = sum(np.trace(trace.data @ trace.data).real 
                        for trace in traces)
        
        return 2 * (1 - purity_sum / n_qubits)
        
    def evaluate_entanglement(self,
                            circuit: QuantumCircuit,
                            n_samples: int = 1000) -> float:
        """Evaluate entangling capability using Meyer-Wallach measure."""
        if len(circuit.parameters) == 0:
            state = Statevector.from_instruction(circuit)
            return self._compute_mw_measure(state, circuit.num_qubits)
            
        total_mw = 0
        n_qubits = circuit.num_qubits
        
        for _ in range(n_samples):
            params = {p: np.random.uniform(0, 2*np.pi) 
                     for p in circuit.parameters}
            state = Statevector.from_instruction(circuit.assign_parameters(params))
            total_mw += self._compute_mw_measure(state, n_qubits)
            
        return total_mw / n_samples

    def get_circuit_metrics(self, 
                          circuit: QuantumCircuit,
                          n_samples: Optional[int] = None) -> Dict[str, float]:
        """Get complete set of circuit quality metrics."""
        n = n_samples or 1000
        return {
            'expressivity': self.evaluate_expressivity(circuit, n_samples=n),
            'entanglement': self.evaluate_entanglement(circuit, n_samples=n),
            'depth': circuit.depth(),
            'n_parameters': len(circuit.parameters),
            'n_cnot': sum(1 for inst in circuit.data 
                         if inst.operation.name == 'cx'),
        }