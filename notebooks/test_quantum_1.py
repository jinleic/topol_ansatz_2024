#!/usr/bin/env python
# coding: utf-8

import sys
from pathlib import Path
from typing import List, Set, Dict
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import time

from topolansatz.topology import TopologyHandler
from topolansatz.circuits import CircuitBuilder
from topolansatz.evaluator import CircuitEvaluator
from topolansatz.stitching import CircuitStitcher
from topolansatz.main import TopolAnsatz

def get_ibm_mumbai_coupling():
    """Returns the coupling map for IBM Mumbai (27 qubit)"""
    return [
        (0,1), (1,2), (2,3), (3,4), # Row 0 
        (0,5), (1,6), (2,7), (3,8), (4,9), # Vertical to Row 1
        (5,6), (6,7), (7,8), (8,9), # Row 1
        (5,10), (6,11), (7,12), (8,13), (9,14), # Vertical to Row 2
        (10,11), (11,12), (12,13), (13,14), # Row 2
        (10,15), (11,16), (12,17), (13,18), (14,19), # Vertical to Row 3
        (15,16), (16,17), (17,18), (18,19), # Row 3
        (15,20), (16,21), (17,22), (18,23), (19,24), # Vertical to Row 4
        (20,21), (21,22), (22,23), (23,24), # Row 4
        (20,25), (21,26), (22,27), (23,28), (24,29) # Vertical to Row 5
    ]

def analyze_subcircuit_generation(topol: TopolAnsatz, n_qubits: int, depth: int):
    """Analyze the subcircuit generation process in detail"""
    print(f"\nAnalyzing {n_qubits}-qubit circuit generation with depth {depth}:")
    
    print("\nStep 1: Circuit Generation")
    print("-" * 40)
    
    start_time = time.time()
    try:
        circuit = topol.generate_ansatz(n_qubits=n_qubits, depth=depth)
        gen_time = time.time() - start_time
        
        print(f"✓ Circuit generated successfully")
        print(f"Generation time: {gen_time:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Circuit generation failed: {str(e)}")
        raise
    
    print("\nStep 2: Circuit Analysis")
    print("-" * 40)
    
    # Basic circuit properties
    print("Basic Properties:")
    print(f"- Number of qubits: {circuit.num_qubits}")
    print(f"- Circuit depth: {circuit.depth()}")
    print(f"- Number of parameters: {len(circuit.parameters)}")
    
    # Gate composition
    gate_counts = {}
    for inst in circuit.data:
        gate_name = inst.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    print("\nGate Composition:")
    for gate, count in gate_counts.items():
        print(f"- {gate}: {count}")
    
    # Parameter analysis
    if circuit.parameters:
        print("\nParameter Analysis:")
        params_by_type = {}
        for param in circuit.parameters:
            param_type = param.name.split('_')[1] if '_' in param.name else 'unknown'
            params_by_type[param_type] = params_by_type.get(param_type, 0) + 1
        
        for param_type, count in params_by_type.items():
            print(f"- {param_type}: {count}")
    
    print("\nStep 3: Quality Metrics")
    print("-" * 40)
    
    start_time = time.time()
    metrics = topol.evaluate_ansatz(circuit)
    eval_time = time.time() - start_time
    
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.4f}")
    
    return circuit, metrics

def verify_topology_constraints(circuit: QuantumCircuit, topology: nx.Graph):
    """Verify that circuit respects topology constraints"""
    print("\nVerifying topology constraints:")
    
    print(f"Circuit width (number of qubits): {circuit.num_qubits}")
    print(f"Total number of gates: {len(circuit.data)}")
    
    violations = 0
    two_qubit_gates = 0
    
    for inst in circuit.data:
        if inst.operation.name in ['cx', 'swap']:
            two_qubit_gates += 1
            # Get the actual qubit indices
            try:
                q1 = inst.qubits[0]._index
                q2 = inst.qubits[1]._index
                
                if not ((q1, q2) in topology.edges() or (q2, q1) in topology.edges()):
                    violations += 1
                    print(f"Topology violation: {inst.operation.name} between qubits {q1}-{q2}")
                else:
                    print(f"Valid {inst.operation.name} gate between qubits {q1}-{q2}")
                    
            except AttributeError as e:
                print(f"Warning: Could not access qubit indices for {inst.operation.name} gate")
                print(f"Qubit information: {inst.qubits}")
                continue
    
    print(f"\nAnalysis complete:")
    print(f"Total two-qubit gates: {two_qubit_gates}")
    if violations == 0:
        print("✓ All two-qubit gates respect topology constraints")
    else:
        print(f"✗ Found {violations} topology violations")
    
    return violations == 0

def compare_partitioning_strategies(topol: TopolAnsatz, n_qubits: int):
    """Compare different circuit partitioning strategies"""
    print("\nComparing partitioning strategies:")
    
    strategies = [
        ([4,4,4], "4-4-4 split"),
        ([6,6], "6-6 split"),
        ([3,3,3,3], "3-3-3-3 split")
    ]
    
    results = {}
    for partition_sizes, name in strategies:
        if sum(partition_sizes) > n_qubits:
            continue
            
        print(f"\nTesting {name}:")
        try:
            circuit = topol.generate_partitioned_ansatz(partition_sizes=partition_sizes)
            metrics = topol.evaluate_ansatz(circuit)
            
            print(f"Circuit depth: {circuit.depth()}")
            print(f"Parameters: {len(circuit.parameters)}")
            print(f"CNOT count: {sum(1 for inst in circuit.data if inst.operation.name == 'cx')}")
            
            for metric, value in metrics.items():
                print(f"- {metric}: {value:.4f}")
                
            results[name] = metrics
            
        except ValueError as e:
            print(f"Strategy failed: {e}")
            
    return results

def main():
    # Use larger coupling map
    coupling_map = get_ibm_mumbai_coupling()
    
    print("Testing TopolAnsatz with IBM Mumbai topology")
    print(f"Coupling map size: {len(coupling_map)} connections")
    
    # Create graph and visualize
    G = nx.Graph(coupling_map)
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.title("IBM Mumbai Qubit Topology")
    plt.show()
    
    # Initialize TopolAnsatz
    topol = TopolAnsatz(coupling_map)
    
    # Test different sizes and depths
    sizes = [8, 12, 16]
    depths = [2, 3, 4]
    
    results = {}
    for n_qubits in sizes:
        for depth in depths:
            print(f"\n{'='*50}")
            print(f"Testing {n_qubits}-qubit circuit with depth {depth}")
            print('='*50)
            
            try:
                circuit, metrics = analyze_subcircuit_generation(topol, n_qubits, depth)
                verify_topology_constraints(circuit, G)
                results[(n_qubits, depth)] = metrics
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    # Compare partitioning strategies
    partition_results = compare_partitioning_strategies(topol, 12)
    
    # Print summary
    print("\n" + "="*50)
    print("Testing Summary:")
    print("="*50)
    
    print("\nCircuit Metrics by Size/Depth:")
    for (size, depth), metrics in results.items():
        print(f"\n{size} qubits, depth {depth}:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
            
    print("\nPartitioning Strategy Results:")
    for strategy, metrics in partition_results.items():
        print(f"\n{strategy}:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
