#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Standard library imports
import sys
from pathlib import Path
from typing import List, Set, Dict

# Scientific computing and visualization
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

# Import our package components
from topolansatz.topology import TopologyHandler
from topolansatz.circuits import CircuitBuilder
from topolansatz.evaluator import CircuitEvaluator
from topolansatz.stitching import CircuitStitcher
from topolansatz.main import TopolAnsatz

def visualize_connected_subgraphs(graph: nx.Graph, subgraphs: List[set]):
    """Visualize connected subgraphs in IBM Quantum hardware topology.
    
    Args:
        graph: NetworkX graph representing hardware topology
        subgraphs: List of sets containing qubit indices for each partition
    """
    plt.figure(figsize=(12, 8))
    
    # Define fixed positions for IBM Quantum layout
    pos = {
        0: (0, 1), 1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (4, 1),  # Top row
        5: (0, 0), 6: (1, 0), 7: (2, 0), 8: (3, 0), 9: (4, 0)   # Bottom row
    }
    
    # Colors for different subgraphs
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    # Draw full topology graph (faded)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    
    # Draw each subgraph with different color
    for i, subgraph_nodes in enumerate(subgraphs):
        subgraph = graph.subgraph(subgraph_nodes)
        nx.draw_networkx_nodes(graph, pos, nodelist=subgraph_nodes, 
                             node_color=colors[i], node_size=500)
        nx.draw_networkx_edges(subgraph, pos, edge_color=colors[i])
        nx.draw_networkx_labels(graph, pos, {n: n for n in subgraph_nodes})
    
    plt.title("Connected Subgraphs in IBM Quantum Hardware Topology")
    plt.axis('equal')
    plt.show()

def verify_subgraphs(G: nx.Graph, subgraphs: List[set]):
    """Verify that subgraphs are valid.
    
    Args:
        G: Full topology graph
        subgraphs: List of sets of qubit indices
    """
    # Check connectivity
    for i, sg in enumerate(subgraphs):
        if not nx.is_connected(G.subgraph(sg)):
            raise ValueError(f"Subgraph {i+1} {sg} is not connected")
    
    # Check overlap
    all_qubits = set()
    for sg in subgraphs:
        if sg.intersection(all_qubits):
            raise ValueError(f"Subgraph {sg} overlaps with others")
        all_qubits.update(sg)
    
    # Check total qubits
    if len(all_qubits) != 10:
        raise ValueError(f"Using {len(all_qubits)} qubits, need 10")

def main():
    # Define IBM Quantum Falcon topology
    coupling_map = [
        (0,1), (1,2), (2,3), (3,4),  # Top row
        (0,5), (1,6), (2,7), (3,8), (4,9),  # Vertical connections
        (5,6), (6,7), (7,8), (8,9)   # Bottom row
    ]
    
    # Create graph
    G = nx.Graph(coupling_map)
    
    # Define connected subgraphs (4+3+3 partitioning)
    subgraphs = [
        set([0, 1, 5, 6]),  # 4-qubit connected square
        set([2, 3, 7]),     # 3-qubit connected triangle
        set([4, 8, 9])      # 3-qubit connected triangle
    ]
    
    print("Verifying subgraph connectivity...")
    verify_subgraphs(G, subgraphs)
    
    print("\nSubgraph partitioning:")
    for i, sg in enumerate(subgraphs):
        print(f"Subgraph {i+1}: {sorted(sg)}")
    
    # Visualize partitioning
    visualize_connected_subgraphs(G, subgraphs)
    
    # Create TopolAnsatz instance
    print("\nGenerating quantum circuits...")
    topol = TopolAnsatz(coupling_map)
    
    try:
        # Generate ansatz with specific connected subgraphs
        ansatz = topol.generate_partitioned_ansatz(
            partition_sizes=[4, 3, 3],
            depth=2,
            specific_subgraphs=subgraphs
        )
        
        print("\nCircuit Structure:")
        display(ansatz.draw())
        
        # Evaluate circuit quality
        metrics = topol.evaluate_ansatz(ansatz)
        print("\nCircuit Metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
        
    except ValueError as e:
        print(f"Error generating ansatz: {e}")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




