import pytest
import networkx as nx
from src.topolansatz.core.topology import TopologyHandler

@pytest.fixture
def sample_coupling_map():
    return [(0,1), (1,2), (2,3), (3,4), (4,5), (1,4)]

@pytest.fixture
def handler(sample_coupling_map):
    return TopologyHandler(sample_coupling_map)

def test_initialization(handler, sample_coupling_map):
    assert len(handler.graph.edges()) == len(sample_coupling_map)

def test_get_connected_subgraphs(handler):
    # Test for 3-qubit subgraphs
    subgraphs_3 = handler.get_connected_subgraphs(3)
    assert len(subgraphs_3) > 0
    assert all(len(g.nodes()) == 3 for g in subgraphs_3)
    assert all(nx.is_connected(g) for g in subgraphs_3)

    # Test for 4-qubit subgraphs
    subgraphs_4 = handler.get_connected_subgraphs(4)
    assert len(subgraphs_4) > 0
    assert all(len(g.nodes()) == 4 for g in subgraphs_4)

def test_normalize_indices(handler):
    subgraphs = handler.get_connected_subgraphs(3)
    norm_graph, index_map = handler.normalize_indices(subgraphs[0])

    assert len(norm_graph.nodes()) == 3
    assert list(norm_graph.nodes()) == [0, 1, 2]
    assert len(index_map) == 3

def test_invalid_size(handler):
    with pytest.raises(ValueError):
        handler.get_connected_subgraphs(7)

def test_verify_placement(handler):
    # Get a valid subgraph
    subgraphs = handler.get_connected_subgraphs(3)
    original_indices = set(subgraphs[0].nodes())
    assert handler.verify_placement(original_indices)

    # Test invalid placement
    invalid_indices = {0, 2, 4}  # Not connected
    assert not handler.verify_placement(invalid_indices)
