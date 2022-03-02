from rl_autonomous_defence import utils
import numpy as np


def test_filter_candidate_neighbor_attacker_state():
    obs1 = [[2,  1, 1, 1,],
            [1,  0, 1, 0],
            [1,  1, 2, 1],
            [1,  0, 1, 0]]

    obs2 = [[2,  1, 0, 1,],
            [1, -1, 0, 1],
            [0,  0, 2, 0],
            [1,  1, 0, 0]]

    expected = set([1, 3])

    actual = utils.filter_candidate_neighbors(np.array(obs1), np.array(obs2))

    assert expected == actual


def test_set_neighbors():
    obs = [[2,  1, 1, 1],
            [1,  0, 1, 0],
            [1,  1, 2, 1],
            [1,  0, 1, 0]]

    expected = [[2,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  1, 2, 1],
            [0,  0, 1, 0]]

    target_node = 0
    value = 0

    actual = utils.set_neighbors(np.array(obs), target_node, value)

    assert np.array_equal(np.array(expected), np.array(actual))


def test_filter_candidate_neighbor_win_condition():
    obs1 = [[0,  0, 1, 0,],
            [0,  0, 1, 0],
            [1,  1, 2, 1],
            [0,  0, 1, 0]]

    obs2 = [[0,  1, 0, 1,],
            [1, -1, 0, 1],
            [0,  0, 2, 0],
            [1,  1, 0, 0]]

    expected = set()


    actual = utils.filter_candidate_neighbors(np.array(obs1), np.array(obs2))

    assert len(expected) == len(actual)


def test_build_clique():
    expected = [[0, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, -1]]

    num_nodes = 6
    start_position = 5

    actual = utils.build_clique(num_nodes, start_position)

    assert np.array_equal(actual, np.array(expected))


def test_build_grid():
    expected = [[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.],
               [ 1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
               [ 0.,  1., -1.,  1.,  0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.],
               [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]]

    num_nodes = 8
    start_position = 2

    actual = utils.build_grid(num_nodes, start_position)

    assert np.array_equal(actual, np.array(expected))
    

def test_build_linear():
    expected = [[ 0,  1,  0,  0,  0,  0,  0.,  1],
               [ 1,  0,  1,  0,  0,  0,  0,  0],
               [ 0,  1, -1,  1,  0,  0,  0,  0],
               [ 0,  0,  1,  0,  1,  0,  0,  0],
               [ 0,  0,  0,  1,  0,  1,  0,  0],
               [ 0,  0,  0,  0,  1,  0,  1,  0],
               [ 0,  0,  0,  0,  0,  1,  0,  1],
               [ 1,  0,  0,  0,  0,  0,  1,  0]]

    num_nodes = 8
    start_position = 2

    actual = utils.build_linear(num_nodes, start_position)

    assert np.array_equal(actual, np.array(expected))
    