from rl_autonomous_defence import utils
import numpy as np
import tensorflow as tf


def test_filter_candidate_neighbor_attacker_state():
    obs1 = [[2,  1, 1, 1,],
            [1,  0, 1, 0],
            [1,  1, 2, 1],
            [1,  0, 1, 0]]

    obs2 = [[2,  1, 0, 1,],
            [1,  3, 0, 1],
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
            [1,  3, 0, 1],
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
                [1, 1, 1, 1, 1, 3]]

    num_nodes = 6
    start_position = 5

    actual = utils.build_clique(num_nodes, start_position)

    assert np.array_equal(actual, np.array(expected))


def test_build_grid():
    expected = [[0,  1,  0,  0,  0,  0,  0,  1],
               [ 1,  0,  1,  0,  0,  0,  1,  0],
               [ 0,  1,  3,  1,  0,  1,  0,  0],
               [ 0,  0,  1,  0,  1,  0,  0,  0],
               [ 0,  0,  0,  1,  0,  1,  0,  0],
               [ 0,  0,  1,  0,  1,  0,  1,  0],
               [ 0,  1,  0,  0,  0,  1,  0,  1],
               [ 1,  0,  0,  0,  0,  0,  1,  0]]

    num_nodes = 8
    start_position = 2

    actual = utils.build_grid(num_nodes, start_position)

    assert np.array_equal(actual, np.array(expected))
    

def test_build_linear():
    expected = [[ 0,  1,  0,  0,  0,  0,  0,  0],
                [ 1,  0,  1,  0,  0,  0,  0,  0],
                [ 0,  1,  3,  1,  0,  0,  0,  0],
                [ 0,  0,  1,  0,  1,  0,  0,  0],
                [ 0,  0,  0,  1,  0,  1,  0,  0],
                [ 0,  0,  0,  0,  1,  0,  1,  0],
                [ 0,  0,  0,  0,  0,  1,  0,  1],
                [ 0,  0,  0,  0,  0,  0,  1,  0]]

    num_nodes = 8
    start_position = 2

    actual = utils.build_linear(num_nodes, start_position)

    assert np.array_equal(actual, np.array(expected))


def test_mask_target_action_attacker():
    node_states = tf.constant([[1, 0, 2, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 2, 0, 0, 0],
                               [2, 1, 2, 1, 2, 2, 1, 1]], dtype=tf.float32)
    target_action_mask = tf.zeros((3, 3))

    print(node_states)
    print(target_action_mask)

    expected = tf.constant([[0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=tf.float32)

    result = utils.mask_target_action(node_states, target_action_mask, 0, 1)

    assert np.array_equal(expected.numpy(), result.numpy())
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape

    expected = tf.constant([[0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 1]], dtype=tf.float32)

    result = utils.mask_target_action(node_states, target_action_mask, 1, 2)

    assert np.array_equal(expected.numpy(), result.numpy())
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape

    expected = tf.constant([[1, 0, 0],
                            [1, 0, 0],
                            [1, 0, 0]], dtype=tf.float32)


    result = utils.mask_target_action(node_states, target_action_mask, 2, 0)

    assert np.array_equal(expected.numpy(), result.numpy())
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape



def test_mask_target_action_defender():
    node_states = tf.constant([[1, 0, 2, 0, 0, 0, 3, 0],
                               [0, 0, 0, 0, 3, 0, 0, 0],
                               [2, 2, 2, 2, 2, 2, 2, 3]], dtype=tf.float32)
    target_action_mask = tf.zeros((3, 3))

    print(node_states)
    print(target_action_mask)

    expected = tf.constant([[1, 0, 0],
                            [1, 0, 0],
                            [0, 0, 0]], dtype=tf.float32)

    result = utils.mask_target_action(node_states, target_action_mask, 0, 0)

    assert np.array_equal(expected.numpy(), result.numpy())
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape

    expected = tf.constant([[0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0]], dtype=tf.float32)

    result = utils.mask_target_action(node_states, target_action_mask, 2, 1)

    assert np.array_equal(expected.numpy(), result.numpy())
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape

    expected = tf.constant([[0, 0, 1],
                            [0, 0, 1],
                            [0, 0, 0]], dtype=tf.float32)


    result = utils.mask_target_action(node_states, target_action_mask, 0, 2)

    assert np.array_equal(expected.numpy(), result.numpy())
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape

    
def test_target_node_mask_attacker():
    batch_size = 1
    attacker_obs = tf.constant([[[0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 1, 2, 0, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 2, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0]]], dtype=tf.float32)
    attacker_obs = tf.constant(attacker_obs)
    target_node_mask = tf.zeros((batch_size, 8))

    expected = tf.constant([[0, 1, 1, 0, 1, 1, 1, 0]], dtype=tf.float32)

    result = utils.set_target_node_mask(attacker_obs, target_node_mask)

    assert np.array_equal(expected.numpy(), result.numpy())
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape

