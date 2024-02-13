from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.policy.view_requirement import ViewRequirement

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import one_hot

import spektral

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification

from tensorflow.keras.utils import to_categorical

from joblib import Parallel, delayed

import numpy as np
import gym
import pandas as pd

import networkx as nx

import timeit

from rl_autonomous_defence.utils import (
    mask_defender_actions,
    normalize_batch
)

tf1, tf, tfv = try_import_tf()


class GCNDefenderActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        assert (
            isinstance(obs_space, gym.spaces.Box)
        )

        super(GCNDefenderActionMaskModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        batch_size = 32
        self.num_states = 4
        self.num_actions = 3
        self.num_frames = model_config["custom_model_config"]["num_frames"]
        self.num_outputs = num_outputs
        self.action_space = action_space
        
        if obs_space.shape[0] == obs_space.shape[1]:
            self.num_nodes = obs_space.shape[0]
        else:
            raise Exception(f"Mismatch in shape of observation space {obs_space.shape[0]} {obs_space.shape[1]}")
        nx_graphs = [nx.generators.random_regular_graph(3, self.num_nodes)for i in range(batch_size)]
        graphs = [nx.to_numpy_array(nx_graph, dtype=np.float32) for nx_graph in nx_graphs]

        features = tf.constant(np.random.randint(0, 2, size=(batch_size, self.num_nodes, 1)))
        features = tf.keras.utils.to_categorical(features, num_classes=self.num_states)

        sg_graphs = []
        for i, graph in enumerate(graphs):
            np.fill_diagonal(graph, 0)

            edge_coords = np.where(graph == 1)
            edges = pd.DataFrame({"source": edge_coords[0], "target": edge_coords[1]})

            sg_graph = sg.StellarGraph(features[i], edges)
            sg_graphs.append(sg_graph)
        self.generator = PaddedGraphGenerator(graphs=sg_graphs)

        gcn_layer_units = self.model_config["custom_model_config"]["gcn_hiddens"]
        dense_layer_units = self.model_config["custom_model_config"]["dense_hiddens"]
        attention_layer_units = 32 #self.model_config["custom_model_config"]["attention_hiddens"]

        pooling_policy = spektral.layers.GlobalAttentionPool(attention_layer_units, kernel_initializer=normc_initializer(1.0)) 
        pooling_value = spektral.layers.GlobalAttentionPool(attention_layer_units, kernel_initializer=normc_initializer(1.0))

        gc_model_1 = GCNSupervisedGraphClassification(
            layer_sizes=gcn_layer_units,
            activations=["tanh"] * len(gcn_layer_units),
            dropout=0,
            bias=True,
            generator=self.generator,
            kernel_initializer=normc_initializer(1.0),
            pooling=pooling_policy
        )
        gc_model_2 = GCNSupervisedGraphClassification(
            layer_sizes=gcn_layer_units,
            activations=["tanh"] * len(gcn_layer_units),
            dropout=0,
            bias=True,
            generator=self.generator,
            kernel_initializer=normc_initializer(1.0),
            pooling=pooling_value
        )
        x_inp_policy, x_out_policy = gc_model_1.in_out_tensors()

        gc_model_inputs = [
            [tf.keras.layers.Input(gc_input.shape[1:])
            for gc_input in x_inp_policy] for _ in range(self.num_frames)
        ]

        actions = tf.keras.layers.Input(shape=(self.num_frames, num_outputs))
        actions_reshaped = tf.keras.layers.Flatten()(
            actions
        )

        gc_policy_embeddings = [gc_model_1(gc_model_input) for gc_model_input in gc_model_inputs]
        gc_value_embeddings = [gc_model_2(gc_model_input) for gc_model_input in gc_model_inputs]

        input_policy = tf.keras.layers.Concatenate(axis=-1)(gc_policy_embeddings + [actions_reshaped])
        input_value = tf.keras.layers.Concatenate(axis=-1)(gc_value_embeddings + [actions_reshaped])

        f1 = tf.keras.layers.Dense(dense_layer_units, name="fc_1", activation="tanh", kernel_initializer=normc_initializer(1.0))(input_policy)
        fcv1 = tf.keras.layers.Dense(dense_layer_units, name="fc_value_1", activation="tanh", kernel_initializer=normc_initializer(1.0))(input_value)

        f2 = tf.keras.layers.Dense(dense_layer_units, name="fc_2", activation="tanh", kernel_initializer=normc_initializer(1.0))(f1)
        fcv2 = tf.keras.layers.Dense(dense_layer_units, name="fc_value_2", activation="tanh", kernel_initializer=normc_initializer(1.0))(fcv1)

        fc_out = tf.keras.layers.Dense(self.num_outputs, name="fc_out", activation="linear", kernel_initializer=normc_initializer(0.01))(f2)
        value_out = tf.keras.layers.Dense(1, name="fc_value_out", activation="linear", kernel_initializer=normc_initializer(0.01))(fcv2)

        self.base_model = tf.keras.Model(inputs=gc_model_inputs + [actions], outputs=[fc_out, value_out])

        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(self.num_frames - 1), space=obs_space
        )

        self.view_requirements["prev_n_actions"] = ViewRequirement(
            data_col="actions",
            shift="-{}:-1".format(self.num_frames),
            space=self.action_space,
        )


    def forward(self, input_dict, state, seq_lens):
        logits, self._value_out = self.call_gc_model(input_dict, state)

        return logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def get_initial_state(self):
        return []
    
    def call_gc_model(self, input_dict, state):
        obs_batch_tensor = input_dict["prev_n_obs"]
        actions_labels = input_dict["prev_n_actions"]


        #bool_mask_actions = tf.equal(actions_labels, 0)
        #is_zeros = tf.reduce_all(bool_mask_actions)
        if len(actions_labels.shape) == 2:
            actions = tf.expand_dims(actions_labels, axis=1)

        #if is_zeros:
        #    if len(actions_labels.shape) == 2:
        #        actions_labels = tf.zeros((actions_labels.shape[0], self.num_frames))
        #    elif len(actions_labels.shape) == 3:
        #        actions_labels = tf.zeros((actions_labels.shape[0], self.num_frames, actions_labels.shape[-1]))

        if isinstance(self.action_space, gym.spaces.Discrete): 
            actions = one_hot(actions_labels, self.action_space)
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            one_hot_node_targets = tf.keras.utils.to_categorical(actions_labels[:, :, 0], num_classes=self.num_nodes)
            one_hot_action_targets = tf.keras.utils.to_categorical(actions_labels[:, :, 1], num_classes=self.num_actions)
            actions = tf.concat([one_hot_node_targets, one_hot_action_targets], axis=-1)


        if not isinstance(obs_batch_tensor, np.ndarray):
            obs_batch_ndarray = obs_batch_tensor.numpy()
        else:
            obs_batch_ndarray = obs_batch_tensor
        
        batch_size = obs_batch_ndarray.shape[0]
        
        start = timeit.default_timer()


        features = tf.linalg.diag_part(obs_batch_tensor)
        one_hot_features = to_categorical(features, num_classes=self.num_states)

        self_loop_array = tf.ones(obs_batch_tensor.shape[:-1])
        obs_batch_edges = tf.linalg.set_diag(obs_batch_tensor, self_loop_array)

        normalized_graphs = normalize_batch(obs_batch_edges)
        #normalized_graphs = degree_matrices - obs_batch_edges

        elapsed = timeit.default_timer() - start
        print(f"Time to preprocess batch (Defender): {elapsed}")
        start = timeit.default_timer()

        logits, value = self.base_model([
            [[one_hot_features[:, i, :, :],
            np.ones((batch_size, self.num_nodes)),
            normalized_graphs[:, i, :, :]] for i in range(self.num_frames)]
            ,
            actions
        ])

        print(f"Model call time (Defender): {timeit.default_timer() - start}")

        if self.model_config["custom_model_config"]["masked_actions"]:
            current_obs_features = tf.linalg.diag_part(input_dict["obs"])
            action_mask = mask_defender_actions(input_dict["obs"], current_obs_features)

            #  Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = tf.maximum(tf.math.log(action_mask), -1e4)
            inf_mask = tf.reshape(inf_mask, logits.shape)
            logits = logits + inf_mask


        return logits, value
