from ray.rllib.models.tf.misc import normc_initializer

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification

import numpy as np
import gym
import pandas as pd

from tensorflow.keras.utils import to_categorical

from rl_autonomous_defence.utils import (
    mask_attacker_actions
)

tf1, tf, tfv = try_import_tf()


class GCNAttackerActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        #self.orig_space = getattr(obs_space, "original_space", obs_space)
        self.model_config = model_config
        assert (
            isinstance(obs_space, gym.spaces.Box)
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        batch_size = 128
        if obs_space.shape[0] == obs_space.shape[1]:
            self.num_nodes = obs_space.shape[0]
        else:
            raise Exception(f"Mismatch in shape of observation space {obs_space.shape[0]} {obs_space.shape[1]}")

        self.num_states = 4
        graphs = np.random.randint(0, 2, size=(batch_size, self.num_nodes, self.num_nodes))
        features = np.random.randint(0, 2, size=(batch_size, self.num_nodes, self.num_states))

        sg_graphs = []
        for i, graph in enumerate(graphs):
            np.fill_diagonal(graph, 0)

            edge_coords = np.where(graph == 1)
            edges = pd.DataFrame({"source": edge_coords[0], "target": edge_coords[1]})

            sg_graph = sg.StellarGraph(features[i], edges)
            sg_graphs.append(sg_graph)

        self.generator = PaddedGraphGenerator(graphs=sg_graphs)

        gc_model = GCNSupervisedGraphClassification(
            layer_sizes=[64, 64],
            activations=["tanh", "tanh"],
            generator=self.generator,
            dropout=0,
            kernel_initializer=normc_initializer(1.0)
        )
        x_inp, x_out = gc_model.in_out_tensors()

        f1 = tf.keras.layers.Dense(128, name="fc_1", activation="tanh", kernel_initializer=normc_initializer(1.0))(x_out)
        fcv1 = tf.keras.layers.Dense(128, name="fc_value_1", activation="tanh", kernel_initializer=normc_initializer(1.0))(x_out)

        f2 = tf.keras.layers.Dense(128, name="fc_2", activation="tanh", kernel_initializer=normc_initializer(1.0))(f1)
        fcv2 = tf.keras.layers.Dense(128, name="fc_value_2", activation="tanh", kernel_initializer=normc_initializer(1.0))(fcv1)

        fc_out = tf.keras.layers.Dense(self.num_outputs, name="fc_out", activation="linear", kernel_initializer=normc_initializer(0.01))(f2)
        value_out = tf.keras.layers.Dense(1, name="fc_value_out", activation="linear", kernel_initializer=normc_initializer(0.01))(fcv2)

        self.base_model = tf.keras.Model(inputs=x_inp, outputs=[fc_out, value_out])

        # disable action masking --> will likely lead to invalid actions
        #self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        print(self.base_model.summary())

        logits, self._value_out = self.call_gc_model(input_dict, state)
        
        if self.model_config["custom_model_config"]["masked_actions"]:
            action_mask = mask_attacker_actions(input_dict["obs"])

            #  Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = tf.maximum(tf.math.log(action_mask), -1e4)
            inf_mask = tf.reshape(inf_mask, logits.shape)
            logits = logits + inf_mask

        return logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def get_initial_state(self):
        return []
    
    def call_gc_model(self, input_dict, state):
        obs_batch = input_dict["obs"]

        if not isinstance(obs_batch, np.ndarray):
            obs_batch = obs_batch.numpy()
        
        batch_size = obs_batch.shape[0]
        features = obs_batch.diagonal(axis1=1, axis2=2)
        one_hot_features = to_categorical(features, num_classes=4)

        sg_graphs = []
        for i, obs in enumerate(obs_batch):
            np.fill_diagonal(obs, 0)
##
            edge_coords = np.where(obs == 1)
            edges = pd.DataFrame({"source": edge_coords[0], "target": edge_coords[1]})

            sg_graph = sg.StellarGraph(one_hot_features[i], edges)
            sg_graphs.append(sg_graph)
        

        generator = PaddedGraphGenerator(graphs=sg_graphs)
        train_gen = generator.flow(sg_graphs, batch_size=batch_size)

        logits, value = self.base_model(train_gen.__getitem__(0)[0])

        return logits, value
