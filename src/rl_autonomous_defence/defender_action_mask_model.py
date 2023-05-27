import os

from gym.spaces import Dict

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
import gym
import numpy as np

from rl_autonomous_defence.utils import (
    mask_target_action
)

tf1, tf, tfv = try_import_tf()



class DefenderActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        # self.orig_space = getattr(obs_space, "original_space", obs_space)

        assert (
            isinstance(obs_space, gym.spaces.Box)
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.base_model = VisionNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        #self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Compute the unmasked logits.
        logits, _ = self.base_model(input_dict)
        masked_logits = logits
        # If action masking is disabled, directly return unmasked logits
        #if self.no_masking:
        #    return logits, state

        action_mask = mask_actions(input_dict["obs"])

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask,), -1e4)
        inf_mask = tf.reshape(inf_mask, logits.shape)
        masked_logits = logits + inf_mask

        #masked_logits = logits * action_mask


        #print(action_mask)
        #print(inf_mask.numpy())


        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.base_model.value_function()

    def setup(self, config):
        super().setup(config)


def mask_actions(observation: tf.Tensor) -> tf.Tensor:
    observation = tf.squeeze(observation, axis=-1)

    agent_target_action_mask = tf.zeros((observation.shape[0], 3))
    diagonals = tf.linalg.diag_part(observation)
    agent_target_node_mask = tf.ones((observation.shape[0], observation.shape[-1]))

    is_critical_indices = tf.where(tf.equal(diagonals, 3))

    agent_target_node_mask = tf.tensor_scatter_nd_update(agent_target_node_mask, is_critical_indices, tf.zeros(is_critical_indices.get_shape()[0]))

    current_size = int(float(os.environ["RL_SDN_NETWORKSIZE"]))
    max_size = int(os.environ.get("RL_SDN_NETWORKSIZE-MAX", str(current_size)))
    assert isinstance(current_size, int)
    assert isinstance(max_size, int)

    if max_size > current_size:
        indices = [[[j, i] for i in range(current_size, max_size)] for j in range(agent_target_node_mask.shape[0])]
        updates = [[0 for i in range(len(indices[0]))] for j in range(agent_target_node_mask.shape[0])]

        indices = tf.constant(indices, dtype=tf.int32)
        updates = tf.constant(updates, dtype=tf.float32)

        agent_target_node_mask = tf.tensor_scatter_nd_update(agent_target_node_mask, indices, updates)

    # migrate node
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 0, 2)
    # check status
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 0, 0)
    # isolate node
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 2, 1)
    return tf.keras.layers.concatenate([agent_target_node_mask, agent_target_action_mask])

