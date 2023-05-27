from gym.spaces import Dict

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.models.tf.attention_net import GTrXLNet
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
import numpy as np
import gym

from rl_autonomous_defence.utils import (
    mask_target_action,
    set_target_node_mask
)

tf1, tf, tfv = try_import_tf()


class AttackerActionMaskModel(TFModelV2):
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

        assert (
            isinstance(obs_space, gym.spaces.Box)
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        num_outputs = action_space[0].n + action_space[1].n

        self.base_model = VisionNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name = "_internal",
        )
        # disable action masking --> will likely lead to invalid actions
        #self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Compute the unmasked logits.
        # 
        #            model_config,
        #    name + "_internal",
        # 
        print(self.base_model.base_model.summary())
        logits, self._value_out = self.base_model(input_dict)

        action_mask = mask_actions(input_dict["obs"])

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), -1e4)
        inf_mask = tf.reshape(inf_mask, logits.shape)
        masked_logits = logits + inf_mask

        #masked_logits = logits * action_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.base_model.value_function()

    def get_initial_state(self):
        return []



def mask_actions(observation: tf.Tensor) -> tf.Tensor:
    observation = tf.squeeze(observation, axis=-1)

    agent_target_action_mask = tf.zeros((observation.shape[0], 3))
    diagonals = tf.linalg.diag_part(observation)
    is_compromised_indices = tf.where(tf.equal(diagonals, 2))

    agent_target_node_mask = tf.zeros((observation.shape[0], observation.shape[-1]))

    agent_target_node_mask = set_target_node_mask(observation, agent_target_node_mask)

    # explore topology
    agent_target_node_mask = tf.tensor_scatter_nd_update(agent_target_node_mask, is_compromised_indices, tf.ones(is_compromised_indices.get_shape()[0]))
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 2, 0)

    # scan vuln
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 0, 1)
    # compromise vuln
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 1, 2)
    return tf.keras.layers.concatenate([agent_target_node_mask, agent_target_action_mask])