from gym.spaces import Dict

from ray.rllib.models.tf.misc import normc_initializer

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

import gym

from rl_autonomous_defence.utils import (
    mask_attacker_actions
)

tf1, tf, tfv = try_import_tf()


class FCNAttackerActionMaskModel(TFModelV2):
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

        self.base_model = FullyConnectedNetwork(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name = "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        print(self.base_model.summary())

        logits, state = self.base_model(input_dict)

        if self.model_config["custom_model_config"]["masked_actions"]:
            action_mask = mask_attacker_actions(input_dict["obs"])

            #  Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = tf.maximum(tf.math.log(action_mask), -1e4)
            inf_mask = tf.reshape(inf_mask, logits.shape)
            logits = logits + inf_mask

        # Return masked logits.
        return logits, state

    def value_function(self):
        return self.base_model.value_function()

    def get_initial_state(self):
        return []
    

