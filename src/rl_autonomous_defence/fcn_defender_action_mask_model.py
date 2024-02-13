import os

from gym.spaces import Dict

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
import gym
import numpy as np

import timeit

from rl_autonomous_defence.utils import (
    mask_defender_actions
)

tf1, tf, tfv = try_import_tf()



class FCNDefenderActionMaskModel(TFModelV2):
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

        self.base_model = FullyConnectedNetwork(
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
        start = timeit.default_timer()
        logits, state = self.base_model(input_dict)
        elapsed = timeit.default_timer() - start
        print(f"Model call time (Defender FCN): {elapsed}")

        start = timeit.default_timer()
        if self.model_config["custom_model_config"]["masked_actions"]:
            action_mask = mask_defender_actions(input_dict["obs"])

            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = tf.maximum(tf.math.log(action_mask,), -1e4)
            inf_mask = tf.reshape(inf_mask, logits.shape)
            logits = logits + inf_mask

        elapsed = timeit.default_timer() - start
        print(f"Time to mask actions (Defender FCN): {elapsed}")

        # Return masked logits.
        return logits, state

    def value_function(self):
        return self.base_model.value_function()

    def setup(self, config):
        super().setup(config)
