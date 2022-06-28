from gym.spaces import Dict

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.torch_utils import FLOAT_MIN
import gym

tf1, tf, tfv = try_import_tf()



class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        self.orig_space = getattr(obs_space, "original_space", obs_space)

        assert (
            isinstance(self.orig_space, gym.spaces.dict.Dict)
            and "action_mask" in self.orig_space.spaces
            and "observation" in self.orig_space.spaces
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.base_model = FullyConnectedNetwork(
            self.orig_space["observation"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        #self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        #print("INPUT DICT", input_dict["obs"])
        #print("STATE OF FORWARD" ,state)
        #print(input_dict["obs"]["observation"])

        # Compute the unmasked logits.
        logits, _ = self.base_model({"obs": input_dict["obs"]["observation"]})

        # If action masking is disabled, directly return unmasked logits
        #if self.no_masking:
        #    return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        #print(action_mask)
        #print(inf_mask.numpy())


        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.base_model.value_function()