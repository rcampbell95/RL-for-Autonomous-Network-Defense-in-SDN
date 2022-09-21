from gym.spaces import Discrete, Tuple

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
import os

tf1, tf, tfv = try_import_tf()


class AutoregressiveActionModel(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(AutoregressiveActionModel, self).__init__( 
            obs_space, action_space, num_outputs, model_config, name
        )
        fcnet_size = 256
        num_nodes = int(float(os.getenv("RL_SDN_NETWORKSIZE").strip()))
        if action_space != Tuple([Discrete(num_nodes), Discrete(3)]):
            print(action_space)
            raise ValueError(f"This model only supports the [{num_nodes}, 3] action space. Action space {action_space}")

        # Inputs
        obs_input = tf.keras.layers.Input(shape=(obs_space.shape), name="obs_input")
        a1_input = tf.keras.layers.Input(shape=(1,), name="a1_input")
        ctx_input = tf.keras.layers.Input(shape=(num_outputs,), name="ctx_input")

        # Output of the model (normally 'logits', but for an autoregressive
        # dist this is more like a context/feature layer encoding the obs)
        flat_observation = tf.keras.layers.Flatten(name="flatten")(obs_input)

        context = tf.keras.layers.Dense(
            num_outputs,
            name="context",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(flat_observation)

        a1_hidden_1 = tf.keras.layers.Dense(
            fcnet_size,
            name="a1_hidden_1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(ctx_input)

        a1_hidden_2 = tf.keras.layers.Dense(
            fcnet_size,
            name="a1_hidden_2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(a1_hidden_1)

        # P(a1 | obs)
        a1_logits = tf.keras.layers.Dense(
            num_nodes,
            name="a1_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(a1_hidden_2)

        value_hidden_1 = tf.keras.layers.Dense(
            fcnet_size,
            name="value_hidden_1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(context)

        value_hidden_2 = tf.keras.layers.Dense(
            fcnet_size,
            name="value_hidden_2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(value_hidden_1)

        # V(s)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(value_hidden_2)

        # P(a2 | a1)
        # --note: typically you'd want to implement P(a2 | a1, obs) as follows:
        a2_context = tf.keras.layers.Concatenate(axis=1)(
             [ctx_input, a1_input])
        # a2_context = a1_input
        a2_hidden_1 = tf.keras.layers.Dense(
            fcnet_size,
            name="a2_hidden_1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(a2_context)

        a2_hidden_2 = tf.keras.layers.Dense(
            fcnet_size,
            name="a2_hidden_2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0),
        )(a2_hidden_1)

        a2_logits = tf.keras.layers.Dense(
            3,
            name="a2_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(a2_hidden_2)

        # Base layers
        self.base_model = tf.keras.Model(obs_input, [context, value_out])
        #self.register_variables(self.base_model.variables)
        self.base_model.summary()

        # Autoregressive action sampler
        self.action_model = tf.keras.Model(
            [ctx_input, a1_input], [a1_logits, a2_logits]
        )
        #self.register_variables(self.action_model.variables)
        self.action_model.summary()

    def forward(self, input_dict, state, seq_lens):
        context, self._value_out = self.base_model(input_dict["obs"])
        return context, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
