import tensorflow as tf
from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution

from rl_autonomous_defence.utils import (
    mask_target_action,
    set_target_node_mask
)

class AttackerAutoregressiveDistribution(ActionDistribution):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        # First, sample a1.
        # mask first action
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        # mask second action
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()

        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def logp(self, actions):
        a1, a2 = actions[:, 0], actions[:, 1]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a1_logits, a2_logits = self.model.action_model([self.inputs, a1_vec])
        return Categorical(a1_logits).logp(a1) + Categorical(a2_logits).logp(a2)

    def sampled_action_logp(self):
        return tf.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        assert isinstance(self.inputs, tf.Tensor)
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        return a1_terms + a2_terms

    def _a1_distribution(self):
        BATCH = tf.shape(self.inputs)[0]
        a1_logits, _ = self.model.action_model([self.inputs, tf.zeros((BATCH, 1))])

        # mask after sampling
        #masked_logits = self._mask_a1(a1_logits)
        a1_dist = Categorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        _, a2_logits = self.model.action_model([self.inputs, a1_vec])
        a2_dist = Categorical(a2_logits)
        return a2_dist

    def _mask_a1(self, logits):
        observation = self.inputs["obs"]

        diagonals = tf.linalg.diag_part(observation)
        is_compromised_indices = tf.where(tf.equal(diagonals, 2))

        agent_target_node_mask = tf.zeros((observation.shape[0], observation.shape[-1]))

        agent_target_node_mask = set_target_node_mask(observation, agent_target_node_mask)

        # explore topology
        agent_target_node_mask = tf.tensor_scatter_nd_update(agent_target_node_mask, is_compromised_indices, tf.ones(is_compromised_indices.get_shape()[0]))

        inf_mask = tf.maximum(tf.math.log(agent_target_node_mask), -1e4)
        masked_logits = logits + inf_mask

        return masked_logits

    def _mask_a2(self, logits):
        observation = self.inputs["obs"]
        return None

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 128  # controls model output feature vector size

