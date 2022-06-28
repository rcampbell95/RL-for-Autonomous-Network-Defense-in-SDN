import ray.rllib.agents.ppo as ppo
import os

class CheckpointWrapperPPO(ppo.PPOTrainer):
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_dirs = {}
        #for agent in ["attacker", "defender"]:#, "attacker_v0", "defender_v0"]:
        #    export_directory = os.path.join(checkpoint_dir, agent)
        #    self.get_policy(agent).export_model(export_directory)
        #    #self.export_policy_checkpoint(export_directory, agent, policy_id=agent)
        #    checkpoint_dirs[agent] = export_directory

        return checkpoint_dirs

if __name__ == "__main__":

    trainer = CheckpointWrapperPPO()
