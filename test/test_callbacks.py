#from rl_autonomous_defence import callbacks
from unittest.mock import Mock

"""
def test_on_train_result():
    result = {"hist_stats": {"policy_reward_mean/attacker": [1,1,1,0]}}

    new_policy = Mock()
    new_policy.set_state.return_value = None

    attacker_policy = Mock()
    attacker_policy.get_state.return_value = {"attacker": "policy"}

    trainer = Mock()
    trainer.iteration = 1
    trainer.add_policy.return_value = new_policy
    trainer.get_policy.return_value = attacker_policy
    trainer.workers.sync_weights.return_value = None

    selfPlayCallback = callbacks.SelfPlayCallback()

    selfPlayCallback.on_train_result(trainer=trainer,  result=result)

    attacker_policy.get_state.assert_called()

    trainer.add_policy.assert_called_once()
    trainer.get_policy.assert_called()
    trainer.workers.sync_weights.assert_called_once()

    assert result["league_size"] == 2

    new_policy.set_state.assert_called_with({'attacker': "policy"})
    
"""