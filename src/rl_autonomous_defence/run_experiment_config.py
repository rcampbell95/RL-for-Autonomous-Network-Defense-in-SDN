import subprocess
import os
import json
import itertools


if __name__ == "__main__":
    NUM_TRIALS = 1
    EXPERIMENT_NAME = os.getenv(f"RL_SDN_EXPERIMENT_NAME").strip()

    config_path = os.getenv(f"RL_SDN_EXPERIMENT_CONFIG").strip()
    config = json.load(open(config_path))

    param_grid = itertools.product(*config.values())
    param_names = list(config.keys())

    for params_values in param_grid:
        params = {}
        for i, param in enumerate(params_values):
            params[param_names[i]] = param
            os.environ[f"RL_SDN_{param_names[i].upper()}"] = str(param)
            assert os.getenv(f"RL_SDN_{param_names[i].upper()}") == str(param)

        trial_name = "_".join([f"{key}={str(value)}" for key, value in params.items()])
        os.environ[f"RL_SDN_EXPERIMENT_DIRECTORY"] = f"./ray_results/{EXPERIMENT_NAME}/{trial_name}"
        for i in range(NUM_TRIALS):
            subprocess.run(["python", "./src/rl_autonomous_defence/tune.py"])