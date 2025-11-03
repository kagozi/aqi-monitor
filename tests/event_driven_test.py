import os
import csv
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
from stable_baselines3 import PPO, DQN, A2C

# Import your Gym wrapper and physics env
from simulated.environment import FanControlGymEnv

# ---------- Policies ----------
best_model = "model_100000"
class Policy:
    """Simple interface for a policy."""
    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError
    
class DQNPolicy(Policy):
    def __init__(self, model_path: str):
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"DQN model not found at: {model_path}")
        self.model = DQN.load(f"{CHECKPOINT_DIR_DQN}/model_100000")

    def act(self, obs: np.ndarray) -> int:
        # Deterministic for evaluation
        action, _ = self.model.predict(obs, deterministic=True)
        # SB3 returns a numpy scalar; ensure int
        return int(action)

class PPOPolicy(Policy):
    def __init__(self, model_path: str):
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"PPO model not found at: {model_path}")
        self.model = PPO.load(f"{CHECKPOINT_DIR_PPO}/model_100000")

    def act(self, obs: np.ndarray) -> int:
        # Deterministic for evaluation
        action, _ = self.model.predict(obs, deterministic=True)
        # SB3 returns a numpy scalar; ensure int
        return int(action)
    
class A2CPolicy(Policy):
    def __init__(self, model_path: str):
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"A2C model not found at: {model_path}")
        self.model = A2C.load(f"{CHECKPOINT_DIR_A2C}/model_100000")

    def act(self, obs: np.ndarray) -> int:
        # Deterministic for evaluation
        action, _ = self.model.predict(obs, deterministic=True)
        # SB3 returns a numpy scalar; ensure int
        return int(action)

class RuleBasedPolicy(Policy):
    """
    Rule-based controller modeled after your MQTT on_message snippet.
    - Increase if AQI > aqi_target and fan < 0.9
    - Decrease if AQI <= 2 and fan >= 0.53
    - Else do nothing
    """
    def __init__(self, aqi_target: float = 3.0):
        self.aqi_target = aqi_target

    def act(self, obs: np.ndarray) -> int:
        # Assumes obs = [normalized_aqi, fan_speed] in [0,1]
        # If your normalization maps AQI∈[1..5] -> [0..1], decode for rules:
        norm_aqi, fan_speed = float(obs[0]), float(obs[1])
        # Try to map back to real AQI ≈ 1..5 if needed
        # If your pipeline already expects normalized AQI for rules, just use norm values directly.
        # Here we map normalized -> [1..5] (best-effort default).
        est_aqi = 1.0 + 4.0 * norm_aqi

        if est_aqi > self.aqi_target and fan_speed < 0.9:
            return 2  # INCREASE
        elif est_aqi <= 2.0 and fan_speed >= 0.53:
            return 0  # DECREASE
        else:
            return 1  # DO_NOTHING

# ---------- Event-Driven Runner ----------

class EventDrivenTestRunner:
    """
    Simulates an event-driven loop:
      - Pulls 'sensor' readings from the env each tick (aqi, fan_speed, power, rpm, etc.)
      - Feeds them to a policy (PPO or Rule-based)
      - Applies the action
      - Logs to CSV
    """

    def __init__(self, env: FanControlGymEnv, policy: Policy, csv_path: str, seed: Optional[int] = 42):
        self.env = env
        self.policy = policy
        self.csv_path = csv_path
        self.seed = seed
        self.energy_joules_cum = 0.0

        # Try to get sim step in seconds from the underlying physics env
        self.dt = getattr(self.env.fan_env, "dt", 1.0)

    def _set_initial_aqi(self, target_aqi: float = 5.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ensure the episode starts at AQI ~ 5.
        This tries:
          1) env.fan_env.reset(initial_aqi=...)
          2) setting env.fan_env.current_aqi then rebuilding obs if possible
        Falls back to the wrapper reset if neither is supported.
        """
        try:
            obs, info = self.env.reset(seed=self.seed)
            # Attempt physics-level reset with initial AQI if available
            try:
                self.env.fan_env.reset(initial_aqi=target_aqi)
                # If physics env exposes a method to compute observation, use it
                if hasattr(self.env.fan_env, "_get_observation"):
                    obs = self.env.fan_env._get_observation()
            except TypeError:
                # Fallback: set the attribute directly
                if hasattr(self.env.fan_env, "current_aqi"):
                    self.env.fan_env.current_aqi = target_aqi
                    if hasattr(self.env.fan_env, "_get_observation"):
                        obs = self.env.fan_env._get_observation()
            # Rebuild info for logging
            info = self.env.get_current_state_info()
            return obs, info
        except Exception:
            # Last resort: plain reset
            return self.env.reset(seed=self.seed)

    def _csv_header(self):
        return [
            "t_sec", "step", "action",
            "obs_aqi_norm", "obs_fan",
            "reward",
            "aqi", "fan_speed", "power_W", "rpm", "aer", "settling_rate", "voltage_V",
            "energy_J_cum", "terminated", "truncated"
        ]

    def run(self, minutes: float = 30.0, start_aqi: float = 5.0) -> None:
        total_steps = int((minutes * 60.0) / self.dt)
        obs, info = self._set_initial_aqi(start_aqi)

        # Prepare CSV
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._csv_header())

            terminated = False
            truncated = False

            for step in range(total_steps):
                # "Sensor event" (obs + info). Policy uses obs.
                action = self.policy.act(obs)

                # Step sim
                time.sleep(5)
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Energy accumulation (J = W * s)
                power = float(info.get("power", 0.0))
                self.energy_joules_cum += power * self.dt

                # Log a row
                writer.writerow([
                    round(step * self.dt, 3), step, action,
                    float(obs[0]), float(obs[1]),
                    float(reward),
                    float(info.get("aqi", 0.0)),
                    float(info.get("fan_speed", 0.0)),
                    float(info.get("power", 0.0)),
                    float(info.get("rpm", 0.0)),
                    float(info.get("aer", 0.0)),
                    float(info.get("settling_rate", 0.0)),
                    float(info.get("voltage", 0.0)),
                    float(self.energy_joules_cum),
                    bool(terminated), bool(truncated),
                ])

                if terminated or truncated:
                    break

        # Clean up
        try:
            self.env.close()
        except Exception:
            pass


# ---------- CLI-style entrypoint ----------

def run_all(
    ppo_model_path: str,
    dqn_model_path: str,
    a2c_model_path: str,
    out_dir: str = "./test_runs",
    minutes: float = 30.0,
    start_aqi: float = 5.0,
    seed: int = 42
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"PPO model: {ppo_model_path} \nDQN model: {dqn_model_path} \nA2C model: {a2c_model_path}")
    # Create one shared environment per run for isolation / reproducibility
    # (You can reuse if you prefer.)
    # --- RL (PPO) run ---
    env_rl = FanControlGymEnv()
    rl_policy = PPOPolicy(ppo_model_path)
    runner_rl = EventDrivenTestRunner(env_rl, rl_policy, os.path.join(out_dir, "ppo_eval.csv"), seed=seed)
    runner_rl.run(minutes=minutes, start_aqi=start_aqi)

    # --- Rule-based run ---
    env_rb = FanControlGymEnv()
    rb_policy = RuleBasedPolicy(aqi_target=3.0)
    runner_rb = EventDrivenTestRunner(env_rb, rb_policy, os.path.join(out_dir, "rule_based_eval.csv"), seed=seed)
    runner_rb.run(minutes=minutes, start_aqi=start_aqi)
    
    # -- RL (DQN) run ---
    env_dqn = FanControlGymEnv()
    dqn_policy = DQNPolicy(dqn_model_path)
    runner_dqn = EventDrivenTestRunner(env_dqn, dqn_policy, os.path.join(out_dir, "dqn_eval.csv"), seed=seed)
    runner_dqn.run(minutes=minutes, start_aqi=start_aqi)
    
    # -- RL (A2C) run ---
    env_a2c = FanControlGymEnv()
    a2c_policy = A2CPolicy(a2c_model_path)
    runner_a2c = EventDrivenTestRunner(env_a2c, a2c_policy, os.path.join(out_dir, "a2c_eval.csv"), seed=seed)
    runner_a2c.run(minutes=minutes, start_aqi=start_aqi)
    

    print(f"[DONE] Wrote:\n  - {os.path.join(out_dir, 'ppo_eval.csv')}\n  - {os.path.join(out_dir, 'rule_based_eval.csv')} \n  - {os.path.join(out_dir, 'dqn_eval.csv')} \n  - {os.path.join(out_dir, 'a2c_eval.csv')}  ")

if __name__ == "__main__":
    # Run a 30-minute test of PPO vs. Rule-based:
    #   python tests/event_driven_test.py
    #
    # Make sure your PPO model exists at this path.
    CHECKPOINT_DIR_PPO = "./ppo/train"
    
    CHECKPOINT_DIR_DQN = "./dqn/train"
    
    CHECKPOINT_DIR_A2C = "./a2c/train"

    run_all(CHECKPOINT_DIR_PPO, CHECKPOINT_DIR_DQN, CHECKPOINT_DIR_A2C, out_dir="./test_runs", minutes=30.0, start_aqi=5.0, seed=42)
