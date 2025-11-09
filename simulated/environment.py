# simulated/environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulated.physics import FanControlEnv, Action

class FanControlGymEnv(gym.Env):
    def __init__(self, fan_env=None):
        super().__init__()
        self.fan_env = fan_env or FanControlEnv()
        self.action_space = spaces.Discrete(3)               # 0,1,2
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        obs = self.fan_env.reset()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action_enum = Action(action)                     # 0→DECREASE, 1→DO_NOTHING, 2→INCREASE
        obs, reward, terminated, info = self.fan_env.step(action_enum)
        info = self._get_info()
        return obs, reward, terminated, False, info

    def render(self, mode='human'):
        return self.fan_env.render(mode)

    def close(self):
        pass

    def _get_info(self):
        aqi = self.fan_env.pm25_to_aqi(self.fan_env.current_conc)
        rpm = self.fan_env._get_actual_rpm()
        airflow = self.fan_env._calculate_airflow(rpm)
        return {
            "aqi": int(aqi),
            "concentration": round(self.fan_env.current_conc, 2),
            "fan_speed": self.fan_env.current_fan_speed,
            "power": round(self.fan_env._calculate_power(), 2),
            "rpm": int(rpm),
            "airflow_m3h": round(airflow * 3600, 1),
            "ach": round(airflow * 3600 / self.fan_env.room_volume, 3),
        }