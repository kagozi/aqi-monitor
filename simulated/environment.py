# File: simulated/environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulated.physics import FanControlEnv, Action

class FanControlGymEnv(gym.Env):
    """Gymnasium wrapper for FanControlEnv"""
    
    def __init__(self, fan_env):
        super(FanControlGymEnv, self).__init__()
        
        # Reference to the physics simulation environment
        self.fan_env = fan_env
        
        # Action space (same discrete actions)
        self.action_space = spaces.Discrete(len(self.fan_env.action_space))
        
        # Observation space: [AQI, fan_speed]
        self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )

        
        # Reward range
        self.reward_range = (-float('inf'), 0.0)
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset the physics simulation
        obs = self.fan_env.reset()
        
        # Get initial energy usage
        energy = self._get_normalized_energy()
        
        # Combine observations
        full_obs = np.array([obs[0], obs[1]])
        
        return full_obs, {}
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        """
        Execute one time step in the environment
        
        Args:
            action: 0 (DECREASE), 1 (DO_NOTHING), 2 (INCREASE)
        """
        # Convert Gym action to FanControlEnv action
        action_enum = list(Action)[action]
        
        # Step the physics simulation
        obs, reward, done, info = self.fan_env.step(action_enum)
        
        # Get current energy usage
        energy = self._get_normalized_energy()
        
        # Combine observations
        full_obs = np.array([obs[0], obs[1]])
        
        return full_obs, reward, done, False, info
    
    def _get_normalized_energy(self):
        """Get normalized energy usage (0-1 scale)"""
        # Get actual power in watts
        power = self.fan_env._calculate_power()
        
        # Normalize based on max expected power (stall current * voltage)
        max_power = self.fan_env.stall_current * self.fan_env.voltage
        return min(power / max_power, 1.0)
    
    def render(self, mode='human'):
        """Render the environment"""
        return self.fan_env.render(mode)
    
    def close(self):
        """Clean up resources"""
        pass

# Example usage
if __name__ == "__main__":
    # Create the physics simulation environment
    physics_env = FanControlEnv()
    
    # Create the Gym environment wrapper
    gym_env = FanControlGymEnv(physics_env)
    
    # Test the Gym interface
    obs, _ = gym_env.reset()
    print(f"Initial observation: {obs}")
    
    for _ in range(3):
        action = gym_env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = gym_env.step(action)
        print(f"Action: {action} | Observation: {obs} | Reward: {reward:.2f}")
        
        if terminated or truncated:
            break