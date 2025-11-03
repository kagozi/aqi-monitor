# File: simulated/environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulated.physics import FanControlEnv, Action

class FanControlGymEnv(gym.Env):
    """Gymnasium wrapper for FanControlEnv"""
    
    def __init__(self, fan_env=None):
        super(FanControlGymEnv, self).__init__()
        
        # Create physics simulation environment if not provided
        if fan_env is None:
            self.fan_env = FanControlEnv()
        else:
            self.fan_env = fan_env
        
        # Action space: 0=DECREASE, 1=DO_NOTHING, 2=INCREASE
        self.action_space = spaces.Discrete(len(self.fan_env.action_space))
        
        # Observation space: [normalized_AQI, fan_speed]
        # Both values are normalized to [0, 1] range
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Reward range (can be negative due to energy costs)
        self.reward_range = (-float('inf'), float('inf'))  # Fixed: should allow positive rewards
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Reset the physics simulation
        obs = self.fan_env.reset()
        
        # Get initial info
        info = self.get_current_state_info()
        
        return obs, info
    
    def step(self, action):
        """
        Execute one time step in the environment
        
        Args:
            action: 0 (DECREASE), 1 (DO_NOTHING), 2 (INCREASE)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {self.action_space}")
        
        # Convert Gym action to FanControlEnv action
        action_enum = list(Action)[action]
        
        # Step the physics simulation - returns 4 values: obs, reward, done, info
        obs, reward, terminated, info = self.fan_env.step(action_enum)
        
        # Gymnasium uses 5-tuple return (obs, reward, terminated, truncated, info)
        truncated = False  # We don't have truncation in this environment
        
        # Get updated info with current state
        current_info = self.get_current_state_info()
        info.update(current_info)
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment"""
        return self.fan_env.render(mode)
    
    def close(self):
        """Clean up resources"""
        pass
    
    def get_current_state_info(self):
        """Get detailed information about current state"""
        # Check if the fan_env has the required methods and attributes
        info = {
            "aqi": getattr(self.fan_env, 'current_aqi', 0),
            "fan_speed": getattr(self.fan_env, 'current_fan_speed', 0),
            "step": getattr(self.fan_env, 'current_step', 0),
        }
        
        # Add calculated values if methods exist
        if hasattr(self.fan_env, '_calculate_power'):
            info["power"] = self.fan_env._calculate_power()
        if hasattr(self.fan_env, '_get_actual_rpm'):
            info["rpm"] = self.fan_env._get_actual_rpm()
        if hasattr(self.fan_env, '_calculate_aer'):
            info["aer"] = self.fan_env._calculate_aer()
        if hasattr(self.fan_env, '_calculate_settling'):
            info["settling_rate"] = self.fan_env._calculate_settling()
        if hasattr(self.fan_env, 'current_voltage'):
            info["voltage"] = self.fan_env.current_voltage
        
        return info

    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self.fan_env, 'np_random'):
                self.fan_env.np_random.seed(seed)
        return [seed]
# Example usage and testing
if __name__ == "__main__":
    # Create the Gym environment
    env = FanControlGymEnv()
    
    # Test the Gym interface
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    print()
    
    # Test a few steps
    for step in range(10):
        # Try different actions
        if step < 3:
            action = 2  # INCREASE
        elif step < 6:
            action = 1  # DO_NOTHING
        else:
            action = 0  # DECREASE
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        action_names = ["DECREASE", "DO_NOTHING", "INCREASE"]
        print(f"Step {step + 1}:")
        print(f"  Action: {action} ({action_names[action]})")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward:.3f}")
        print(f"  AQI: {info['aqi']}, Fan Speed: {info['fan_speed']:.2f}")
        print(f"  Power: {info['power']:.1f}W, RPM: {int(info['rpm'])}")
        print()
        
        if terminated or truncated:
            print("Episode terminated!")
            break
    
    # Test with random actions
    print("Testing with random actions:")
    obs, info = env.reset()
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Random action {action}: AQI={info['aqi']}, "
              f"Fan={info['fan_speed']:.2f}, Reward={reward:.3f}")
        
        if terminated or truncated:
            break
    
    env.close()
        
        
        
