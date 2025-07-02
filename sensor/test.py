'''
Test with trained model
This script will load the trained model and run it in the environment to see how it performs.
'''

import os
from stable_baselines3 import PPO
from ppo_environment import MQTTFanEnv  # Your custom env
import time

# Get the directory where the current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to the script location
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'ppo', 'train')

# 1. Load Environment and Model
env = MQTTFanEnv()  # Use same IP as training
model = PPO.load(f"{CHECKPOINT_DIR}/best_model_17000")  # Path to your saved model

# 2. Test Loop
# obs, _ = env.reset()
# for _ in range(1000):  # Run for 1000 steps
#     action, _ = model.predict(obs, deterministic=True)  # Use trained policy
#     obs, reward, terminated, truncated, info = env.step(action)
    
#     print(f"Step {info['episode']['l']}:")
#     print(f"  AQI: {info['aqi']:.1f} | Fan: {info['fan_speed']:.2f}")
#     print(f"  Target Fan: {info.get('target_fan_speed', 0):.2f} | Reward: {reward:.2f}")
    
#     if terminated or truncated:
#         obs, _ = env.reset()
    
#     time.sleep(1)  # Match your sensor's publish rate

# env.close()

import matplotlib.pyplot as plt
import numpy as np

aqi_range = np.linspace(1.0, 5.0, 20)
speed_range = np.linspace(0.0, 1.0, 20)
action_grid = np.zeros((20, 20))

for i, aqi in enumerate(aqi_range):
    for j, speed in enumerate(speed_range):
        obs = np.array([aqi, speed], dtype=np.float32)
        action, _ = model.predict(obs)
        action_grid[i, j] = action

plt.figure(figsize=(10,6))
plt.imshow(action_grid, extent=[0,1,1,5], aspect='auto', cmap='coolwarm')
plt.colorbar(ticks=[0,1,2], label='0=Decrease, 1=Hold, 2=Increase')
plt.xlabel('Current Fan Speed')
plt.ylabel('AQI')
plt.title('Model Policy Visualization')
plt.show()