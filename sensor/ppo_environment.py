# mqtt_env_gymnasium.py
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import queue
import time
import json
from stable_baselines3.common.callbacks import BaseCallback
from scipy.special import expit


class MQTTFanEnv(gym.Env):
    def __init__(self, broker_ip="10.0.0.194"):
        super().__init__()

        self.broker_ip = broker_ip
        self.time_step = 0

        self.observation_space = gym.spaces.Box(low=np.array([1.0, 0.0]), high=np.array([5.0, 1.0]), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # 0=decrease, 1=nothing, 2=increase

        self.state_queue = queue.Queue()
        self.last_state = np.array([3.0, 0.5], dtype=np.float32)  # Fallback state

        # Setup MQTT
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker_ip)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected to MQTT broker with code", rc)
        client.subscribe("sensor/data")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            aqi = float(payload.get("aqi", 3))
            fan_speed = float(payload.get("fan_speed", 0.5))
            state = np.array([aqi, fan_speed], dtype=np.float32)
            print(f"Received state: AQI={aqi}, Fan Speed={fan_speed}")
            self.last_state = state
            self.state_queue.put(state)
        except Exception as e:
            print("MQTT message error:", e)

    def send_action(self, action_idx):
        action_map = ["decrease", "nothing", "increase"]
        action_payload = {"action": action_map[action_idx]}
        self.client.publish("agent/action", json.dumps(action_payload))
        print(f"Sent action: {action_map[action_idx]}")

    def wait_for_state(self, timeout=5.0):
        """Returns None if no new data arrives, rather than falling back to last state"""
        try:
            return self.state_queue.get(timeout=timeout)
        except queue.Empty:
            print("Sensor timeout - no new data received")
            return None
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        print("Waiting for initial state...")
        obs = self.wait_for_state()
        return obs, {}

 
    def step(self, action):
        # First check for fresh sensor data before taking any action
        obs = self.wait_for_state(timeout=2.0)
        if obs is None:
            # No new data - return zero reward and don't increment timestep
            return self.last_state, 0.0, False, False, {
                'aqi': self.last_state[0],
                'fan_speed': self.last_state[1],
                'episode': {'r': 0.0, 'l': self.time_step},
                'warning': 'no_new_data'
            }
        
        # Only proceed if we have fresh data
        self.send_action(action)
        time.sleep(0.5)  # Allow hardware to respond
        
        # Get post-action state
        new_obs = self.wait_for_state(timeout=2.0)
        aqi, fan_speed = new_obs if new_obs is not None else obs  # Fallback to pre-action state
        
        # Calculate target fan speed (linear mapping AQI 1->0.0, AQI 5->1.0)
        target_fan_speed = np.clip((aqi - 1) / 4, 0.0, 1.0)
        
        # Reward components
        speed_error = (fan_speed - target_fan_speed) ** 2  # Quadratic penalty for deviation
        reward = (
            -aqi                     # Primary penalty for high AQI
            - 2 * speed_error        # Strong penalty for wrong fan speed
            + 0.2 * (1 - fan_speed)  # Small energy efficiency bonus
        )
        
        self.time_step += 1
        return (new_obs if new_obs is not None else obs), reward, False, False, {
            'aqi': aqi,
            'fan_speed': fan_speed,
            'target_fan_speed': target_fan_speed,
            'speed_error': speed_error,
            'episode': {'r': reward, 'l': self.time_step}
        }
    
    def render(self, mode='human'):
        print(f"Current state: AQI={self.last_state[0]}, Fan Speed={self.last_state[1]}")

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


class EnhancedLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(EnhancedLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.metrics = {
            'aqi': [],
            'fan_speed': []
        }
        self.df = pd.DataFrame(columns=['timestep', 'aqi', 'fan_speed'])

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Collect metrics every step
        for info in self.locals.get('infos', []):
            if isinstance(info, dict):  # Ensure it's a dictionary
                self.metrics['aqi'].append(info.get('aqi', 0))
                self.metrics['fan_speed'].append(info.get('fan_speed', 0))

        # Log to CSV more frequently without resetting metrics
        if self.n_calls % 10 == 0:  # Every 10 steps now
            new_row = {
                'timestep': self.n_calls,
                'aqi': self.metrics['aqi'][-1] if self.metrics['aqi'] else 0,
                'fan_speed': self.metrics['fan_speed'][-1] if self.metrics['fan_speed'] else 0,
            }
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            self.df.to_csv(os.path.join(self.save_path, 'training_metrics.csv'), index=False)

        # Less frequent model checkpointing and tensorboard logging
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
            
            # Only log averages to tensorboard, don't reset metrics
            if self.metrics['aqi']:  # Only if we have data
                self.logger.record('env/aqi', np.mean(self.metrics['aqi']))
                self.logger.record('env/fan_speed', np.mean(self.metrics['fan_speed']))
            
        return True