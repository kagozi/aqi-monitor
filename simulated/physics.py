# File: simulated/physics.py
import numpy as np
from enum import Enum
import math

class Action(Enum):
    DECREASE = -1
    DO_NOTHING = 0
    INCREASE = 1

class FanControlEnv:
    def __init__(self, 
                 room_volume=50,         # m³
                 floor_area=25,          # m²
                 outdoor_aqi=2,          # baseline outdoor air quality (1-5)
                 init_aqi=5,             # initial indoor AQI (1-5)
                 init_fan_speed=0.5,      # initial fan speed (0-1)
                 target_aqi=1,           # desired AQI level (1-5)
                 max_steps=1000,
                 energy_weight=0.3,      # tradeoff between AQI and energy
                 particle_diameter=1e-6,  # m (1 micron)
                 particle_density=1000    # kg/m³ (water-like)
                 ):
        # Constants
        self.room_volume = room_volume
        self.floor_area = floor_area
        self.outdoor_aqi = outdoor_aqi
        self.target_aqi = target_aqi
        self.max_steps = max_steps
        self.energy_weight = energy_weight
        self.particle_diameter = particle_diameter
        self.particle_density = particle_density

        # Simulation parameters
        self.time_step = 5.0  # seconds (more realistic for AQI changes)
        self.g = 9.81  # m/s², gravitational acceleration
        self.mu = 1.8e-5  # Pa·s, dynamic viscosity of air at 20°C
        
        # Dynamic state
        self.current_aqi = init_aqi
        self.current_fan_speed = init_fan_speed
        self.current_step = 0
        
        # Motor/propeller constants (775 motor + 6x4 prop)
        self.voltage = 22.5  # 6S Li-ion
        self.max_rpm = 18750  # 20000 * (22.5/24)
        self.no_load_current = 0.5  # Amps (typical for 775 motor)
        self.stall_current = 7.0    # Amps
        self.motor_resistance = 2.5  # Ohms
        
        # Propeller performance (6x4)
        self.prop_diameter = 0.1524  # 6" in meters
        self.prop_pitch = 0.1016     # 4" in meters
        self.prop_efficiency = 0.65  # Typical for small propellers
        
        # Action space
        self.action_space = [a for a in Action]
        self.action_size = len(self.action_space)
        
        # Observation space: normalized AQI and fan speed
        self.observation_space = np.array([0.0, 0.0])
        
    def reset(self):
        """Reset the environment to initial state"""
        self.current_aqi = np.clip(np.random.normal(3, 1), 1, 5)
        self.current_fan_speed = np.clip(np.random.normal(0.5, 0.1), 0, 1)
        self.current_step = 0
        return self._get_obs()
    
    def _calculate_airflow(self, rpm):
        """Calculate airflow in m³/s for 6x4 propeller"""
        # Advance ratio calculation
        pitch_speed = rpm/60 * self.prop_pitch  # m/s
        theoretical_flow = pitch_speed * np.pi * (self.prop_diameter/2)**2
        
        # Apply efficiency and empirical correction
        return theoretical_flow * self.prop_efficiency * 0.85
    
    # def _get_obs(self):
    #     """Observations [AQI, fan speed]"""
    #     return np.array([self.current_aqi, self.current_fan_speed])
    def _get_obs(self):
        """Normalize observations to [0, 1] range"""
        norm_aqi = (self.current_aqi - 1) / 4       # AQI from 1–5 → 0–1
        return np.array([norm_aqi, self.current_fan_speed ], dtype=np.float32)

    
    def _calculate_aer(self):
        """Air Exchange Rate based on fan speed"""
        rpm = self.current_fan_speed * self.max_rpm
        Q = self._calculate_airflow(rpm)  # m³/s
        return Q / self.room_volume  # 1/s
    
    def _calculate_settling(self):
        """Particle settling rate using Stokes' law"""
        v_s = (self.g * (self.particle_diameter**2) * self.particle_density) / (18 * self.mu)
        return (v_s * self.floor_area) / self.room_volume  # 1/s
    
    def _calculate_power(self):
        """Calculate motor power draw in watts"""
        # Electrical power = V*I
        rpm = self.current_fan_speed * self.max_rpm
        load_factor = min(rpm/self.max_rpm, 0.95)  # Prevent division by zero
        
        # Current calculation (simplified motor model)
        current = self.no_load_current + (self.stall_current - self.no_load_current) * (1 - load_factor)
        return self.voltage * current
    
    def _update_aqi(self):
        """Update AQI based on ventilation and settling"""
        aer = self._calculate_aer()
        settling_rate = self._calculate_settling()
        total_removal = aer + settling_rate

        decay_factor = np.exp(-total_removal * self.time_step)
        noise = np.random.normal(0, 0.03)

        new_aqi = self.outdoor_aqi + (self.current_aqi - self.outdoor_aqi) * decay_factor + noise

        # ✅ Round to nearest discrete level from 1–5
        self.current_aqi = int(np.clip(round(new_aqi), 1, 5))
    
    def _calculate_reward(self):
        """Calculate reward balancing AQI and energy use"""
        aqi_error = abs(self.current_aqi - self.target_aqi)
        # aqi_reward = -aqi_error  # Penalize deviation from target
        
        # Energy cost using actual power calculation
        # energy_cost = self.energy_weight * (self._calculate_power() / 100)  # Scaled to similar magnitude
        
        # return aqi_reward - energy_cost
        reward = 1 / (1 + aqi_error) - self.energy_weight * (self._calculate_power() / 100)

        return reward

    
    def step(self, action):
        """
        Execute one time step in the environment
        Action: -1 (decrease), 0 (nothing), +1 (increase)
        """
        # Apply action with realistic response
        speed_change = 0.1 * action.value
        self.current_fan_speed = np.clip(self.current_fan_speed + speed_change, 0, 1)
        
        # Update physics
        self._update_aqi()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, {
            "aqi": self.current_aqi,
            "fan_speed": self.current_fan_speed,
            "power": self._calculate_power()
        }

    
    def render(self, mode='human'):
        """Visualize the current state"""
        rpm = self.current_fan_speed * self.max_rpm
        power = self._calculate_power()
        print(f"Step: {self.current_step:4d} | AQI: {self.current_aqi:.2f} | "
              f"Fan: {self.current_fan_speed:.2f} ({int(rpm)} RPM) | "
              f"Power: {power:.1f}W")

# # Example usage
# if __name__ == "__main__":
#     env = FanControlEnv()
#     state = env.reset()
    
#     # Test random policy
#     for _ in range(1000):
#         action = np.random.choice(env.action_space)
#         next_state, reward, done, _ = env.step(action)
#         env.render()
        
#         if done:
#             break