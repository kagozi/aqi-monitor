import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from enum import Enum

class Action(Enum):
    DECREASE = -1
    DO_NOTHING = 0
    INCREASE = 1

class FanControlEnvironment(gym.Env):
    """
    RL Environment for fan control based on AQI using mass balance equations
    with realistic hardware constraints (775 motor + 6x4 propeller)
    """
    
    def __init__(self, 
                 room_volume=50.0,
                 floor_area=20.0,
                 outdoor_aqi=50.0,
                 target_aqi=50.0,
                 init_aqi=80.0,
                 max_steps=1000,
                 energy_weight=0.5,
                 particle_diameter=2.5e-6,
                 particle_density=1200.0):
        
        super(FanControlEnvironment, self).__init__()
        
        # Physical parameters
        self.room_volume = room_volume
        self.floor_area = floor_area
        self.outdoor_aqi = outdoor_aqi
        self.target_aqi = target_aqi
        self.max_steps = max_steps
        self.energy_weight = energy_weight
        self.particle_diameter = particle_diameter
        self.particle_density = particle_density

        # Simulation parameters
        self.time_step = 5.0  # seconds
        self.g = 9.81  # m/s²
        self.mu = 1.8e-5  # Pa·s, dynamic viscosity of air at 20°C
        
        # Motor/propeller constants (775 motor + 6x4 prop)
        self.voltage = 22.5  # 6S Li-ion
        self.max_rpm = 18750  # 20000 * (22.5/24)
        self.no_load_current = 0.5  # Amps
        self.stall_current = 7.0    # Amps
        self.motor_resistance = 2.5  # Ohms
        
        # Propeller performance (6x4)
        self.prop_diameter = 0.1524  # 6" in meters
        self.prop_pitch = 0.1016     # 4" in meters
        self.prop_efficiency = 0.65  # Typical for small propellers
        
        # Action space: Discrete actions (DECREASE, DO_NOTHING, INCREASE)
        self.action_space = spaces.Discrete(3)  # 0=DECREASE, 1=DO_NOTHING, 2=INCREASE
        self.duty_cycle_step = 0.1  # Step size for duty cycle changes
        
        # State space: [current_AQI, outdoor_AQI, fan_duty_cycle, dAQI_dt]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -100.0]),
            high=np.array([500.0, 500.0, 1.0, 100.0]),
            dtype=np.float32
        )
        
        # Dynamic state
        self.current_aqi = init_aqi
        self.current_duty_cycle = 0.0
        self.previous_duty_cycle = 0.0
        self.daqi_dt = 0.0
        self.current_step = 0
        
        # Initialize in reset
        self.reset()

    def calculate_flow_rate(self, duty_cycle):
        """Calculate volumetric flow rate based on duty cycle and motor/propeller specs"""
        if duty_cycle <= 0.0:
            return 0.0
        
        rpm = duty_cycle * self.max_rpm
        rps = rpm / 60.0
        pitch_speed = rps * self.prop_pitch
        theoretical_flow = pitch_speed * np.pi * (self.prop_diameter / 2) ** 2
        flow_rate_m3_per_s = theoretical_flow * self.prop_efficiency * 0.85
        flow_rate = flow_rate_m3_per_s * 60.0  # m³/min
        return flow_rate
    
    def calculate_power_consumption(self, duty_cycle):
        """Calculate power consumption based on motor characteristics"""
        if duty_cycle <= 0.0:
            return 0.0
        
        current = self.no_load_current + (self.stall_current - self.no_load_current) * duty_cycle * 0.7
        power = self.voltage * current
        return power  # Watts
    
    def calculate_settling_velocity(self):
        """Calculate particle settling velocity using Stokes' Law"""
        rho_air = 1.225
        settling_velocity = (
            (self.particle_density - rho_air) * self.g * self.particle_diameter ** 2
        ) / (18 * self.mu)
        return settling_velocity  # m/s
    
    def mass_balance_dynamics(self, current_aqi, outdoor_aqi, duty_cycle, source_rate):
        """Calculate AQI change based on mass balance"""
        flow_rate = self.calculate_flow_rate(duty_cycle)
        flow_rate_per_sec = flow_rate / 60.0
        ventilation_rate = flow_rate_per_sec / self.room_volume
        
        settling_velocity = self.calculate_settling_velocity()
        settling_rate = (settling_velocity * self.floor_area) / self.room_volume
        
        ventilation_term = ventilation_rate * (outdoor_aqi - current_aqi)
        source_term = source_rate / self.room_volume
        settling_term = settling_rate * current_aqi
        
        aqi_change_per_sec = ventilation_term + source_term - settling_term
        return aqi_change_per_sec
    
    def step(self, action):
        # action is scalar integer: 0, 1, or 2
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Map discrete action to duty cycle change
        if action == 0:  # DECREASE
            duty_change = -self.duty_cycle_step
        elif action == 2:  # INCREASE
            duty_change = self.duty_cycle_step
        else:  # DO_NOTHING
            duty_change = 0.0

        # Update duty cycle
        self.previous_duty_cycle = self.current_duty_cycle
        self.current_duty_cycle = np.clip(
            self.current_duty_cycle + duty_change, 0.0, 1.0
        )
        duty_cycle = self.current_duty_cycle

        # Simulate pollutant source
        base_source = 2.0 + 3.0 * np.sin(self.current_step * 0.1)
        noise_source = np.random.normal(0, 1.0)
        source_rate = max(0, base_source + noise_source)

        # Update AQI
        daqi_dt = self.mass_balance_dynamics(
            self.current_aqi, self.outdoor_aqi, duty_cycle, source_rate
        )
        self.current_aqi += daqi_dt * self.time_step
        self.current_aqi = np.clip(self.current_aqi, 0.0, 500.0)

        # Power and reward
        power = self.calculate_power_consumption(duty_cycle)
        reward = self._calculate_reward(duty_cycle, daqi_dt, power)

        # Update state
        self.daqi_dt = daqi_dt
        self.current_step += 1

        # Termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Observation
        obs = self._get_observation()

        return obs, reward, terminated, truncated, {
            'power': power,
            'aqi': self.current_aqi,
            'duty_cycle': duty_cycle
        }
    
    def _calculate_reward(self, duty_cycle, daqi_dt, power):
        """Calculate reward"""
        aqi_error = abs(self.current_aqi - self.target_aqi)
        air_quality_reward = -aqi_error / 100.0
        if aqi_error < 10.0:
            air_quality_reward += 1.0
        
        energy_penalty = -(power / 160.0) * self.energy_weight
        comfort_penalty = -abs(duty_cycle - self.previous_duty_cycle) * 0.3
        
        improvement_bonus = 0.0
        if self.current_aqi > self.target_aqi and daqi_dt < 0 and self.outdoor_aqi < self.current_aqi:
            improvement_bonus = 0.5
        elif self.current_aqi <= self.target_aqi and abs(daqi_dt) < 1.0:
            improvement_bonus = 0.3
        
        if (self.outdoor_aqi - self.current_aqi) > 20.0 and duty_cycle > 0.3:
            improvement_bonus -= 0.5

        return air_quality_reward + energy_penalty + comfort_penalty + improvement_bonus
    
    def _get_observation(self):
        """Return current observation"""
        return np.array([
            self.current_aqi,
            self.outdoor_aqi,
            self.current_duty_cycle,  # ← Correct state variable
            self.daqi_dt
        ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_aqi = np.random.uniform(40.0, 120.0)
        self.outdoor_aqi = np.random.uniform(20.0, 150.0)
        self.current_duty_cycle = 0.0
        self.previous_duty_cycle = 0.0
        self.daqi_dt = 0.0
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def render(self):
        """Print current state"""
        rpm = self.current_duty_cycle * self.max_rpm
        power = self.calculate_power_consumption(self.current_duty_cycle)
        print(f"Step: {self.current_step:4d} | AQI: {self.current_aqi:.2f} | "
              f"Fan: {self.current_duty_cycle:.2f} ({int(rpm)} RPM) | "
              f"Power: {power:.1f}W")


# Training function
def train_fan_controller(total_timesteps=100000, use_dqn=True):
    env = FanControlEnvironment(
        room_volume=50.0,
        floor_area=20.0,
        outdoor_aqi=50.0,
        target_aqi=50.0,
        max_steps=1000,
        energy_weight=0.5
    )
    check_env(env, warn=True)
    
    if use_dqn:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log="./dqn_fan_tensorboard/"
        )
        model_name = "fan_controller_dqn_hardware"
    else:
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./ppo_fan_tensorboard/"
        )
        model_name = "fan_controller_ppo_hardware"
    
    algo = "DQN" if use_dqn else "PPO"
    print(f"Training fan controller with {algo} and hardware constraints...")
    print(f"Motor: 775 @ {env.voltage}V | Prop: 6x4 | Max RPM: {env.max_rpm}")
    print(f"Action: Discrete(3) → Duty Cycle ±0.1")
    print("-" * 60)
    
    model.learn(total_timesteps=total_timesteps)
    model.save(model_name)
    print(f"Model saved as '{model_name}.zip'")
    
    return model, env


# Testing and visualization
def test_controller(model, env, episodes=5, test_steps=200):
    results = {
        'aqi': [], 'outdoor_aqi': [], 'duty_cycle': [],
        'rewards': [], 'power': [], 'rpm': [], 'flow_rate': []
    }
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(test_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            results['aqi'].append(obs[0])
            results['outdoor_aqi'].append(obs[1])
            results['duty_cycle'].append(obs[2])
            results['rewards'].append(reward)
            results['power'].append(info['power'])
            results['rpm'].append(obs[2] * env.max_rpm)
            results['flow_rate'].append(env.calculate_flow_rate(obs[2]))
            
            episode_reward += reward
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} | Reward: {episode_reward:.2f} | "
              f"Final AQI: {obs[0]:.1f} | Duty: {obs[2]:.2f}")

    # Plotting
    time_min = np.arange(len(results['aqi'])) * env.time_step / 60.0
    fig = plt.figure(figsize=(16, 12))

    plt.subplot(3, 2, 1)
    plt.plot(time_min, results['aqi'], label='Indoor AQI', color='red', linewidth=2)
    plt.plot(time_min, results['outdoor_aqi'], label='Outdoor AQI', color='orange', alpha=0.7)
    plt.axhline(y=env.target_aqi, color='green', linestyle='--', label='Target AQI')
    plt.ylabel('AQI'); plt.xlabel('Time (min)'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title('Air Quality Control')

    plt.subplot(3, 2, 2)
    plt.plot(time_min, results['duty_cycle'], color='blue', linewidth=2)
    plt.ylabel('Duty Cycle'); plt.xlabel('Time (min)'); plt.grid(True, alpha=0.3)
    plt.title('Fan Control Signal')

    plt.subplot(3, 2, 3)
    plt.plot(time_min, results['rewards'], color='purple', linewidth=1.5)
    plt.ylabel('Reward'); plt.xlabel('Time (min)'); plt.grid(True, alpha=0.3)
    plt.title('Instantaneous Reward')

    plt.subplot(3, 2, 4)
    plt.plot(time_min, results['power'], color='brown', linewidth=2)
    plt.ylabel('Power (W)'); plt.xlabel('Time (min)'); plt.grid(True, alpha=0.3)
    plt.title('Power Consumption (775 Motor)')

    plt.subplot(3, 2, 5)
    plt.plot(time_min, results['rpm'], color='darkgreen', linewidth=2)
    plt.ylabel('RPM'); plt.xlabel('Time (min)'); plt.grid(True, alpha=0.3)
    plt.title(f'Motor Speed (Max: {env.max_rpm} RPM)')

    plt.subplot(3, 2, 6)
    plt.plot(time_min, results['flow_rate'], color='teal', linewidth=2)
    plt.ylabel('Flow Rate (m³/min)'); plt.xlabel('Time (min)'); plt.grid(True, alpha=0.3)
    plt.title('Airflow (6x4 Propeller)')

    plt.tight_layout()
    plt.show()

    # Summary
    total_energy_wh = np.sum(results['power']) * env.time_step / 3600
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY ({episodes} episodes)")
    print(f"{'='*60}")
    print(f"Avg AQI: {np.mean(results['aqi']):.2f} ± {np.std(results['aqi']):.2f}")
    print(f"Avg Duty: {np.mean(results['duty_cycle']):.3f} | Step: 0.1")
    print(f"Avg Power: {np.mean(results['power']):.2f}W | Peak: {np.max(results['power']):.1f}W")
    print(f"Energy Used: {total_energy_wh:.2f} Wh")
    print(f"Avg RPM: {np.mean(results['rpm']):.0f}")
    print(f"{'='*60}")

    return results


# Example usage
if __name__ == "__main__":
    # Train with DQN (recommended for Discrete actions)
    model, env = train_fan_controller(total_timesteps=100000, use_dqn=True)
    
    # Test
    print("\nTesting trained controller...")
    results = test_controller(model, env, episodes=3, test_steps=200)