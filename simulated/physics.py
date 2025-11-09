from __future__ import annotations
import numpy as np
from enum import IntEnum
from typing import Tuple, Dict, Any


class Action(IntEnum):
    DECREASE   = 0
    DO_NOTHING = 1
    INCREASE   = 2


class FanControlEnv:
    def __init__(
        self,
        room_volume: float = 50.0,
        floor_area: float = 25.0,
        outdoor_conc: float = 80.0,
        init_conc: float = 300.0,
        max_steps: int = 1000,
        energy_weight: float = 2.0,
        particle_diameter: float = 2.5e-6,
        particle_density: float = 1400.0,
        prop_diameter: float = 0.1524,
        prop_pitch: float = 0.1016,
        prop_efficiency: float = 0.65,
        voltage: float = 24.0,
        min_voltage: float = 12.0,
        max_rpm: float = 20000.0,
        curriculum_stage: int = -1,
    ) -> None:
        self.room_volume = room_volume
        self.floor_area = floor_area
        self.outdoor_conc = outdoor_conc
        self.max_steps = max_steps
        self.energy_weight = energy_weight
        self.time_step = 5.0
        self.curriculum_stage = curriculum_stage

        self.particle_diameter = particle_diameter
        self.particle_density = particle_density

        self.prop_diameter = prop_diameter
        self.prop_pitch = prop_pitch
        self.prop_efficiency = prop_efficiency
        self.voltage = voltage
        self.min_voltage = min_voltage
        self.min_speed = self.min_voltage / self.voltage
        self.max_rpm = max_rpm

        self.current_conc: float = init_conc
        self.current_fan_speed: float = 0.5
        self.current_step: int = 0
        self._prev_aqi: int = 1
        self._prev_conc: float = init_conc

        self.action_space = list(Action)
        self.observation_space = np.array([0.0, 0.0], dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset with optional curriculum learning"""
        if self.curriculum_stage == 0:
            # Stage 1: Start at moderate AQI (easy to improve)
            self.current_conc = float(np.clip(np.random.normal(150, 30), 100, 200))
            self.current_fan_speed = 0.5
        elif self.curriculum_stage == 1:
            # Stage 2: Start at high AQI (learn aggressive cleaning)
            self.current_conc = float(np.clip(np.random.normal(400, 50), 300, 600))
            self.current_fan_speed = float(np.random.uniform(0.5, 0.8))
        else:
            # Stage 3 or default: Random (full task)
            self.current_conc = float(np.clip(np.random.normal(300, 80), 100, 600))
            self.current_fan_speed = float(np.clip(np.random.normal(0.5, 0.1), 0.0, 1.0))
        
        self.current_step = 0
        self._prev_aqi = self.pm25_to_aqi(self.current_conc)
        self._prev_conc = self.current_conc
        return self._get_obs()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        speed_change = 0.1 * (action - 1)
        new_speed = self.current_fan_speed + speed_change
        if 0.0 < new_speed < self.min_speed:
            new_speed = 0.0
        else:
            new_speed = np.clip(new_speed, self.min_speed, 1.0)
        self.current_fan_speed = float(new_speed)

        self._update_concentration()
        reward = self._calculate_reward()

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        aqi = self.pm25_to_aqi(self.current_conc)
        rpm = self._get_actual_rpm()
        airflow = self._calculate_airflow(rpm)
        info = {
            "aqi": int(aqi),
            "concentration": round(self.current_conc, 2),
            "fan_speed": self.current_fan_speed,
            "power": round(self._calculate_power(), 2),
            "rpm": int(rpm),
            "airflow_m3h": round(airflow * 3600, 1),
            "ach": round(airflow * 3600 / self.room_volume, 3),
        }
        return self._get_obs(), reward, terminated, info

    @staticmethod
    def pm25_to_aqi(pm25: float) -> int:
        if pm25 <= 12.0:   return 1
        if pm25 <= 35.4:   return 2
        if pm25 <= 55.4:   return 3
        if pm25 <= 150.4:  return 4
        return 5

    def _get_obs(self) -> np.ndarray:
        aqi = self.pm25_to_aqi(self.current_conc)
        return np.array([(aqi - 1)/4.0, self.current_fan_speed], dtype=np.float32)

    # --- AIRFLOW: FULL IMPACT ---
    def _calculate_airflow(self, rpm: float) -> float:
        if rpm <= 0: return 0.0
        pitch_speed = (rpm / 60.0) * self.prop_pitch
        area = np.pi * (self.prop_diameter / 2.0) ** 2
        theoretical = pitch_speed * area
        return theoretical * self.prop_efficiency * 1.0

    def _calculate_settling_rate(self) -> float:
        rho_p, rho_a = self.particle_density, 1.225
        d, mu, g = self.particle_diameter, 1.81e-5, 9.81
        v_s = (rho_p - rho_a) * g * d**2 / (18.0 * mu)
        return (v_s * self.floor_area) / self.room_volume

    def _update_concentration(self) -> None:
        rpm = self._get_actual_rpm()
        Q = self._calculate_airflow(rpm)
        aer = Q / self.room_volume
        settling = self._calculate_settling_rate()
        total_rate = aer + settling
        decay = np.exp(-total_rate * self.time_step)
        new_conc = self.outdoor_conc * (1.0 - decay) + self.current_conc * decay
        noise = np.random.normal(0.0, 0.5)
        self.current_conc = float(np.clip(new_conc + noise, 0.0, 1000.0))

    # --- REWARD: PROXIMITY TO TARGET + CONTINUOUS IMPROVEMENT ---
    def _calculate_reward(self) -> float:
        """
        Reward = target_achievement - energy_penalty
        
        Key fixes:
        1. Reward PROXIMITY to target AQI (not just penalty)
        2. Scale energy cost to match AQI scale
        3. Add continuous improvement signal
        """
        cur_aqi = self.pm25_to_aqi(self.current_conc)
        
        # Target: AQI 1-2 (good air quality)
        target_aqi = 1.5
        distance_to_target = abs(cur_aqi - target_aqi)
        
        # Proximity reward: +10 at target, 0 at AQI=5
        proximity_reward = 10.0 * (1.0 - distance_to_target / 3.5)
        
        # Continuous improvement (not just category changes)
        conc_improvement = self._prev_conc - self.current_conc
        improvement_reward = 0.05 * conc_improvement  # Small bonus for any reduction
        
        # Energy penalty: scaled to match AQI scale
        power_w = self._calculate_power()
        energy_penalty = self.energy_weight * (power_w / 80.0)
        
        # Store for next step
        self._prev_conc = self.current_conc
        
        return proximity_reward + improvement_reward - energy_penalty

    def _get_actual_rpm(self) -> float:
        if self.current_fan_speed < self.min_speed:
            return 0.0
        effective = (self.current_fan_speed - self.min_speed) / (1.0 - self.min_speed)
        return effective * self.max_rpm

    def _calculate_power(self) -> float:
        if self.current_fan_speed < self.min_speed:
            return 0.0
        speed_norm = (self.current_fan_speed - self.min_speed) / (1.0 - self.min_speed)
        P_no_load = 6.0
        P_full_load = 80.0 / 0.70
        return P_no_load + (P_full_load - P_no_load) * (speed_norm ** 2)