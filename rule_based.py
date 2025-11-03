import time
import json
import sqlite3
import paho.mqtt.client as mqtt

# MQTT setup
BROKER_IP = "10.0.0.189"
DATA_TOPIC = "sensor/data"
ACTION_TOPIC = "agent/action"

# SQLite database setup
conn = sqlite3.connect("fan_metrics_30.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS metrics (
    timestamp TEXT,
    aqi INTEGER,
    tvoc INTEGER,
    eco2 INTEGER,
    voltage REAL,
    fan_speed REAL,
    action TEXT,
    reward REAL,
    energy REAL,
    algorithm TEXT
)''')
conn.commit()

# Energy tracking
last_energy = 0.0
TIMESTEP_S = 1.0
aqi_target = 2.0  # Target AQI threshold for fan speed adjustment

# Timer setup (30 minutes = 1800 seconds)
START_TIME = time.time()
RUN_DURATION = 1800  # seconds

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker with result code", rc)
    client.subscribe(DATA_TOPIC)

def calculate_reward(current_aqi, target_aqi, current_fan_speed, energy_weight=0.3):
    """Calculate reward balancing AQI and energy use"""
    aqi_error = abs(current_aqi - target_aqi)
    aqi_reward = -aqi_error  # Penalize deviation from target

    # Energy cost using actual power calculation
    energy_cost = energy_weight * (calculate_power(current_fan_speed) / 100)  # Scaled to similar magnitude

    return aqi_reward - energy_cost

def calculate_power(current_fan_speed, min_speed=0.533, max_rpm=18750, no_load_current=0.5, stall_current=7.0, voltage=22.5):
    """Calculate motor power draw in watts"""
    # Electrical power = V*I
    if current_fan_speed < min_speed:
        return 0.0
    effective_speed = (current_fan_speed - min_speed) / (1 - min_speed)
    rpm = effective_speed * max_rpm
    # rpm = self.current_fan_speed * self.max_rpm
    load_factor = min(rpm / max_rpm, 0.95)  # Prevent division by zero
    # Current calculation (simplified motor model)
    current = no_load_current + (stall_current - no_load_current) * (1 - load_factor)
    return voltage * current

def on_message(client, userdata, msg):
    global last_energy
    try:
        payload = json.loads(msg.payload.decode())
        print("Received state:", payload)

        aqi = payload.get("aqi", 0)
        tvoc = payload.get("tvoc", 0)
        eco2 = payload.get("eco2", 0)
        voltage = payload.get("voltage", 0.0)
        fan_speed = payload.get("fan_speed", 0.0)

        ## Use a rule-based approach to determine fan speed
        if aqi > aqi_target and fan_speed < 0.9:
            action = 2  # Increase fan speed
        elif aqi <= 2 and fan_speed >= 0.53:
            action = 0
        else:
            action = 1
            
        reward = calculate_reward(aqi, aqi_target, fan_speed)

        # Map numeric action to string
        actions = ["decrease", "nothing", "increase"]
        action_str = actions[action]

        # Publish action
        action_payload = {"action": action_str}
        client.publish(ACTION_TOPIC, json.dumps(action_payload))
        print(f"Published action: {action_payload}")

        # Log metrics 
        cursor.execute("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            aqi,
            tvoc,
            eco2,
            voltage,
            fan_speed,
            action_str,
            reward,
            last_energy,
            "ruleBased"  # Algorithm name
        ))
        conn.commit()

        # Stop after 30 minutes
        if time.time() - START_TIME > RUN_DURATION:
            print("Test duration completed (0.5 hours). Exiting...")
            client.loop_stop()
            client.disconnect()
            conn.close()
            exit(0)

    except Exception as e:
        print("Error decoding or logging:", e)

# MQTT client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER_IP)

print("Starting MQTT listener for 30-minute test...")
client.loop_start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    client.loop_stop()
    client.disconnect()
    conn.close()
    print("Clean exit.")

