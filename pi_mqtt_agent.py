# main.py
import serial
import time
import json
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO
import numpy as np

# MQTT setup
BROKER_IP = "10.0.0.194"  # Your Mac's IP
DATA_TOPIC = "sensor/data"
ACTION_TOPIC = "agent/action"

# BTS7960 GPIO Configuration (Right Channel Only)
RPWM = 22    # GPIO22 (Pin 15) - PWM for forward
R_EN = 24    # GPIO24 (Pin 18) - Enable forward channel
LPWM = 23    # GPIO23 (Pin 16) - Disabled
L_EN = 25    # GPIO25 (Pin 22) - Disabled

# Motor parameters
MIN_VOLTAGE = 12.0  # Minimum operating voltage
MAX_VOLTAGE = 22.5   # 6S LiPo
MIN_DUTY_CYCLE = (MIN_VOLTAGE / MAX_VOLTAGE) * 100  # ≈53%

GPIO.setmode(GPIO.BCM)
GPIO.setup(RPWM, GPIO.OUT)
GPIO.setup(R_EN, GPIO.OUT)
GPIO.setup(LPWM, GPIO.OUT, initial=GPIO.LOW)  # Disable reverse
GPIO.setup(L_EN, GPIO.OUT, initial=GPIO.LOW)  # Disable reverse channel

pwm = GPIO.PWM(RPWM, 20000)  # 20kHz PWM frequency
GPIO.output(R_EN, GPIO.HIGH)  # Enable forward channel
fan_speed = 0.0  # Normalized speed (0.0-1.0)

def set_fan_speed(speed):
    global fan_speed
    speed = np.clip(speed, 0.0, 1.0)
    
    # Handle 12V minimum voltage threshold
    if speed < (MIN_VOLTAGE / MAX_VOLTAGE):
        if fan_speed >= (MIN_VOLTAGE / MAX_VOLTAGE):
            # Gradual ramp-down when crossing threshold
            speed = max(0, (MIN_VOLTAGE / MAX_VOLTAGE) - 0.05)
        else:
            speed = 0.0
    
    # Scale duty cycle for 53%-100% range
    if speed > 0:
        duty_cycle = MIN_DUTY_CYCLE + (speed - (MIN_VOLTAGE / MAX_VOLTAGE)) * (100 - MIN_DUTY_CYCLE) / (1 - (MIN_VOLTAGE / MAX_VOLTAGE))
        pwm.start(duty_cycle)
    else:
        pwm.stop()
    
    fan_speed = speed
    print(f"Set speed: {speed:.2f} → Duty cycle: {duty_cycle if speed > 0 else 0}%")

# Serial connection to Pico
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)

# MQTT callback
def on_message(client, userdata, msg):
    global fan_speed
    try:
        payload = json.loads(msg.payload.decode())
        action = payload.get("action")
        if action == "increase":
            set_fan_speed(fan_speed + 0.1)
        elif action == "decrease":
            set_fan_speed(fan_speed - 0.1)
        elif action == "nothing":
            pass  # do nothing
        print(f"Action received: {action} → Fan speed now: {fan_speed:.2f}")
    except Exception as e:
        print("Action decode error:", e)

# MQTT setup
client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER_IP)
client.subscribe(ACTION_TOPIC)
client.loop_start()

# Main loop
try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if "AQI:" in line:
            try:
                aqi = int(line.split("AQI:")[1].split()[0])
                state = {
                    "aqi": aqi,
                    "fan_speed": round(fan_speed, 2),
                    "voltage": MAX_VOLTAGE * fan_speed if fan_speed >= (MIN_VOLTAGE / MAX_VOLTAGE) else 0.0
                }
                client.publish(DATA_TOPIC, json.dumps(state))
                print("Published state:", state)
            except Exception as e:
                print("State parse error:", e)
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    pwm.stop()
    GPIO.output(R_EN, GPIO.LOW)  # Disable driver
    GPIO.cleanup()
    client.loop_stop()
    client.disconnect()