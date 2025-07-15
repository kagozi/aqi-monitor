# pi_mqtt_agent.py
import serial
import time
import json
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO

# MQTT setup
BROKER_IP = "10.0.0.194"  # Your Mac's IP
DATA_TOPIC = "sensor/data"
ACTION_TOPIC = "agent/action"

# Fan control GPIO
IN1, IN2, ENA = 17, 27, 22
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
pwm = GPIO.PWM(ENA, 1000)
# pwm.start(0)
fan_speed = 0.0

def set_fan_speed(speed):
    global fan_speed
    fan_speed = max(0.0, min(speed, 1.0))
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(fan_speed * 100)

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
                    "fan_speed": round(fan_speed, 2)
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
    GPIO.cleanup()
    client.loop_stop()
    client.disconnect()
