import pandas as pd
import numpy as np

np.random.seed(42)
samples = 1000

# Simulate sensor inputs
ldr = np.random.randint(0, 1024, samples)          # Ambient light
pir = np.random.choice([0, 1], samples)            # Motion
temp = np.random.uniform(20, 35, samples)          # Temperature
hour = np.random.randint(0, 24, samples)           # Time of day

# Rule-based light strength
def simulate_light_strength(ldr, pir, temp, hour):
    if pir == 0:
        return 0
    base = max(0, 255 - ldr // 4)  # darker = brighter
    temp_factor = 1 if temp < 30 else 0.8
    time_factor = 1 if 18 <= hour <= 23 else 0.6
    return int(base * temp_factor * time_factor)

light_strength = [simulate_light_strength(ldr[i], pir[i], temp[i], hour[i]) for i in range(samples)]

df = pd.DataFrame({
    'ambient_light': ldr,
    'motion_detected': pir,
    'temperature': temp,
    'hour': hour,
    'light_strength': light_strength
})

df.to_csv('../data/smartlight_data.csv', index=False)
print("Dataset generated and saved to data/smartlight_data.csv")
