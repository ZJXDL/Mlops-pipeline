import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
    'engine_rpm': np.random.normal(3000, 500, n_samples),
    'engine_temp': np.random.normal(90, 10, n_samples),
    'vibration_level': np.random.normal(2.5, 0.8, n_samples),
}
df = pd.DataFrame(data)

# Introduce a rule for failure: high temp + high vibration = failure
df['failure'] = ((df['engine_temp'] > 105) & (df['vibration_level'] > 3.5)).astype(int)

df.to_csv('data/raw/telemetry.csv', index=False)
print("Dataset generated at data/raw/telemetry.csv")