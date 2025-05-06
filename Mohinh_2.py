
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

df = pd.read_csv("podcast_dataset.csv")
df_wed = df[df['Publication_Day'] == 'Wednesday'].copy()
df_wed = df_wed[['Listening_Time_minutes']].dropna().reset_index(drop=True)
observations = df_wed['Listening_Time_minutes'].values

kf = KalmanFilter(
    transition_matrices=[[1, 1], [0, 1]],
    observation_matrices=[[1, 0]],
    initial_state_mean=[observations[0], 0],
    initial_state_covariance=np.eye(2),
    observation_covariance=1,
    transition_covariance=np.eye(2)*0.01
)
state_means, _ = kf.filter(observations)

plt.figure(figsize=(12, 5))
plt.plot(observations, 'k.', label='Observed')
plt.plot(state_means[:, 0], 'r-', label='Kalman Filter')
plt.title("Kalman Filter với xu hướng tuyến tính")
plt.xlabel("Time Index")
plt.ylabel("Listening Time (minutes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
