import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('data/run_20260310_152301.csv')

# Identify step boundaries
df['step_count'] = df['step_count'].astype(int)
step_changes = df['step_count'].diff().fillna(0)
step_indices = df.index[df['step_count'] > 0]

# Plot joint angles for 3 steps
plt.figure(figsize=(12, 8))

# Pelvic tilt: base_roll_rad
plt.subplot(3, 1, 1)
plt.plot(df['sim_time_s'], df['base_roll_rad'], label='Pelvic Tilt (roll)', color='purple')
plt.ylabel('Pelvic Tilt (rad)')
plt.legend()

# Hip flex/extension: hip_right_rad, hip_left_rad
plt.subplot(3, 1, 2)
plt.plot(df['sim_time_s'], df['hip_right_rad'], label='Hip Right', color='blue')
plt.plot(df['sim_time_s'], df['hip_left_rad'], label='Hip Left', color='cyan', linestyle='--')
plt.ylabel('Hip Angle (rad)')
plt.legend()

# Knee flex/extension: knee_right_rad, knee_left_rad
plt.subplot(3, 1, 3)
plt.plot(df['sim_time_s'], df['knee_right_rad'], label='Knee Right', color='green')
plt.plot(df['sim_time_s'], df['knee_left_rad'], label='Knee Left', color='lime', linestyle='--')
plt.ylabel('Knee Angle (rad)')
plt.xlabel('Time (s)')
plt.legend()

plt.tight_layout()
plt.savefig('data/joint_angles_3steps.png', dpi=300)
plt.show()
