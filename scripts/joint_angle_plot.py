import csv
import matplotlib.pyplot as plt

# Read CSV manually
csv_path = 'data/run_20260310_152301.csv'
rows = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Extract columns
sim_time = [float(r['sim_time_s']) for r in rows]
pelvic_tilt = [float(r['base_roll_rad']) for r in rows]
hip_right = [float(r['hip_right_rad']) for r in rows]
hip_left = [float(r['hip_left_rad']) for r in rows]
knee_right = [float(r['knee_right_rad']) for r in rows]
knee_left = [float(r['knee_left_rad']) for r in rows]
step_count = [int(r['step_count']) for r in rows]

# Find indices for 3 steps
step_indices = [i for i, s in enumerate(step_count) if s > 0]
if len(step_indices) >= 3:
    start = step_indices[0]
    end = step_indices[2] + 1
else:
    start = 0
    end = len(sim_time)

# Slice for 3 steps
sim_time = sim_time[start:end]
pelvic_tilt = pelvic_tilt[start:end]
hip_right = hip_right[start:end]
hip_left = hip_left[start:end]
knee_right = knee_right[start:end]
knee_left = knee_left[start:end]

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(sim_time, pelvic_tilt, label='Pelvic Tilt (roll)', color='purple')
plt.ylabel('Pelvic Tilt (rad)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sim_time, hip_right, label='Hip Right', color='blue')
plt.plot(sim_time, hip_left, label='Hip Left', color='cyan', linestyle='--')
plt.ylabel('Hip Angle (rad)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sim_time, knee_right, label='Knee Right', color='green')
plt.plot(sim_time, knee_left, label='Knee Left', color='lime', linestyle='--')
plt.ylabel('Knee Angle (rad)')
plt.xlabel('Time (s)')
plt.legend()

plt.tight_layout()
plt.savefig('data/joint_angles_3steps.png', dpi=300)
plt.show()
