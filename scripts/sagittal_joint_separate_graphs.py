csv_path = 'data/run_20260310_152301.csv'

import csv
import matplotlib.pyplot as plt
import os

# Robust path handling
csv_path = 'data/run_20260310_152301.csv'
csv_path = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'

rows = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

sim_time = [float(r['sim_time_s']) for r in rows]
hip_right = [float(r['hip_right_rad']) for r in rows]
hip_left = [float(r['hip_left_rad']) for r in rows]
knee_right = [float(r['knee_right_rad']) for r in rows]
knee_left = [float(r['knee_left_rad']) for r in rows]
hip_height = [float(r['slope_height_m']) for r in rows]

# Single figure with 3 subplots
plt.figure(figsize=(12, 10))

# Hip angles
plt.subplot(3, 1, 1)
plt.plot(sim_time, hip_right, label='Hip Right', color='blue')
plt.plot(sim_time, hip_left, label='Hip Left', color='cyan', linestyle='--')
plt.axhline(0, color='gray', linewidth=0.7)
plt.ylabel('Hip Angle (rad)')
plt.title('Sagittal Plane: Hip Angles')
plt.legend()

# Knee angles
plt.subplot(3, 1, 2)
plt.plot(sim_time, knee_right, label='Knee Right', color='green')
plt.plot(sim_time, knee_left, label='Knee Left', color='lime', linestyle='--')
plt.axhline(0, color='gray', linewidth=0.7)
plt.ylabel('Knee Angle (rad)')
plt.title('Sagittal Plane: Knee Angles')
plt.legend()


# Use absolute paths for CSV and PNG
csv_path = '/home/elec330-admin/ros2_ws/data/run_20260310_153542.csv'
png_path = '/home/elec330-admin/ros2_ws/data/sagittal_joint_summary.png'

rows = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

sim_time = [float(r['sim_time_s']) for r in rows]
hip_right = [float(r['hip_right_rad']) for r in rows]
hip_left = [float(r['hip_left_rad']) for r in rows]
knee_right = [float(r['knee_right_rad']) for r in rows]
knee_left = [float(r['knee_left_rad']) for r in rows]
hip_height = [float(r['slope_height_m']) for r in rows]

plt.figure(figsize=(12, 10))

# Hip angles
plt.subplot(3, 1, 1)
plt.plot(sim_time, hip_right, label='Hip Right', color='blue')
plt.plot(sim_time, hip_left, label='Hip Left', color='cyan', linestyle='--')
plt.axhline(0, color='gray', linewidth=0.7)
plt.ylabel('Hip Angle (rad)')
plt.title('Sagittal Plane: Hip Angles')
plt.legend()

# Knee angles
plt.subplot(3, 1, 2)
plt.plot(sim_time, knee_right, label='Knee Right', color='green')
plt.plot(sim_time, knee_left, label='Knee Left', color='lime', linestyle='--')
plt.axhline(0, color='gray', linewidth=0.7)
plt.ylabel('Knee Angle (rad)')
plt.title('Sagittal Plane: Knee Angles')
plt.legend()

# Hip height
plt.subplot(3, 1, 3)
plt.plot(sim_time, hip_height, label='Hip Height (slope frame)', color='magenta')
plt.ylabel('Hip Height (m)')
plt.xlabel('Time (s)')
plt.title('Sagittal Plane: Hip Height Over Time')
plt.legend()

plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.show()
