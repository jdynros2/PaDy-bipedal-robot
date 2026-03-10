import csv
import matplotlib.pyplot as plt

csv_path = 'data/run_20260310_152301.csv'
rows = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Extract columns
sim_time = [float(r['sim_time_s']) for r in rows]
hip_right = [float(r['hip_right_rad']) for r in rows]
hip_left = [float(r['hip_left_rad']) for r in rows]
knee_right = [float(r['knee_right_rad']) for r in rows]
knee_left = [float(r['knee_left_rad']) for r in rows]
hip_height = [float(r['base_y_m']) for r in rows]

# Plot hip and knee angles (sagittal plane)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(sim_time, hip_right, label='Hip Right', color='blue')
plt.plot(sim_time, hip_left, label='Hip Left', color='cyan', linestyle='--')
plt.plot(sim_time, knee_right, label='Knee Right', color='green')
plt.plot(sim_time, knee_left, label='Knee Left', color='lime', linestyle='--')
plt.axhline(0, color='gray', linewidth=0.7)
plt.ylabel('Angle (rad)')
plt.title('Sagittal Plane: Hip & Knee Angles')
plt.legend()

# Plot hip height
plt.subplot(2, 1, 2)
plt.plot(sim_time, hip_height, label='Hip Height (Y)', color='magenta')
plt.ylabel('Hip Height (m)')
plt.xlabel('Time (s)')
plt.title('Sagittal Plane: Hip Height Over Time')
plt.legend()

plt.tight_layout()
plt.savefig('data/sagittal_joint_height_curve.png', dpi=300)
plt.show()
