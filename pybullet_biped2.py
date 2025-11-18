"""
PyBullet Passive Walker with KNEES + COUNTER-SWING ARMS
Complete bipedal design:
- Thigh and shin segments (knees with limits)
- Counter-swinging arms for balance
- Starts mid-gait on extended slope
- Passive dynamics only
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

class KneedWalkerConfig:
    """Complete walker configuration"""
    
    def __init__(self):
        # LEG GEOMETRY (meters)
        self.thigh_length = 0.45
        self.shin_length = 0.43
        self.leg_radius = 0.025
        
        # ARM GEOMETRY (meters)
        self.upper_arm_length = 0.30
        self.forearm_length = 0.25
        self.arm_radius = 0.02
        
        # MASS (kg)
        self.hip_mass = 10.0
        self.thigh_mass = 7.0
        self.shin_mass = 3.5
        self.foot_mass = 0.5
        self.upper_arm_mass = 2.0
        self.forearm_mass = 1.5
        
        # JOINT CONSTRAINTS (degrees)
        self.min_knee_angle = 0
        self.max_knee_angle = 65
        self.min_elbow_angle = 0
        self.max_elbow_angle = 90
        
        # SLOPE
        self.slope_angle_deg = 3.0
        self.slope_length = 50.0      # Extended!
        self.slope_width = 3.0
        
        # PHYSICS
        self.gravity = -9.81
        self.time_step = 1/240
        
        # SPAWN POSITION - Mid-gait on slope
        self.spawn_distance_down_slope = 5.0  # meters from top
        self.spawn_height_above_ground = 0.02  # Just above surface

config = KneedWalkerConfig()

# ============================================================================
# GENERATE URDF WITH ARMS
# ============================================================================

def generate_walker_urdf(config, filename="armed_walker.urdf"):
    """
    Generate URDF with legs AND arms
    Structure: 
      Hip (base)
        ├─ Thigh1 -> Shin1 -> Foot1
        ├─ Thigh2 -> Shin2 -> Foot2
        ├─ UpperArm1 -> Forearm1
        └─ UpperArm2 -> Forearm2
    """
    
    urdf = f"""<?xml version="1.0"?>
<robot name="armed_walker">
  
  <!-- Materials -->
  <material name="blue"><color rgba="0 0 1 1"/></material>
  <material name="red"><color rgba="1 0 0 1"/></material>
  <material name="green"><color rgba="0 1 0 1"/></material>
  <material name="orange"><color rgba="1 0.5 0 1"/></material>
  <material name="purple"><color rgba="0.5 0 1 1"/></material>
  <material name="yellow"><color rgba="1 1 0 1"/></material>
  
  <!-- HIP (Base) -->
  <link name="hip">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{config.hip_mass}"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.15 0.1"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.15 0.1"/>
      </geometry>
    </collision>
  </link>
  
  <!-- ================================================================ -->
  <!-- LEG 1 (Right/Stance) - BLUE -->
  <!-- ================================================================ -->
  
  <link name="thigh1">
    <inertial>
      <origin xyz="0 0 -{config.thigh_length/2}" rpy="0 0 0"/>
      <mass value="{config.thigh_mass}"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -{config.thigh_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.leg_radius}" length="{config.thigh_length}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -{config.thigh_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.leg_radius}" length="{config.thigh_length}"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="hip1" type="continuous">
    <parent link="hip"/>
    <child link="thigh1"/>
    <origin xyz="0 0.06 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.02" friction="0.0"/>
  </joint>
  
  <link name="shin1">
    <inertial>
      <origin xyz="0 0 -{config.shin_length/2}" rpy="0 0 0"/>
      <mass value="{config.shin_mass}"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -{config.shin_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.leg_radius}" length="{config.shin_length}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -{config.shin_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.leg_radius}" length="{config.shin_length}"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="knee1" type="revolute">
    <parent link="thigh1"/>
    <child link="shin1"/>
    <origin xyz="0 0 -{config.thigh_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="{np.radians(config.min_knee_angle)}" upper="{np.radians(config.max_knee_angle)}" effort="0" velocity="10"/>
    <dynamics damping="0.02" friction="0.0"/>
  </joint>
  
  <link name="foot1">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{config.foot_mass}"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.035"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.035"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="ankle1" type="fixed">
    <parent link="shin1"/>
    <child link="foot1"/>
    <origin xyz="0 0 -{config.shin_length}" rpy="0 0 0"/>
  </joint>
  
  <!-- ================================================================ -->
  <!-- LEG 2 (Left/Swing) - ORANGE -->
  <!-- ================================================================ -->
  
  <link name="thigh2">
    <inertial>
      <origin xyz="0 0 -{config.thigh_length/2}" rpy="0 0 0"/>
      <mass value="{config.thigh_mass}"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -{config.thigh_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.leg_radius}" length="{config.thigh_length}"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 -{config.thigh_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.leg_radius}" length="{config.thigh_length}"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="hip2" type="continuous">
    <parent link="hip"/>
    <child link="thigh2"/>
    <origin xyz="0 -0.06 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.02" friction="0.0"/>
  </joint>
  
  <link name="shin2">
    <inertial>
      <origin xyz="0 0 -{config.shin_length/2}" rpy="0 0 0"/>
      <mass value="{config.shin_mass}"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -{config.shin_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.leg_radius}" length="{config.shin_length}"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 -{config.shin_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.leg_radius}" length="{config.shin_length}"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="knee2" type="revolute">
    <parent link="thigh2"/>
    <child link="shin2"/>
    <origin xyz="0 0 -{config.thigh_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="{np.radians(config.min_knee_angle)}" upper="{np.radians(config.max_knee_angle)}" effort="0" velocity="10"/>
    <dynamics damping="0.02" friction="0.0"/>
  </joint>
  
  <link name="foot2">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{config.foot_mass}"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.035"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.035"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="ankle2" type="fixed">
    <parent link="shin2"/>
    <child link="foot2"/>
    <origin xyz="0 0 -{config.shin_length}" rpy="0 0 0"/>
  </joint>
  
  <!-- ================================================================ -->
  <!-- ARM 1 (Right) - PURPLE (Counter-swings with left leg) -->
  <!-- ================================================================ -->
  
  <link name="upper_arm1">
    <inertial>
      <origin xyz="0 0 {config.upper_arm_length/2}" rpy="0 0 0"/>
      <mass value="{config.upper_arm_mass}"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {config.upper_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.arm_radius}" length="{config.upper_arm_length}"/>
      </geometry>
      <material name="purple"/>
    </visual>
    <collision>
      <origin xyz="0 0 {config.upper_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.arm_radius}" length="{config.upper_arm_length}"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="shoulder1" type="continuous">
    <parent link="hip"/>
    <child link="upper_arm1"/>
    <origin xyz="0 0.12 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.01" friction="0.0"/>
  </joint>
  
  <link name="forearm1">
    <inertial>
      <origin xyz="0 0 {config.forearm_length/2}" rpy="0 0 0"/>
      <mass value="{config.forearm_mass}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {config.forearm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.arm_radius}" length="{config.forearm_length}"/>
      </geometry>
      <material name="purple"/>
    </visual>
    <collision>
      <origin xyz="0 0 {config.forearm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.arm_radius}" length="{config.forearm_length}"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="elbow1" type="revolute">
    <parent link="upper_arm1"/>
    <child link="forearm1"/>
    <origin xyz="0 0 {config.upper_arm_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="{np.radians(config.min_elbow_angle)}" upper="{np.radians(config.max_elbow_angle)}" effort="0" velocity="10"/>
    <dynamics damping="0.01" friction="0.0"/>
  </joint>
  
  <!-- ================================================================ -->
  <!-- ARM 2 (Left) - YELLOW (Counter-swings with right leg) -->
  <!-- ================================================================ -->
  
  <link name="upper_arm2">
    <inertial>
      <origin xyz="0 0 {config.upper_arm_length/2}" rpy="0 0 0"/>
      <mass value="{config.upper_arm_mass}"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {config.upper_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.arm_radius}" length="{config.upper_arm_length}"/>
      </geometry>
      <material name="yellow"/>
    </visual>
    <collision>
      <origin xyz="0 0 {config.upper_arm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.arm_radius}" length="{config.upper_arm_length}"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="shoulder2" type="continuous">
    <parent link="hip"/>
    <child link="upper_arm2"/>
    <origin xyz="0 -0.12 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.01" friction="0.0"/>
  </joint>
  
  <link name="forearm2">
    <inertial>
      <origin xyz="0 0 {config.forearm_length/2}" rpy="0 0 0"/>
      <mass value="{config.forearm_mass}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {config.forearm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.arm_radius}" length="{config.forearm_length}"/>
      </geometry>
      <material name="yellow"/>
    </visual>
    <collision>
      <origin xyz="0 0 {config.forearm_length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{config.arm_radius}" length="{config.forearm_length}"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="elbow2" type="revolute">
    <parent link="upper_arm2"/>
    <child link="forearm2"/>
    <origin xyz="0 0 {config.upper_arm_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="{np.radians(config.min_elbow_angle)}" upper="{np.radians(config.max_elbow_angle)}" effort="0" velocity="10"/>
    <dynamics damping="0.01" friction="0.0"/>
  </joint>
  
</robot>
"""
    
    with open(filename, 'w') as f:
        f.write(urdf)
    
    return filename

# ============================================================================
# CREATE EXTENDED SLOPE
# ============================================================================

def create_extended_slope(config):
    """Create long sloped ramp"""
    
    slope_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[config.slope_length/2, config.slope_width/2, 0.05]
    )
    
    slope_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[config.slope_length/2, config.slope_width/2, 0.05],
        rgbaColor=[0.6, 0.4, 0.2, 1]
    )
    
    # Position slope
    slope_angle_rad = np.radians(config.slope_angle_deg)
    slope_position = [config.slope_length/2 * np.cos(slope_angle_rad), 
                     0, 
                     -config.slope_length/2 * np.sin(slope_angle_rad)]
    
    slope_orientation = p.getQuaternionFromEuler([0, -slope_angle_rad, 0])
    
    slope_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=slope_collision,
        baseVisualShapeIndex=slope_visual,
        basePosition=slope_position,
        baseOrientation=slope_orientation
    )
    
    p.changeDynamics(slope_id, -1, lateralFriction=1.2, restitution=0.0)
    
    return slope_id, slope_angle_rad

# ============================================================================
# CALCULATE SPAWN POSITION
# ============================================================================

def calculate_spawn_position(config, slope_angle_rad):
    """Calculate where to spawn robot on slope for mid-gait start"""
    
    # Distance down slope
    x_on_slope = config.spawn_distance_down_slope
    
    # Calculate 3D position on slope surface
    x_world = x_on_slope * np.cos(slope_angle_rad)
    z_world = -x_on_slope * np.sin(slope_angle_rad)
    
    # Add height above surface for hip position
    total_leg_length = config.thigh_length + config.shin_length
    z_world += total_leg_length + config.spawn_height_above_ground
    
    return [x_world, 0, z_world]

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_armed_walker(config, gui=True, duration=15.0):
    """Run complete walker simulation with arms"""
    
    if gui:
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        # Set better camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=50,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.5]
        )
        print("✓ PyBullet GUI opened")
    else:
        physics_client = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, config.gravity)
    p.setTimeStep(config.time_step)
    
    # Add visual grid
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable during setup
    
    # Create world
    print("  Creating extended slope...")
    slope_id, slope_angle_rad = create_extended_slope(config)
    print(f"    Slope ID: {slope_id}")
    
    print("  Generating robot URDF with arms...")
    urdf_file = generate_walker_urdf(config)
    print(f"    URDF saved: {urdf_file}")
    
    print("  Calculating spawn position...")
    spawn_pos = calculate_spawn_position(config, slope_angle_rad)
    spawn_orn = p.getQuaternionFromEuler([0, 0, 0])  # Start upright first
    
    print(f"    Spawn position: {spawn_pos}")
    print(f"    Spawn orientation: {spawn_orn}")
    
    print("  Loading robot...")
    try:
        robot_id = p.loadURDF(urdf_file, spawn_pos, spawn_orn, useFixedBase=False)
        print(f"    ✓ Robot loaded! ID: {robot_id}")
    except Exception as e:
        print(f"    ❌ Failed to load robot: {e}")
        p.disconnect()
        return
    
    # Enable rendering now
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    # Get joint info
    joint_dict = {}
    num_joints = p.getNumJoints(robot_id)
    print(f"    Robot has {num_joints} joints")
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        joint_dict[joint_name] = i
        print(f"      Joint {i}: {joint_name}")
    
    # Check robot position
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    print(f"\n  Robot current position: {pos}")
    print(f"  Robot current orientation: {orn}")
    
    print("\n  Setting mid-gait starting pose...")
    
    # Legs: Right leg back (stance), left leg forward (swing)
    p.resetJointState(robot_id, joint_dict['hip1'], np.radians(-10))
    p.resetJointState(robot_id, joint_dict['knee1'], np.radians(5))
    p.resetJointState(robot_id, joint_dict['hip2'], np.radians(20))
    p.resetJointState(robot_id, joint_dict['knee2'], np.radians(15))
    
    # Arms: Counter-swing
    p.resetJointState(robot_id, joint_dict['shoulder1'], np.radians(25))
    p.resetJointState(robot_id, joint_dict['elbow1'], np.radians(20))
    p.resetJointState(robot_id, joint_dict['shoulder2'], np.radians(-20))
    p.resetJointState(robot_id, joint_dict['elbow2'], np.radians(15))
    
    # Set velocities
    p.resetJointStateMultiDof(robot_id, joint_dict['hip1'], 
                               targetValue=[np.radians(-10)], targetVelocity=[0.5])
    p.resetJointStateMultiDof(robot_id, joint_dict['hip2'],
                               targetValue=[np.radians(20)], targetVelocity=[-0.8])
    p.resetJointStateMultiDof(robot_id, joint_dict['shoulder1'],
                               targetValue=[np.radians(25)], targetVelocity=[-1.0])
    p.resetJointStateMultiDof(robot_id, joint_dict['shoulder2'],
                               targetValue=[np.radians(-20)], targetVelocity=[0.8])
    
    # Set friction on feet
    for foot_joint in ['foot1', 'foot2']:
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            if info[12].decode('utf-8') == foot_joint:
                p.changeDynamics(robot_id, i, lateralFriction=1.2, restitution=0.0)
                print(f"    Set friction on {foot_joint}")
    
    print("\n✓ Robot setup complete! You should see it now.")
    print("  If you don't see it, try:")
    print("    - Use mouse to rotate view")
    print("    - Scroll to zoom out")
    print("    - Look for green box (hip)")
    
    # Wait a moment for user to see robot
    if gui:
        print("\n⏸  Pausing 3 seconds so you can see the robot...")
        for i in range(3):
            p.stepSimulation()
            time.sleep(1)
            print(f"    {3-i}...")
    
    # Data collection
    data = {
        'times': [],
        'hip_pos': [],
        'hip1': [], 'knee1': [],
        'hip2': [], 'knee2': [],
        'shoulder1': [], 'shoulder2': []
    }
    
    print(f"\n🏃 Starting {duration} second simulation...")
    print("  LEGEND:")
    print("    🔵 Blue = Right leg")
    print("    🟠 Orange = Left leg")
    print("    🟣 Purple = Right arm")
    print("    🟡 Yellow = Left arm")
    print("    🟢 Green = Hip")
    
    steps = int(duration / config.time_step)
    for step in range(steps):
        p.stepSimulation()
        
        if step % 10 == 0:
            t = step * config.time_step
            data['times'].append(t)
            
            hip_pos, _ = p.getBasePositionAndOrientation(robot_id)
            data['hip_pos'].append(hip_pos)
            
            data['hip1'].append(np.degrees(p.getJointState(robot_id, joint_dict['hip1'])[0]))
            data['knee1'].append(np.degrees(p.getJointState(robot_id, joint_dict['knee1'])[0]))
            data['hip2'].append(np.degrees(p.getJointState(robot_id, joint_dict['hip2'])[0]))
            data['knee2'].append(np.degrees(p.getJointState(robot_id, joint_dict['knee2'])[0]))
            data['shoulder1'].append(np.degrees(p.getJointState(robot_id, joint_dict['shoulder1'])[0]))
            data['shoulder2'].append(np.degrees(p.getJointState(robot_id, joint_dict['shoulder2'])[0]))
        
        # Show progress
        if step % 1000 == 0:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            print(f"    t={step*config.time_step:.1f}s, pos={pos[0]:.2f}m, height={pos[2]:.2f}m")
        
        if gui and step % 2 == 0:
            time.sleep(config.time_step * 0.3)
    
    print("✓ Simulation complete!")
    
    # Analysis
    hip_positions = np.array(data['hip_pos'])
    distance = hip_positions[-1][0] - hip_positions[0][0]
    velocity = distance / duration
    
    print(f"\n📊 WALKING ANALYSIS:")
    print(f"   Distance traveled: {distance:.2f} m")
    print(f"   Average velocity: {velocity:.2f} m/s")
    print(f"   Final height: {hip_positions[-1][2]:.2f} m")
    print(f"   Knee 1 range: {min(data['knee1']):.1f}° to {max(data['knee1']):.1f}°")
    print(f"   Knee 2 range: {min(data['knee2']):.1f}° to {max(data['knee2']):.1f}°")
    print(f"   Arm swing: Right {min(data['shoulder1']):.1f}° to {max(data['shoulder1']):.1f}°")
    
    success = hip_positions[-1][2] > 0.3 and distance > 1.0
    print(f"\n   {'✓ STABLE WALKING!' if success else '⚠ Walker fell or stalled'}")
    
    # Visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Leg angles
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(data['times'], data['hip1'], 'b-', linewidth=2, label='Right Hip')
    ax1.plot(data['times'], data['hip2'], 'orange', linewidth=2, label='Left Hip')
    ax1.set_ylabel('Hip Angle (°)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Hip Joints', fontweight='bold')
    
    # Knee angles
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(data['times'], data['knee1'], 'b-', linewidth=2, label='Right Knee')
    ax2.plot(data['times'], data['knee2'], 'orange', linewidth=2, label='Left Knee')
    ax2.axhline(config.min_knee_angle, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(config.max_knee_angle, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Knee Angle (°)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Knee Joints (with limits)', fontweight='bold')
    
    # Arm angles
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(data['times'], data['shoulder1'], 'purple', linewidth=2, label='Right Arm')
    ax3.plot(data['times'], data['shoulder2'], 'gold', linewidth=2, label='Left Arm')
    ax3.set_ylabel('Shoulder Angle (°)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Arm Counter-Swing', fontweight='bold')
    
    # Hip trajectory
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(hip_positions[:,0], hip_positions[:,2], 'g-', linewidth=2)
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Height (m)')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Hip Trajectory (Forward Motion)', fontweight='bold')
    
    # Forward progress
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(data['times'], hip_positions[:,0], 'purple', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Forward Distance (m)')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Distance vs Time', fontweight='bold')
    
    # Coordination plot (legs vs arms)
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(data['times'], data['hip1'], 'b-', linewidth=1.5, label='Right Leg', alpha=0.7)
    ax6.plot(data['times'], data['hip2'], 'orange', linewidth=1.5, label='Left Leg', alpha=0.7)
    ax6.plot(data['times'], data['shoulder1'], 'purple', linewidth=1.5, label='Right Arm', alpha=0.7)
    ax6.plot(data['times'], data['shoulder2'], 'gold', linewidth=1.5, label='Left Arm', alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Joint Angle (°)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Coordination: Arms Counter-Swing with Legs', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 INTERPRETATION:")
    print("   - Arms should swing opposite to legs (counter-swing)")
    print("   - Right arm forward when left leg forward (and vice versa)")
    print("   - This provides angular momentum balance")
    print("   - Knees should bend during swing phase for ground clearance")
    
    if gui:
        print("\n👁 Close PyBullet window when done")
        input("Press Enter to exit...")
    
    p.disconnect()

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 COMPLETE PASSIVE WALKER SIMULATION")
    print("Features: Knees + Counter-Swing Arms + Extended Slope")
    print("=" * 70)
    
    config = KneedWalkerConfig()
    
    print("\n📋 ROBOT SPECIFICATIONS:")
    print(f"\n  LEGS:")
    print(f"    Thigh: {config.thigh_length}m, {config.thigh_mass}kg")
    print(f"    Shin:  {config.shin_length}m, {config.shin_mass}kg")
    print(f"    Knee limits: {config.min_knee_angle}° to {config.max_knee_angle}°")
    
    print(f"\n  ARMS:")
    print(f"    Upper arm: {config.upper_arm_length}m, {config.upper_arm_mass}kg")
    print(f"    Forearm:   {config.forearm_length}m, {config.forearm_mass}kg")
    print(f"    Elbow limits: {config.min_elbow_angle}° to {config.max_elbow_angle}°")
    
    print(f"\n  BODY:")
    print(f"    Hip mass: {config.hip_mass}kg")
    print(f"    Total mass: {config.hip_mass + 2*config.thigh_mass + 2*config.shin_mass + 2*config.foot_mass + 2*config.upper_arm_mass + 2*config.forearm_mass:.1f}kg")
    
    print(f"\n  ENVIRONMENT:")
    print(f"    Slope: {config.slope_angle_deg}°")
    print(f"    Slope length: {config.slope_length}m")
    print(f"    Spawn position: {config.spawn_distance_down_slope}m down slope")
    
    print("\n" + "=" * 70)
    print("🎬 Starting simulation...")
    print("=" * 70)
    
    run_armed_walker(config, gui=True, duration=15.0)
    
    print("\n" + "=" * 70)
    print("✅ SIMULATION COMPLETE!")
    print("\n📝 NEXT STEPS:")
    print("   1. Adjust parameters in KneedWalkerConfig class")
    print("   2. Experiment with different slopes, masses, arm lengths")
    print("   3. Once stable gait found, export parameters to CAD")
    print("   4. Use these dimensions for 3D printing!")
    print("=" * 70)
