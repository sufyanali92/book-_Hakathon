---
title: "Simulated Sensors and Perception"
sidebar_position: 2
---

# 2. Simulated Sensors and Perception

## Introduction

Sensors are the eyes and ears of robotic systems, providing the raw data that enables perception and decision-making. In simulation environments, accurately modeling sensors is crucial for developing and testing perception algorithms that will eventually run on real robots. This chapter explores the principles and implementation of simulated sensors, with a focus on perception systems that process sensor data to understand the environment.

## Learning Objectives

- Understand the principles of sensor simulation in robotics
- Identify different types of simulated sensors and their characteristics
- Design perception algorithms that work with simulated sensor data
- Recognize the differences between simulated and real sensors
- Validate perception algorithms across simulation and reality

## Conceptual Foundations

Simulated sensors in robotics must balance several competing requirements:

**Realism vs. Performance**: Sensor models must be realistic enough to provide meaningful training and testing data while maintaining simulation performance. This balance is critical for practical development workflows.

**Physics-Based Modeling**: High-fidelity sensor simulation incorporates the underlying physics of sensing, including light propagation for cameras, wave propagation for LIDAR and sonar, and electromagnetic effects for other sensors.

**Noise and Imperfections**: Real sensors have noise, bias, and other imperfections that must be modeled to ensure algorithms are robust to real-world conditions.

**Latency and Timing**: Sensors have inherent delays and update rates that must be accurately simulated to test real-time perception systems.

**Cross-Modal Integration**: Modern perception systems often combine data from multiple sensor types, requiring careful coordination between simulated sensors.

## Technical Explanation

### Types of Simulated Sensors

**Camera Sensors**: Simulate visual perception with:
- Geometric projection models
- Lens distortion effects
- Photometric properties (exposure, gain, noise)
- Dynamic range and quantization effects
- Frame rate and latency characteristics

**LIDAR Sensors**: Model light detection and ranging with:
- Beam propagation and reflection
- Angular resolution and range accuracy
- Multiple return capabilities
- Motion distortion effects
- Reflectivity-based intensity modeling

**IMU Sensors**: Simulate inertial measurement with:
- Accelerometer and gyroscope models
- Bias, drift, and noise characteristics
- Cross-axis sensitivity
- Temperature effects
- Vibration and shock responses

**Other Sensors**: Include models for:
- GPS with accuracy and multipath effects
- Sonar with beam patterns and acoustic properties
- Force/torque sensors with compliance and noise
- Magnetic sensors with interference modeling

### Perception Pipeline in Simulation

The perception pipeline in simulation typically includes:

**Raw Data Generation**: The physics engine generates raw sensor measurements based on the environment model and sensor properties.

**Preprocessing**: Raw data is processed to correct for known sensor characteristics (e.g., lens distortion, temperature compensation).

**Feature Extraction**: Algorithms extract meaningful features from sensor data (e.g., edges, corners, objects).

**State Estimation**: Algorithms estimate the state of the environment (e.g., object positions, robot pose).

**Scene Understanding**: Higher-level algorithms interpret the scene to enable decision-making.

### Sensor Noise and Uncertainty

Simulated sensors incorporate various types of noise and uncertainty:

- **Gaussian Noise**: Random variations in measurements
- **Bias**: Systematic offsets in measurements
- **Drift**: Slow changes in sensor characteristics over time
- **Quantization**: Discretization effects due to finite resolution
- **Temporal Effects**: Latency, jitter, and timing variations

## Practical Examples

### Example 1: Camera Sensor with Noise Model

Implementing a camera sensor with realistic noise characteristics:

```python
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math

class SimulatedCamera:
    def __init__(self, width=640, height=480, fov=60, noise_level=0.01):
        self.width = width
        self.height = height
        self.fov = fov  # Field of view in degrees
        self.noise_level = noise_level  # Noise level as fraction of signal

        # Camera intrinsic parameters
        focal_length = width / (2 * math.tan(math.radians(fov/2)))
        self.camera_matrix = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ])

        # Distortion coefficients [k1, k2, p1, p2, k3]
        self.distortion_coeffs = np.array([0.1, -0.2, 0.001, 0.001, 0.05])

        # Noise parameters
        self.read_noise_std = 2.0  # Standard deviation of read noise
        self.photon_noise_factor = 0.02  # Factor for photon noise
        self.bias = 10  # Bias offset

        self.cv_bridge = CvBridge()

    def simulate_image(self, scene_data):
        """
        Simulate camera image from scene data
        scene_data: 3D scene information
        """
        # Simulate the ideal image based on scene
        ideal_image = self.render_ideal_image(scene_data)

        # Apply lens distortion
        distorted_image = self.apply_distortion(ideal_image)

        # Add various types of noise
        noisy_image = self.add_noise(distorted_image)

        # Apply quantization to simulate digital sensor
        digital_image = self.quantize_image(noisy_image)

        return digital_image

    def render_ideal_image(self, scene_data):
        """Render ideal image from 3D scene"""
        # This would typically interface with a rendering engine
        # For simulation, we'll create a simple synthetic image
        image = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

        # Add some synthetic features
        cv2.circle(image, (self.width//2, self.height//2), 50, (255, 0, 0), -1)
        cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), 2)

        return image.astype(np.float32)

    def apply_distortion(self, image):
        """Apply lens distortion to image"""
        h, w = image.shape[:2]

        # Generate new camera matrix with optimal principal point
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coeffs, (w, h), 1, (w, h))

        # Undistort the image
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.distortion_coeffs, None, new_camera_matrix)

        return undistorted

    def add_noise(self, image):
        """Add realistic sensor noise"""
        # Convert to appropriate range for noise calculations
        image_normalized = image / 255.0

        # Add photon noise (proportional to signal)
        photon_noise = np.random.normal(
            0, self.photon_noise_factor * np.sqrt(image_normalized), image_normalized.shape)

        # Add read noise (constant)
        read_noise = np.random.normal(0, self.read_noise_std/255.0, image_normalized.shape)

        # Combine noises
        total_noise = photon_noise + read_noise

        # Add noise to image
        noisy_image = image_normalized + total_noise

        # Add bias
        noisy_image += self.bias / 255.0

        return np.clip(noisy_image, 0, 1) * 255

    def quantize_image(self, image):
        """Apply quantization to simulate digital sensor"""
        # Convert to 8-bit integer
        quantized = np.round(image).astype(np.uint8)
        return quantized

    def get_ros_image(self, scene_data):
        """Get ROS Image message from simulated camera"""
        simulated_image = self.simulate_image(scene_data)
        ros_image = self.cv_bridge.cv2_to_imgmsg(simulated_image, "bgr8")
        ros_image.header.stamp = self.get_current_time()
        ros_image.header.frame_id = "camera_frame"
        return ros_image

    def get_current_time(self):
        """Get current time (placeholder)"""
        import time
        from builtin_interfaces.msg import Time
        current_time = Time()
        current_time.sec = int(time.time())
        current_time.nanosec = int((time.time() % 1) * 1e9)
        return current_time

# Example usage
camera = SimulatedCamera(width=640, height=480, fov=60, noise_level=0.02)
simulated_scene = {}  # Placeholder for scene data
image = camera.simulate_image(simulated_scene)
print(f"Simulated image shape: {image.shape}")
```

### Example 2: Multi-Sensor Fusion for Object Detection

Combining data from multiple simulated sensors for perception:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Dict
import math

class MultiSensorPerception:
    def __init__(self):
        # Initialize sensor models
        self.camera = SimulatedCamera()
        self.lidar = SimulatedLidar()
        self.imu = SimulatedIMU()

        # Sensor poses relative to robot base
        self.sensor_poses = {
            'camera': np.array([0.1, 0.0, 0.2]),  # x, y, z offset
            'lidar': np.array([0.0, 0.0, 0.3]),
            'imu': np.array([0.0, 0.0, 0.1])
        }

        # Covariance matrices for each sensor
        self.sensor_covariances = {
            'camera': np.diag([0.01, 0.01, 0.05]),  # x, y, angle
            'lidar': np.diag([0.02, 0.02, 0.02]),
            'imu': np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])  # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        }

    def detect_objects_camera(self, image):
        """Detect objects in camera image"""
        # Simulate object detection (in practice, this would use computer vision)
        detected_objects = []

        # For simulation, we'll detect the synthetic features we added
        # In real implementation, this would use actual detection algorithms
        # like YOLO, SSD, or custom detectors

        # Simulate detection of a blue circle
        detected_objects.append({
            'class': 'circle',
            'confidence': 0.85,
            'bbox': [270, 190, 350, 270],  # x1, y1, x2, y2
            'center_2d': [310, 230]  # x, y in image coordinates
        })

        # Simulate detection of a green rectangle
        detected_objects.append({
            'class': 'rectangle',
            'confidence': 0.92,
            'bbox': [100, 100, 200, 200],
            'center_2d': [150, 150]
        })

        return detected_objects

    def detect_objects_lidar(self, pointcloud):
        """Detect objects in LIDAR pointcloud"""
        # Simulate pointcloud-based object detection
        detected_objects = []

        # For simulation, we'll create some clusters
        # In real implementation, this would use clustering algorithms
        # like DBSCAN, Euclidean clustering, etc.

        # Simulate detection of ground plane and obstacles
        for i in range(3):  # Simulate 3 obstacles
            obstacle = {
                'type': 'obstacle',
                'position': [i * 2.0, 0.0, 0.5],  # x, y, z
                'size': [0.5, 0.5, 1.0],  # width, depth, height
                'confidence': 0.75 + 0.1 * np.random.random()
            }
            detected_objects.append(obstacle)

        return detected_objects

    def fuse_sensor_data(self, camera_objects, lidar_objects, robot_pose, timestamp):
        """Fuse data from multiple sensors"""
        fused_objects = []

        # For each camera detection, try to associate with LIDAR detection
        for cam_obj in camera_objects:
            # Convert 2D camera detection to 3D using depth from LIDAR
            # This is a simplified approach
            closest_lidar_obj = self.find_closest_lidar_object(
                cam_obj['center_2d'], lidar_objects, robot_pose)

            if closest_lidar_obj and self.is_consistent(cam_obj, closest_lidar_obj):
                # Fuse the detections
                fused_obj = self.combine_detections(cam_obj, closest_lidar_obj, robot_pose)
                fused_objects.append(fused_obj)

        # Add LIDAR-only detections that don't have camera matches
        for lidar_obj in lidar_objects:
            # Check if this LIDAR object was already fused with a camera object
            already_fused = False
            for fused_obj in fused_objects:
                if (abs(fused_obj['position'][0] - lidar_obj['position'][0]) < 0.5 and
                    abs(fused_obj['position'][1] - lidar_obj['position'][1]) < 0.5):
                    already_fused = True
                    break

            if not already_fused:
                fused_objects.append({
                    'type': lidar_obj['type'],
                    'position': lidar_obj['position'],
                    'size': lidar_obj['size'],
                    'confidence': lidar_obj['confidence'],
                    'sensor_sources': ['lidar']
                })

        return fused_objects

    def find_closest_lidar_object(self, camera_center, lidar_objects, robot_pose):
        """Find the closest LIDAR object to a camera detection"""
        if not lidar_objects:
            return None

        # Convert camera 2D point to 3D ray
        # This is a simplified approach - in practice, you'd use camera calibration
        # and depth estimation
        closest_obj = None
        min_distance = float('inf')

        for lidar_obj in lidar_objects:
            # Calculate distance in robot's coordinate frame
            distance = math.sqrt(
                (lidar_obj['position'][0] - robot_pose[0])**2 +
                (lidar_obj['position'][1] - robot_pose[1])**2
            )

            if distance < min_distance:
                min_distance = distance
                closest_obj = lidar_obj

        return closest_obj

    def is_consistent(self, camera_obj, lidar_obj):
        """Check if camera and LIDAR detections are consistent"""
        # This is a simplified consistency check
        # In practice, you'd use more sophisticated geometric and temporal checks
        return True

    def combine_detections(self, camera_obj, lidar_obj, robot_pose):
        """Combine camera and LIDAR detections"""
        # Combine information from both sensors
        fused_obj = {
            'type': camera_obj['class'],
            'position': lidar_obj['position'],  # Use LIDAR for position
            'size': lidar_obj['size'],  # Use LIDAR for size
            'confidence': (camera_obj['confidence'] + lidar_obj['confidence']) / 2,
            'sensor_sources': ['camera', 'lidar'],
            'camera_info': camera_obj,
            'lidar_info': lidar_obj
        }

        return fused_obj

    def run_perception_pipeline(self, scene_data, robot_pose, timestamp):
        """Run the complete perception pipeline"""
        # Get sensor data
        camera_image = self.camera.simulate_image(scene_data)
        lidar_pointcloud = self.lidar.simulate_scan(scene_data)
        imu_data = self.imu.simulate_reading()

        # Process each sensor modality
        camera_objects = self.detect_objects_camera(camera_image)
        lidar_objects = self.detect_objects_lidar(lidar_pointcloud)

        # Fuse sensor data
        fused_objects = self.fuse_sensor_data(
            camera_objects, lidar_objects, robot_pose, timestamp)

        return fused_objects

class SimulatedLidar:
    def __init__(self, num_beams=360, max_range=10.0, noise_std=0.01):
        self.num_beams = num_beams
        self.max_range = max_range
        self.noise_std = noise_std

    def simulate_scan(self, scene_data):
        """Simulate LIDAR scan from scene"""
        # Simulate a simple scan with some obstacles
        ranges = []
        for i in range(self.num_beams):
            angle = 2 * math.pi * i / self.num_beams

            # Simulate distance to nearest obstacle
            # In a real implementation, this would ray-cast into the 3D scene
            distance = self.max_range  # Default to max range (free space)

            # Add some simulated obstacles
            if 0.3 < (angle % (2*math.pi)) < 0.4:
                distance = 2.0  # Obstacle at 2m
            elif 1.5 < (angle % (2*math.pi)) < 1.6:
                distance = 3.5  # Obstacle at 3.5m

            # Add noise
            noisy_distance = distance + np.random.normal(0, self.noise_std)
            ranges.append(max(0.1, min(self.max_range, noisy_distance)))  # Clamp to valid range

        return ranges

class SimulatedIMU:
    def __init__(self):
        self.bias_acc = np.array([0.01, -0.02, 0.005])  # Bias in m/s^2
        self.bias_gyro = np.array([0.001, -0.002, 0.0005])  # Bias in rad/s
        self.noise_acc = 0.001  # Noise std for accelerometer
        self.noise_gyro = 0.0001  # Noise std for gyroscope

    def simulate_reading(self):
        """Simulate IMU reading"""
        # Simulate gravity vector (in sensor frame)
        gravity = np.array([0, 0, 9.81])

        # Simulate linear acceleration (zero for stationary robot)
        linear_acc = np.array([0, 0, 0])

        # Simulate angular velocity (zero for non-rotating robot)
        angular_vel = np.array([0, 0, 0])

        # Combine with bias and noise
        measured_acc = gravity + linear_acc + self.bias_acc + \
                      np.random.normal(0, self.noise_acc, 3)
        measured_gyro = angular_vel + self.bias_gyro + \
                       np.random.normal(0, self.noise_gyro, 3)

        return {
            'acceleration': measured_acc,
            'angular_velocity': measured_gyro
        }

# Example usage
perception_system = MultiSensorPerception()
scene_data = {}  # Placeholder
robot_pose = [0.0, 0.0, 0.0]  # x, y, theta
timestamp = 0.0

detected_objects = perception_system.run_perception_pipeline(
    scene_data, robot_pose, timestamp)

print(f"Detected {len(detected_objects)} objects:")
for i, obj in enumerate(detected_objects):
    print(f"  Object {i+1}: {obj['type']} at {obj['position']}, "
          f"confidence: {obj['confidence']:.2f}")
```

### Example 3: Perception Pipeline with State Estimation

A complete perception system that maintains state over time:

```python
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Dict, Any
import time

class TrackingPerceptionSystem:
    def __init__(self):
        self.objects = []  # List of tracked objects
        self.next_id = 0
        self.association_threshold = 1.0  # Maximum distance for association
        self.max_inactive_frames = 10  # Remove tracks after this many frames without detection

        # State covariance for Kalman filter
        self.process_noise = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # [x, y, z, vx, vy, vz]
        self.measurement_noise = np.diag([0.05, 0.05, 0.05])  # [x, y, z]

    def predict_objects(self, dt):
        """Predict object states forward in time"""
        for obj in self.objects:
            if obj['active']:
                # Simple constant velocity model
                obj['state'][0] += obj['state'][3] * dt  # x = x + vx*dt
                obj['state'][1] += obj['state'][4] * dt  # y = y + vy*dt
                obj['state'][2] += obj['state'][5] * dt  # z = z + vz*dt

                # Predict covariance (simplified)
                obj['covariance'] += self.process_noise * dt

    def associate_detections(self, detections):
        """Associate new detections with existing tracks"""
        if not self.objects:
            # No existing tracks, create new ones
            for detection in detections:
                self.objects.append(self.create_new_track(detection))
            return

        # Calculate distances between detections and existing tracks
        if detections:
            track_states = np.array([obj['state'][:3] for obj in self.objects if obj['active']])  # x, y, z only
            detection_positions = np.array([det['position'] for det in detections])

            # Calculate distance matrix
            distances = cdist(track_states, detection_positions)

            # Assign detections to tracks (greedy assignment)
            assigned_detections = set()
            assigned_tracks = set()

            # Sort assignments by distance
            assignments = []
            for i in range(len(self.objects)):
                if not self.objects[i]['active']:
                    continue
                for j in range(len(detections)):
                    assignments.append((distances[i, j], i, j))

            assignments.sort(key=lambda x: x[0])

            for dist, track_idx, det_idx in assignments:
                if (dist < self.association_threshold and
                    track_idx not in assigned_tracks and
                    det_idx not in assigned_detections):

                    # Update existing track with new detection
                    self.update_track(track_idx, detections[det_idx])
                    assigned_tracks.add(track_idx)
                    assigned_detections.add(det_idx)

            # Create new tracks for unassigned detections
            for det_idx, detection in enumerate(detections):
                if det_idx not in assigned_detections:
                    self.objects.append(self.create_new_track(detection))

        # Update active/inactive status
        for i, obj in enumerate(self.objects):
            if i not in assigned_tracks:
                obj['inactive_frames'] += 1
                if obj['inactive_frames'] > self.max_inactive_frames:
                    obj['active'] = False  # Mark for removal
            else:
                obj['inactive_frames'] = 0

    def create_new_track(self, detection):
        """Create a new object track"""
        track = {
            'id': self.next_id,
            'state': np.array([detection['position'][0], detection['position'][1], detection['position'][2],
                              0.0, 0.0, 0.0]),  # [x, y, z, vx, vy, vz]
            'covariance': self.measurement_noise.copy(),
            'type': detection['type'],
            'confidence': detection['confidence'],
            'active': True,
            'inactive_frames': 0,
            'first_seen': time.time(),
            'last_seen': time.time()
        }
        self.next_id += 1
        return track

    def update_track(self, track_idx, detection):
        """Update an existing track with new detection"""
        track = self.objects[track_idx]

        # Simple update (in practice, use Kalman filter)
        # For simplicity, we'll just update position and reset velocity
        H = np.array([[1, 0, 0, 0, 0, 0],   # Measure x
                      [0, 1, 0, 0, 0, 0],   # Measure y
                      [0, 0, 1, 0, 0, 0]])  # Measure z

        # Innovation
        z = np.array(detection['position'])
        x_pred = track['state']
        innovation = z - H @ x_pred

        # Simplified update (proper Kalman filter would be more complex)
        track['state'][0:3] = z  # Update position
        track['state'][3:6] = 0.0  # Reset velocity to 0

        # Update confidence
        track['confidence'] = max(track['confidence'], detection['confidence'])
        track['last_seen'] = time.time()

    def remove_inactive_tracks(self):
        """Remove tracks that have been inactive too long"""
        self.objects = [obj for obj in self.objects if obj['active']]

    def get_tracked_objects(self):
        """Get currently tracked objects"""
        return [obj for obj in self.objects if obj['active']]

    def process_frame(self, detections, dt=0.1):
        """Process a single frame of detections"""
        # Predict object states
        self.predict_objects(dt)

        # Associate new detections
        self.associate_detections(detections)

        # Remove inactive tracks
        self.remove_inactive_tracks()

        return self.get_tracked_objects()

# Example usage
tracking_system = TrackingPerceptionSystem()

# Simulate a sequence of detections over time
for frame in range(10):
    # Simulate detections (in practice, these would come from perception algorithms)
    detections = []

    # Simulate a moving object
    moving_obj_x = 1.0 + frame * 0.1  # Moving in x direction
    moving_obj_y = 0.5 + frame * 0.05  # Moving in y direction

    detections.append({
        'position': [moving_obj_x, moving_obj_y, 0.5],
        'type': 'obstacle',
        'confidence': 0.8
    })

    # Add some noise to simulation
    if np.random.random() > 0.3:  # Sometimes add additional detections
        detections.append({
            'position': [2.0 + np.random.normal(0, 0.1),
                        1.0 + np.random.normal(0, 0.1),
                        0.5 + np.random.normal(0, 0.1)],
            'type': 'obstacle',
            'confidence': 0.6
        })

    # Process the frame
    tracked_objects = tracking_system.process_frame(detections, dt=0.1)

    print(f"Frame {frame+1}: {len(tracked_objects)} tracked objects")
    for obj in tracked_objects:
        print(f"  Object {obj['id']}: pos=({obj['state'][0]:.2f}, {obj['state'][1]:.2f}, {obj['state'][2]:.2f}), "
              f"type={obj['type']}, confidence={obj['confidence']:.2f}")

print(f"\nFinal tracked objects: {len(tracking_system.get_tracked_objects())}")
```

## System Integration Perspective

Effective simulated sensor and perception systems require integration across multiple system components:

**Sensor Calibration**: Simulated sensors should match the characteristics of real sensors:
- Intrinsic and extrinsic calibration parameters
- Noise characteristics and systematic errors
- Temporal synchronization between sensors

**Computational Efficiency**: Perception systems must run efficiently in simulation:
- Optimized algorithms for real-time performance
- Level-of-detail approaches for complex scenes
- Parallel processing capabilities

**Ground Truth Integration**: Simulation provides access to ground truth information:
- Perfect knowledge of object states for validation
- Ability to test perception in controlled conditions
- Quantification of perception accuracy

**Transfer Learning**: Ensuring algorithms work across simulation and reality:
- Domain randomization techniques
- Sim-to-real transfer methods
- Validation protocols

**System Validation**: Comprehensive testing of perception systems:
- Unit testing of individual components
- Integration testing with full robot systems
- Validation against real-world data

## Summary

- Simulated sensors must balance realism with computational performance
- Multi-sensor fusion combines data from different modalities for robust perception
- Tracking systems maintain object state over time for consistent perception
- Proper validation ensures simulation results transfer to reality
- System integration requires careful calibration and synchronization

## Exercises

1. **Sensor Modeling**: Design a realistic model for a stereo camera system including rectification, disparity computation, and depth estimation. How would you model the various sources of error?

2. **Multi-Sensor Fusion**: Implement a fusion algorithm that combines camera, LIDAR, and IMU data for object detection and tracking. What are the advantages of each sensor modality?

3. **Perception Pipeline**: Design a complete perception pipeline for a mobile robot that includes detection, tracking, and scene understanding. How would you handle different environmental conditions?

4. **Validation Strategy**: Develop a validation plan to ensure your simulated perception system behaves like a real system. What metrics would you use?

5. **Real-time Performance**: How would you optimize a perception system to run in real-time during simulation? What trade-offs would you consider?
