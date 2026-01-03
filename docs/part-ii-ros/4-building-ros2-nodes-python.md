---
title: "Building ROS 2 Nodes with Python"
sidebar_position: 4
---

# 4. Building ROS 2 Nodes with Python

## Introduction

Python is one of the most popular languages for robotics development due to its simplicity, extensive libraries, and strong community support. ROS 2 provides comprehensive Python support through the rclpy client library, enabling developers to create robust robotic applications. This chapter covers the fundamentals of building ROS 2 nodes using Python, including node structure, communication patterns, and best practices for production systems.

## Learning Objectives

- Understand the structure and lifecycle of ROS 2 Python nodes
- Implement publishers, subscribers, services, and actions in Python
- Apply best practices for Python node development
- Handle errors and exceptions in ROS 2 Python nodes
- Design efficient and maintainable node architectures

## Conceptual Foundations

Building ROS 2 nodes in Python involves several key concepts:

**Node Structure**: A ROS 2 Python node is a class that inherits from `rclpy.node.Node`. The node serves as a container for publishers, subscribers, services, and other ROS entities.

**Lifecycle Management**: Nodes have a lifecycle that includes initialization, spinning (processing callbacks), and cleanup. Proper resource management is essential for robust nodes.

**Threading and Concurrency**: ROS 2 provides different executor types to handle concurrent callbacks and multi-threading scenarios.

**Client Library Integration**: The rclpy library provides Python bindings to the ROS 2 client library, handling low-level communication details.

**Parameter System**: Nodes can declare and use parameters for configuration, with automatic type checking and validation.

## Technical Explanation

### Node Creation and Structure

A basic ROS 2 Python node follows this structure:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Initialize publishers, subscribers, etc.

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Publishers and Subscribers

Publishers and subscribers are created within the node:

```python
# Creating a publisher
publisher = self.create_publisher(MessageType, 'topic_name', qos_profile)

# Creating a subscriber
subscriber = self.create_subscription(
    MessageType, 'topic_name', callback_function, qos_profile)
```

### Executors

ROS 2 provides different executors for handling callbacks:

- **SingleThreadedExecutor**: Processes callbacks sequentially in a single thread
- **MultiThreadedExecutor**: Processes callbacks in multiple threads
- **Custom Executors**: For specialized threading requirements

### Parameter Declaration

Parameters are declared to enable runtime configuration:

```python
self.declare_parameter('param_name', default_value)
value = self.get_parameter('param_name').value
```

## Practical Examples

### Example 1: Comprehensive Node with Multiple Communication Patterns

A node that demonstrates publishers, subscribers, services, and parameters:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Int32, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from your_package.srv import SetMode  # Custom service
import math
import threading

class ComprehensiveRobotNode(Node):
    def __init__(self):
        super().__init__('comprehensive_robot_node')

        # Declare parameters with descriptions
        self.declare_parameter('linear_speed', 0.5,
            'Linear speed for robot movement')
        self.declare_parameter('angular_speed', 0.5,
            'Angular speed for robot rotation')
        self.declare_parameter('safety_distance', 0.5,
            'Minimum distance to obstacles before stopping')
        self.declare_parameter('control_frequency', 10,
            'Frequency of control loop in Hz')

        # Initialize parameters
        self.update_parameters()

        # Create QoS profiles
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        cmd_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', cmd_qos)
        self.status_pub = self.create_publisher(String, 'robot_status', 10)
        self.battery_pub = self.create_publisher(Float32, 'battery_level', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, sensor_qos)
        self.goal_sub = self.create_subscription(
            Twist, 'navigation_goal', self.goal_callback, 10)

        # Service server
        self.mode_service = self.create_service(
            SetMode, 'set_robot_mode', self.mode_callback)

        # Internal state
        self.latest_scan = None
        self.navigation_goal = None
        self.current_mode = 'idle'
        self.safety_engaged = False
        self.battery_level = 100.0

        # Timer for control loop
        control_period = 1.0 / self.get_parameter('control_frequency').value
        self.control_timer = self.create_timer(control_period, self.control_loop)

        # Parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info('Comprehensive Robot Node initialized')

    def update_parameters(self):
        """Update internal values from parameters"""
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.control_frequency = self.get_parameter('control_frequency').value

    def parameter_callback(self, params):
        """Handle parameter changes"""
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')

        self.update_parameters()
        return SetParametersResult(successful=True)

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.latest_scan = msg
        # Check for obstacles
        if self.has_obstacle_ahead(msg):
            self.safety_engaged = True
            self.get_logger().warn('Safety engaged: obstacle detected ahead')
        else:
            self.safety_engaged = False

    def has_obstacle_ahead(self, scan_msg):
        """Check if there's an obstacle in front of the robot"""
        if not scan_msg.ranges:
            return False

        # Check the front 30 degrees
        mid_idx = len(scan_msg.ranges) // 2
        front_range = scan_msg.ranges[mid_idx - 15:mid_idx + 15]
        valid_ranges = [r for r in front_range if
                       scan_msg.range_min < r < scan_msg.range_max]

        return valid_ranges and min(valid_ranges) < self.safety_distance

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.navigation_goal = msg
        self.get_logger().info(f'Received navigation goal: '
                              f'linear={msg.linear.x}, angular={msg.angular.z}')

    def mode_callback(self, request, response):
        """Handle mode change request"""
        valid_modes = ['idle', 'navigation', 'manual', 'emergency']

        if request.mode in valid_modes:
            old_mode = self.current_mode
            self.current_mode = request.mode

            response.success = True
            response.message = f'Mode changed from {old_mode} to {request.mode}'

            status_msg = String()
            status_msg.data = f'Mode: {self.current_mode}'
            self.status_pub.publish(status_msg)
        else:
            response.success = False
            response.message = f'Invalid mode: {request.mode}. Valid: {valid_modes}'

        return response

    def control_loop(self):
        """Main control loop"""
        cmd = Twist()

        if self.safety_engaged:
            # Emergency stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().warn('Safety stop activated')
        elif self.current_mode == 'navigation' and self.navigation_goal:
            # Follow navigation goal
            cmd.linear.x = self.navigation_goal.linear.x * self.linear_speed
            cmd.angular.z = self.navigation_goal.angular.z * self.angular_speed
        elif self.current_mode == 'manual':
            # Manual control - could receive from joystick
            cmd.linear.x = 0.0  # Placeholder
            cmd.angular.z = 0.0
        else:
            # Idle mode
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Apply safety limits
        cmd.linear.x = max(-self.linear_speed, min(self.linear_speed, cmd.linear.x))
        cmd.angular.z = max(-self.angular_speed, min(self.angular_speed, cmd.angular.z))

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Simulate battery drain
        self.battery_level = max(0.0, self.battery_level - 0.001)
        battery_msg = Float32()
        battery_msg.data = self.battery_level
        self.battery_pub.publish(battery_msg)

        # Log status periodically
        if self.get_clock().now().nanoseconds % 1000000000 < 10000000:  # Every ~1 second
            self.get_logger().info(
                f'Mode: {self.current_mode}, Battery: {self.battery_level:.1f}%')

def main(args=None):
    rclpy.init(args=args)

    try:
        node = ComprehensiveRobotNode()

        # Use MultiThreadedExecutor to handle multiple callbacks
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('Keyboard interrupt received')
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 2: Advanced Node with Actions and Error Handling

A node implementing actions with proper error handling and state management:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from your_package.action import NavigateToPose  # Custom action
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import threading
import time
import math

class NavigationActionNode(Node):
    def __init__(self):
        super().__init__('navigation_action_node')

        # Create action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=MutuallyExclusiveCallbackGroup())

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.status_pub = self.create_publisher(String, 'navigation_status', 10)

        # Internal state
        self.current_pose = None
        self.active_goal = None
        self._goal_lock = threading.Lock()
        self.navigation_active = False

        # Parameters
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('angular_tolerance', 0.1)

        self.get_logger().info('Navigation Action Node initialized')

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def goal_callback(self, goal_request):
        """Accept or reject navigation goal"""
        try:
            # Check if we can accept the goal
            if self.navigation_active:
                self.get_logger().warn('Navigation already active, rejecting new goal')
                return GoalResponse.REJECT

            # Validate goal
            target = goal_request.target_pose.pose
            if not self.is_valid_pose(target):
                self.get_logger().error('Invalid navigation goal')
                return GoalResponse.REJECT

            self.get_logger().info(f'Accepting navigation goal: ({target.position.x:.2f}, {target.position.y:.2f})')
            return GoalResponse.ACCEPT
        except Exception as e:
            self.get_logger().error(f'Error in goal callback: {str(e)}')
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def is_valid_pose(self, pose):
        """Check if pose is valid for navigation"""
        # Check if pose is finite (not NaN or infinity)
        return (math.isfinite(pose.position.x) and math.isfinite(pose.position.y) and
                math.isfinite(pose.orientation.w))

    def distance_2d(self, pose1, pose2):
        """Calculate 2D distance between two poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx*dx + dy*dy)

    def angle_to_target(self, current_pose, target_pose):
        """Calculate angle from current pose to target pose"""
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        target_angle = math.atan2(dy, dx)

        # Extract yaw from quaternion
        q = current_pose.orientation
        current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        return math.atan2(math.sin(target_angle - current_yaw),
                         math.cos(target_angle - current_yaw))

    async def execute_callback(self, goal_handle):
        """Execute navigation goal"""
        self.get_logger().info('Starting navigation execution')

        # Set active goal and update state
        with self._goal_lock:
            self.active_goal = goal_handle
            self.navigation_active = True

        # Initialize result
        result = NavigateToPose.Result()
        result.success = False
        result.message = "Navigation failed"

        try:
            goal = goal_handle.request.target_pose.pose
            tolerance = goal_handle.request.tolerance

            # Initialize feedback
            feedback_msg = NavigateToPose.Feedback()
            start_pose = self.current_pose if self.current_pose else goal

            while goal_handle.is_active:
                # Check for cancellation
                if goal_handle.is_canceling():
                    self.get_logger().info('Navigation canceled')
                    cmd = Twist()
                    self.cmd_pub.publish(cmd)  # Stop robot
                    result.success = False
                    result.message = "Navigation canceled"
                    goal_handle.canceled()
                    break

                # Check if we have current pose
                if not self.current_pose:
                    self.get_logger().warn('No current pose available')
                    await rclpy.asyncio.sleep_until_future_complete(self, rclpy.create_timer(0.1))
                    continue

                # Calculate navigation commands
                distance = self.distance_2d(self.current_pose, goal)
                angle_to_target = self.angle_to_target(self.current_pose, goal)

                # Create command
                cmd = Twist()

                # If we're close to target, just rotate to correct orientation
                if distance > tolerance:
                    # Move toward target
                    cmd.linear.x = min(self.get_parameter('linear_speed').value,
                                      distance * 0.5)  # Proportional control
                    cmd.angular.z = max(-self.get_parameter('angular_speed').value,
                                       min(self.get_parameter('angular_speed').value,
                                          angle_to_target * 2.0))
                else:
                    # Reached position, now orient correctly
                    target_yaw = math.atan2(
                        goal.orientation.z * goal.orientation.w * 2,
                        goal.orientation.w**2 - goal.orientation.z**2
                    )
                    current_yaw = math.atan2(
                        2.0 * (self.current_pose.orientation.w * self.current_pose.orientation.z +
                               self.current_pose.orientation.x * self.current_pose.orientation.y),
                        1.0 - 2.0 * (self.current_pose.orientation.y**2 +
                                     self.current_pose.orientation.z**2))

                    angle_diff = math.atan2(
                        math.sin(target_yaw - current_yaw),
                        math.cos(target_yaw - current_yaw))

                    cmd.linear.x = 0.0
                    cmd.angular.z = max(-self.get_parameter('angular_speed').value,
                                       min(self.get_parameter('angular_speed').value,
                                          angle_diff * 2.0))

                # Publish command
                self.cmd_pub.publish(cmd)

                # Update feedback
                feedback_msg.current_pose.pose = self.current_pose
                feedback_msg.distance_remaining = distance
                feedback_msg.progress_percentage = max(0.0, min(100.0,
                    (1.0 - distance/self.distance_2d(start_pose, goal)) * 100.0))

                goal_handle.publish_feedback(feedback_msg)

                # Check if we've reached the goal
                if distance <= tolerance and abs(angle_to_target) <= self.get_parameter('angular_tolerance').value:
                    result.success = True
                    result.message = "Navigation successful"
                    result.final_pose.pose = self.current_pose
                    result.distance_traveled = self.distance_2d(start_pose, self.current_pose)

                    # Stop the robot
                    cmd = Twist()
                    self.cmd_pub.publish(cmd)

                    goal_handle.succeed()
                    break

                # Sleep briefly to avoid busy waiting
                await rclpy.asyncio.sleep_until_future_complete(self, rclpy.create_timer(0.05))

        except Exception as e:
            self.get_logger().error(f'Error during navigation: {str(e)}')
            result.success = False
            result.message = f"Navigation failed: {str(e)}"
        finally:
            # Clean up
            with self._goal_lock:
                self.active_goal = None
                self.navigation_active = False

        return result

def main(args=None):
    rclpy.init(args=args)

    try:
        node = NavigationActionNode()

        # Use MultiThreadedExecutor for better performance
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('Keyboard interrupt received')
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 3: Node with Lifecycle Management and Best Practices

A well-structured node following ROS 2 best practices:

```python
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
import cv2
import numpy as np
from typing import Optional

class LifecycleCameraNode(LifecycleNode):
    """
    A lifecycle camera node that demonstrates best practices for
    resource management and state transitions in ROS 2.
    """

    def __init__(self, node_name: str = 'lifecycle_camera_node'):
        super().__init__(node_name)

        # Store configuration
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        # Initialize internal variables (but not resources)
        self.camera: Optional[cv2.VideoCapture] = None
        self.publisher = None
        self.status_publisher = None
        self.capture_timer = None

        self.get_logger().info('Lifecycle Camera Node created')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when configuring the node"""
        self.get_logger().info(f'Configuring {self.get_name()}')

        # Get parameters
        self.camera_id = self.get_parameter('camera_id').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value

        # Create publishers (but don't start camera yet)
        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.publisher = self.create_publisher(Image, 'camera/image_raw', qos_profile)
        self.status_publisher = self.create_publisher(String, 'camera/status', 10)

        # Publish status
        status_msg = String()
        status_msg.data = 'CONFIGURED'
        self.status_publisher.publish(status_msg)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when activating the node"""
        self.get_logger().info(f'Activating {self.get_name()}')

        # Initialize camera
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                self.get_logger().error(f'Failed to open camera {self.camera_id}')
                return TransitionCallbackReturn.FAILURE

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)

            # Create timer for capturing images
            timer_period = 1.0 / self.frame_rate
            self.capture_timer = self.create_timer(timer_period, self.capture_callback)

            # Publish status
            status_msg = String()
            status_msg.data = 'ACTIVE'
            self.status_publisher.publish(status_msg)

            self.get_logger().info(f'Camera {self.camera_id} activated successfully')

        except Exception as e:
            self.get_logger().error(f'Error activating camera: {str(e)}')
            return TransitionCallbackReturn.FAILURE

        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when deactivating the node"""
        self.get_logger().info(f'Deactivating {self.get_name()}')

        # Stop timer
        if self.capture_timer:
            self.capture_timer.cancel()
            self.capture_timer = None

        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None

        # Publish status
        status_msg = String()
        status_msg.data = 'INACTIVE'
        self.status_publisher.publish(status_msg)

        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when cleaning up the node"""
        self.get_logger().info(f'Cleaning up {self.get_name()}')

        # Destroy publishers
        if self.publisher:
            self.destroy_publisher(self.publisher)
            self.publisher = None
        if self.status_publisher:
            self.destroy_publisher(self.status_publisher)
            self.status_publisher = None

        # Publish status
        status_msg = String()
        status_msg.data = 'CLEANED_UP'
        self.status_publisher.publish(status_msg) if self.status_publisher else None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when shutting down the node"""
        self.get_logger().info(f'Shutting down {self.get_name()}')

        # Release any remaining resources
        if self.camera:
            self.camera.release()
            self.camera = None

        if self.capture_timer:
            self.capture_timer.cancel()
            self.capture_timer = None

        return TransitionCallbackReturn.SUCCESS

    def capture_callback(self):
        """Capture and publish camera image"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()

            if ret:
                # Convert OpenCV image to ROS Image message
                # (Simplified conversion - in practice, use cv_bridge)
                image_msg = Image()
                image_msg.header.stamp = self.get_clock().now().to_msg()
                image_msg.header.frame_id = 'camera_frame'
                image_msg.height = frame.shape[0]
                image_msg.width = frame.shape[1]
                image_msg.encoding = 'bgr8'
                image_msg.is_bigendian = False
                image_msg.step = frame.shape[1] * 3  # 3 channels
                image_msg.data = frame.tobytes()

                self.publisher.publish(image_msg)
            else:
                self.get_logger().warn('Failed to capture image from camera')

def main(args=None):
    rclpy.init(args=args)

    try:
        node = LifecycleCameraNode()

        # For lifecycle nodes, you typically want to use the lifecycle node
        # manager or manually transition through states
        # For simplicity, we'll just spin it normally here
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## System Integration Perspective

Building robust ROS 2 Python nodes requires attention to several system-level considerations:

**Resource Management**: Properly managing resources like file handles, network connections, and memory is crucial for stable systems. Always implement proper cleanup in the `destroy_node()` method.

**Error Handling**: Robust error handling prevents nodes from crashing and enables graceful degradation. Use try-catch blocks for operations that might fail.

**Threading Considerations**: Understanding when and how to use multiple threads is important for performance. The MultiThreadedExecutor can improve responsiveness but requires careful handling of shared state.

**Logging and Debugging**: Comprehensive logging helps with debugging and monitoring. Use appropriate log levels (info, warn, error) and include relevant context in log messages.

**Performance Optimization**: For real-time systems, consider the performance implications of Python. Use efficient data structures and algorithms, and consider using C++ for performance-critical components.

**Testing and Validation**: Unit tests and integration tests are essential for ensuring node reliability. ROS 2 provides testing frameworks specifically designed for robotic systems.

## Summary

- Python nodes follow a class-based structure inheriting from rclpy.node.Node
- Proper resource management and error handling are essential for robust nodes
- Executors control how callbacks are processed in multi-threaded environments
- Lifecycle nodes provide better resource management for complex systems
- Following best practices ensures maintainable and reliable robotic systems

## Exercises

1. **Node Design**: Design a Python node that reads sensor data from multiple sources (IMU, camera, LIDAR) and publishes a fused state estimate. What communication patterns would you use and why?

2. **Error Handling**: Modify the comprehensive robot node example to include proper error handling for sensor failures, communication timeouts, and invalid parameter values.

3. **Performance Optimization**: For a real-time control node that needs to run at 200Hz, identify potential performance bottlenecks in a Python implementation and suggest optimizations.

4. **Lifecycle Management**: Design a lifecycle node for a robotic arm that has multiple states (idle, calibrated, ready, moving). What transitions would you implement and what resources would be managed in each state?

5. **System Architecture**: Design a system architecture using multiple Python nodes for a mobile robot that performs autonomous navigation. Identify the nodes, their responsibilities, and the communication patterns between them.
