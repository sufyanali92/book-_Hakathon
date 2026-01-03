---
title: "Robot Communication with Nodes, Topics, and Messages"
sidebar_position: 2
---

# 2. Robot Communication with Nodes, Topics, and Messages

## Introduction

Communication is the backbone of any robotic system, enabling different components to coordinate and share information. In ROS 2, communication is structured around a distributed architecture where independent nodes communicate through topics, services, and actions. This chapter explores the fundamental communication patterns in ROS 2, focusing on nodes, topics, and messages, which form the foundation of robot communication.

## Learning Objectives

- Understand the node-based architecture of ROS 2
- Explain the publish-subscribe communication pattern
- Identify the structure and types of messages in ROS 2
- Design effective communication architectures for robotic systems
- Recognize best practices for message design and communication

## Conceptual Foundations

ROS 2 communication is built on a distributed architecture where:

**Nodes** are independent processes that perform computation. Each node can publish information, subscribe to information, provide services, or call services. Nodes are the basic computational units in ROS 2.

**Topics** are named buses over which nodes exchange messages. Topics implement a publish-subscribe communication pattern where publishers send messages and subscribers receive them.

**Messages** are the data structures sent between nodes. They are defined in special interface definition files (.msg) and are strongly typed.

**Publish-Subscribe Pattern** allows for asynchronous, decoupled communication where publishers and subscribers don't need to know about each other directly.

**Anonymous Communication** nodes can communicate without knowing each other's identity, promoting modularity and flexibility.

## Technical Explanation

### Node Architecture

Nodes in ROS 2 are implemented as objects that inherit from the Node class:

- **Node Naming**: Each node has a unique name within the ROS graph
- **Namespace Support**: Nodes can be organized into namespaces for better organization
- **Parameter System**: Nodes can have parameters that can be configured at runtime
- **Logging Interface**: Standardized logging system for debugging and monitoring
- **Timer Support**: Built-in timer functionality for periodic tasks
- **Callback Groups**: Mechanism to control how callbacks are executed

### Topic Communication

Topics implement a publish-subscribe pattern:

- **Publishers**: Send messages to a topic
- **Subscribers**: Receive messages from a topic
- **Message Types**: Each topic has a specific message type that all publishers and subscribers must use
- **Discovery**: Nodes automatically discover publishers and subscribers through DDS discovery mechanisms
- **Synchronization**: Publishers and subscribers can have different lifetimes

### Message Structure

Messages in ROS 2 are defined using the .msg file format:

- **Primitive Types**: bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, string, byte
- **Arrays**: Fixed-size and variable-size arrays of primitive types
- **Nested Messages**: Messages can contain other message types
- **Constants**: Named constants can be defined in message files
- **Timestamps**: Common message types include header fields with timestamps

### Quality of Service (QoS)

QoS policies control communication behavior:

- **Reliability**: Ensures all messages are delivered (RELIABLE) or best-effort delivery (BEST_EFFORT)
- **Durability**: Controls whether late-joining subscribers receive historical data
- **History**: Determines how many messages are kept in the publisher's queue
- **Depth**: Maximum number of messages to store in the queue

## Practical Examples

### Example 1: Basic Node with Publisher and Subscriber

Creating a node that both publishes and subscribes to topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import LaserScan
import time

class CommunicationNode(Node):
    def __init__(self):
        super().__init__('communication_node')

        # Create a publisher for a string topic
        self.publisher_ = self.create_publisher(String, 'robot_status', 10)

        # Create a publisher for an integer topic
        self.count_publisher_ = self.create_publisher(Int32, 'counter', 10)

        # Create a subscriber for laser scan data
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10)

        # Create a timer to periodically publish status
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        # Publish robot status
        msg = String()
        msg.data = f'Robot is operational - Count: {self.i}'
        self.publisher_.publish(msg)

        # Publish counter value
        count_msg = Int32()
        count_msg.data = self.i
        self.count_publisher_.publish(count_msg)

        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

    def laser_callback(self, msg):
        # Process laser scan data
        if len(msg.ranges) > 0:
            min_distance = min([r for r in msg.ranges if r > 0])
            self.get_logger().info(f'Min obstacle distance: {min_distance:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    communication_node = CommunicationNode()

    try:
        rclpy.spin(communication_node)
    except KeyboardInterrupt:
        pass
    finally:
        communication_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 2: Custom Message Definition

Creating and using custom message types:

```python
# Custom message definition (saved as msg/RobotPose.msg)
# This would typically be in a separate package
"""
# RobotPose.msg
# Represents the pose of a robot in 3D space

# Standard header
std_msgs/Header header

# Position
float64 x
float64 y
float64 z

# Orientation (quaternion)
float64 qx
float64 qy
float64 qz
float64 qw

# Robot-specific information
string robot_name
uint8[] joint_angles  # Joint angles in degrees
"""

# Node using the custom message
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from your_package.msg import RobotPose  # Custom message

class RobotPosePublisher(Node):
    def __init__(self):
        super().__init__('robot_pose_publisher')

        # Publisher for robot pose
        self.pose_publisher = self.create_publisher(RobotPose, 'robot_pose', 10)

        # Timer to publish pose periodically
        self.timer = self.create_timer(0.1, self.publish_pose)  # 10Hz
        self.pose_counter = 0

    def publish_pose(self):
        msg = RobotPose()

        # Fill header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # Fill pose information
        msg.x = 1.0 + 0.1 * self.pose_counter
        msg.y = 2.0 + 0.1 * self.pose_counter
        msg.z = 0.0  # Assume 2D navigation

        # Simple orientation (facing along x-axis)
        msg.qx = 0.0
        msg.qy = 0.0
        msg.qz = 0.0
        msg.qw = 1.0

        # Robot name
        msg.robot_name = 'turtlebot3'

        # Simulate joint angles (4 joints)
        msg.joint_angles = [90, 45, 0, -45]  # In degrees

        self.pose_publisher.publish(msg)
        self.get_logger().info(f'Published pose: ({msg.x:.2f}, {msg.y:.2f})')
        self.pose_counter += 1

class RobotPoseSubscriber(Node):
    def __init__(self):
        super().__init__('robot_pose_subscriber')

        # Subscriber for robot pose
        self.pose_subscriber = self.create_subscription(
            RobotPose,
            'robot_pose',
            self.pose_callback,
            10)

    def pose_callback(self, msg):
        self.get_logger().info(
            f'Received pose for {msg.robot_name}: '
            f'({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})'
        )
        self.get_logger().info(f'Joint angles: {list(msg.joint_angles)}')
```

### Example 3: Multiple Publishers and Subscribers Pattern

A more complex communication pattern with multiple publishers and subscribers:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import threading
import time

class MultiSensorNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_node')

        # Publishers for different data types
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'robot_pose', 10)

        # Subscribers for sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        # Internal state
        self.latest_scan = None
        self.latest_imu = None
        self.latest_odom = None
        self.safety_engaged = False

        # Timers for different rates
        self.sensor_timer = self.create_timer(0.1, self.process_sensors)  # 10Hz
        self.control_timer = self.create_timer(0.05, self.control_loop)   # 20Hz

    def laser_callback(self, msg):
        self.latest_scan = msg
        # Check for obstacles
        if self.has_obstacle_ahead(msg):
            self.safety_engaged = True
        else:
            self.safety_engaged = False

    def imu_callback(self, msg):
        self.latest_imu = msg

    def odom_callback(self, msg):
        self.latest_odom = msg

    def has_obstacle_ahead(self, scan_msg):
        """Check if there's an obstacle in front of the robot"""
        if scan_msg.ranges:
            # Check the front 30 degrees
            front_ranges = scan_msg.ranges[:len(scan_msg.ranges)//12] + \
                          scan_msg.ranges[-len(scan_msg.ranges)//12:]
            valid_ranges = [r for r in front_ranges if r > scan_msg.range_min and r < scan_msg.range_max]
            if valid_ranges and min(valid_ranges) < 0.5:  # 0.5m threshold
                return True
        return False

    def process_sensors(self):
        """Process sensor data and update internal state"""
        if self.latest_scan:
            # Process laser data for navigation
            pass

        if self.latest_imu:
            # Process IMU data for orientation
            pass

    def control_loop(self):
        """Main control loop"""
        cmd = Twist()

        if self.safety_engaged:
            # Emergency stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Normal navigation
            cmd.linear.x = 0.2  # Move forward at 0.2 m/s
            cmd.angular.z = 0.0  # No rotation

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    multi_sensor_node = MultiSensorNode()

    try:
        rclpy.spin(multi_sensor_node)
    except KeyboardInterrupt:
        pass
    finally:
        multi_sensor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## System Integration Perspective

Effective robot communication requires consideration of several system-level factors:

**Message Design**: Messages should be designed for efficiency and clarity:
- Keep messages appropriately sized (not too large for network transmission)
- Include only necessary data
- Use appropriate data types for the information being conveyed
- Consider frequency of message transmission

**Communication Topology**: The pattern of communication between nodes affects system performance:
- Minimize unnecessary data transmission
- Consider the frequency of message publication
- Use appropriate QoS settings for different types of data
- Design for fault tolerance and graceful degradation

**Real-time Considerations**: Communication patterns must support real-time requirements:
- Critical messages should use reliable QoS
- Timing-critical data should have appropriate deadlines
- System should handle communication delays gracefully

**Security**: Communication channels may need security measures:
- Authentication of nodes
- Encryption of sensitive data
- Access control to topics

**Scalability**: The communication architecture should scale with system complexity:
- Namespacing for multiple robots or subsystems
- Efficient discovery mechanisms
- Load balancing for high-frequency topics

## Summary

- Nodes are independent processes that communicate through topics
- Topics use publish-subscribe pattern for asynchronous communication
- Messages are strongly-typed data structures defined in .msg files
- QoS policies control communication behavior and reliability
- Effective communication design is critical for robot system performance

## Exercises

1. **Message Design**: Design a message type for a robot that needs to share its battery status, current location, and operational mode. What fields would you include and why?

2. **Communication Topology**: For a multi-robot system with 5 robots, design a communication topology that allows each robot to share its location with all others. How would you organize the topics?

3. **QoS Configuration**: For a robot's sensor data (camera, LIDAR, IMU), control commands, and logging information, design appropriate QoS policies for each type of data.

4. **Node Architecture**: Design a node architecture for a robot that performs mapping, localization, and navigation. Identify the nodes, their roles, and the topics they would use.

5. **Real-time Performance**: How would you design the communication system for a robot that needs to maintain a 200Hz control loop? What considerations would you make for message frequency and QoS?
