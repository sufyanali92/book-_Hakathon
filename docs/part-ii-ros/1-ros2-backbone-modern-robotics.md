---
title:  "ROS 2: The Backbone of Modern Robotics"
---

# 1. ROS 2: The Backbone of Modern Robotics

## Introduction

Robot Operating System 2 (ROS 2) represents the evolution of the Robot Operating System, designed specifically for production robotics applications. Unlike its predecessor, ROS 2 is built on modern middleware technologies and addresses the critical needs of real-world robotic systems: real-time performance, security, determinism, and robustness. This chapter explores the architecture, design principles, and practical applications of ROS 2 in modern robotics.

## Learning Objectives

- Understand the fundamental architecture of ROS 2
- Identify the key differences between ROS 1 and ROS 2
- Explain the role of DDS in ROS 2 communication
- Recognize the benefits of ROS 2 for production robotics
- Describe the core concepts of nodes, topics, services, and actions

## Conceptual Foundations

ROS 2 is built on a fundamentally different architecture compared to ROS 1, addressing many of the limitations of the original system:

**Middleware-based Architecture**: ROS 2 uses Data Distribution Service (DDS) as its underlying communication middleware, providing a standards-based, vendor-neutral communication system that offers quality of service (QoS) controls, real-time performance, and security features.

**Real-time Support**: Unlike ROS 1 which relied on TCPROS/UDPROS, ROS 2 is designed with real-time systems in mind, supporting deterministic communication and real-time scheduling.

**Security by Design**: ROS 2 incorporates security features at the communication level, including authentication, access control, and data encryption.

**Deterministic Communication**: With DDS, ROS 2 can provide more predictable communication patterns, essential for safety-critical applications.

**Language and Platform Independence**: While maintaining Python and C++ support, ROS 2 is designed to be more easily portable to different platforms and languages.

## Technical Explanation

### Architecture Components

**DDS Implementation**: ROS 2 uses DDS as its communication layer, with different vendors providing implementations (Fast DDS, Cyclone DDS, RTI Connext DDS, etc.). DDS provides:
- Publish/subscribe communication patterns
- Request/reply communication patterns
- Quality of Service (QoS) policies for reliability, durability, and performance
- Discovery mechanisms for automatic node detection
- Data modeling with type definitions

**RMW (ROS Middleware) Layer**: The ROS Middleware layer abstracts the specific DDS implementation, allowing users to switch between different DDS vendors without changing application code.

**rcl/rclcpp/rclpy**: The ROS Client Libraries provide the interface between user code and the middleware layer.

### Quality of Service (QoS) Policies

ROS 2 introduces QoS policies that allow fine-tuning of communication behavior:

- **Reliability**: Reliable (all messages delivered) or Best Effort (messages may be dropped)
- **Durability**: Transient Local (historical data available) or Volatile (only new data)
- **History**: Keep All or Keep Last N messages
- **Depth**: Size of the queue for storing messages
- **Deadline**: Maximum time between consecutive messages
- **Lifespan**: Maximum lifetime of a message

### Node Architecture

Nodes in ROS 2 are more robust and secure than in ROS 1:
- Nodes can be distributed across multiple machines
- Each node has its own executor for managing callbacks
- Nodes can be configured with security policies
- Lifecycle nodes provide state management for complex systems

## Practical Examples

### Example 1: Basic Publisher and Subscriber

A simple publisher and subscriber in ROS 2:

```python
# publisher_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# subscriber_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 2: Quality of Service Configuration

Configuring QoS policies for different communication needs:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class QoSDemoNode(Node):
    def __init__(self):
        super().__init__('qos_demo_node')

        # For sensor data (e.g., camera images) - best effort, keep last
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        self.sensor_publisher = self.create_publisher(Image, 'sensor_data', sensor_qos)

        # For control commands - reliable, keep all
        control_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL
        )
        self.control_publisher = self.create_publisher(Twist, 'cmd_vel', control_qos)

        # For safety-critical data - reliable, with deadline
        safety_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            deadline=rclpy.duration.Duration(seconds=0.1)  # 100ms deadline
        )
        self.safety_publisher = self.create_publisher(String, 'safety_status', safety_qos)

    def publish_sensor_data(self, image_data):
        """Publish sensor data with best-effort QoS"""
        msg = Image()
        # Fill in image data
        self.sensor_publisher.publish(msg)

    def publish_control_command(self, cmd_vel):
        """Publish control commands with reliable QoS"""
        msg = Twist()
        msg.linear.x = cmd_vel.linear_x
        msg.angular.z = cmd_vel.angular_z
        self.control_publisher.publish(msg)
```

### Example 3: Lifecycle Node

A lifecycle node that manages its own state:

```python
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from std_msgs.msg import String

class LifecycleTalker(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_talker')
        self.pub = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when configuring the node"""
        self.get_logger().info(f'Configuring {self.get_name()}')
        self.pub = self.create_publisher(String, 'lifecycle_chatter', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when activating the node"""
        self.get_logger().info(f'Activating {self.get_name()}')
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when deactivating the node"""
        self.get_logger().info(f'Deactivating {self.get_name()}')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Called when cleaning up the node"""
        self.get_logger().info(f'Cleaning up {self.get_name()}')
        self.destroy_publisher(self.pub)
        self.pub = None
        return TransitionCallbackReturn.SUCCESS

    def publish_message(self):
        """Publish a message if active"""
        if self.pub is not None and self.pub.handle is not None:
            msg = String()
            msg.data = f'Lifecycle chatter: {self.get_clock().now().nanoseconds}'
            self.pub.publish(msg)
```

## System Integration Perspective

ROS 2 serves as the backbone for modern robotics systems by providing:

**Communication Infrastructure**: ROS 2 handles the complex task of inter-process communication, allowing developers to focus on application logic rather than communication protocols.

**Standardized Interfaces**: The common message types and service definitions provide standardized interfaces that enable code reuse and system integration.

**Middleware Abstraction**: The RMW layer allows switching between different DDS implementations without changing application code, providing flexibility for different deployment scenarios.

**Security Framework**: Built-in security features enable secure communication between nodes, essential for production systems.

**Real-time Capabilities**: With proper configuration, ROS 2 can meet real-time requirements critical for many robotic applications.

**Distributed Computing**: ROS 2 supports distributed systems across multiple machines, enabling complex robotic systems with multiple computational units.

**Ecosystem Integration**: The extensive package ecosystem provides ready-made solutions for common robotic tasks, from perception to control.

## Summary

- ROS 2 uses DDS as its communication middleware, providing QoS controls and real-time capabilities
- Key improvements over ROS 1 include security, real-time support, and deterministic communication
- QoS policies allow fine-tuning of communication behavior for different use cases
- Lifecycle nodes provide state management for complex systems
- ROS 2 serves as a comprehensive backbone for modern robotics applications

## Exercises

1. **Architecture Analysis**: Compare the communication architecture of ROS 1 and ROS 2. What are the key differences and why were these changes made?

2. **QoS Configuration**: For a mobile robot with a camera, LIDAR, and control system, design appropriate QoS policies for each communication channel. Justify your choices.

3. **Security Implementation**: How would you implement security for a ROS 2 system that operates in a public environment? What specific security features would you enable?

4. **Real-time Performance**: A robot needs to maintain a 100Hz control loop while also processing sensor data. How would you configure ROS 2 to support this real-time requirement?

5. **System Design**: Design a ROS 2 system architecture for a humanoid robot with perception, planning, and control components. Identify the nodes, topics, services, and QoS policies needed.
