---
title: "Services, Actions, and Parameters in ROS 2"
sidebar_position: 3
---

# 3. Services, Actions, and Parameters in ROS 2

## Introduction

While topics enable asynchronous, continuous communication between nodes in ROS 2, services, actions, and parameters provide additional communication patterns for different types of interactions. Services offer synchronous request-response communication, actions provide goal-oriented communication with feedback, and parameters enable configuration management. This chapter explores these three essential communication patterns that complement the publish-subscribe model.

## Learning Objectives

- Understand the service-server communication pattern in ROS 2
- Explain the action-goal communication model with feedback and status
- Identify the role of parameters in ROS 2 node configuration
- Design appropriate communication patterns for different use cases
- Recognize when to use services, actions, or parameters vs. topics

## Conceptual Foundations

ROS 2 provides three additional communication patterns beyond topics:

**Services** implement a synchronous request-response pattern where a client sends a request to a server and waits for a response. This is ideal for operations that have a clear beginning and end, such as triggering a behavior or requesting a computation.

**Actions** provide a more complex communication pattern for long-running tasks that require feedback, goal management, and status updates. Actions are perfect for tasks like navigation, manipulation, or calibration that take time and need to report progress.

**Parameters** offer a configuration system that allows nodes to be configured at runtime. Parameters are key-value pairs that can be set at launch time or changed during operation, enabling dynamic reconfiguration of robot systems.

Each pattern serves different communication needs and should be chosen based on the characteristics of the interaction required.

## Technical Explanation

### Services

Services in ROS 2 use a request-response pattern:

- **Service Types**: Defined in .srv files with separate request and response message definitions
- **Synchronous Communication**: Client blocks until response is received
- **One-to-One**: Each service call is between one client and one server
- **Request-Response Structure**: Service definition includes both request and response message types

Service definition structure:
```
# Request message
string command
int32 value
---
# Response message
bool success
string message
```

### Actions

Actions are designed for long-running operations:

- **Goal-Result-Feedback Pattern**: Clients send goals, receive feedback during execution, and get results when complete
- **State Management**: Actions have built-in state management (pending, active, succeeded, aborted, canceled)
- **Preemption**: Goals can be canceled or preempted by new goals
- **Action Types**: Defined in .action files with three message types (Goal, Result, Feedback)

Action definition structure:
```
# Goal definition
float64 target_x
float64 target_y
float64 target_theta
---
# Result definition
bool success
string message
float64 final_x
float64 final_y
---
# Feedback definition
float64 current_x
float64 current_y
float64 distance_remaining
```

### Parameters

Parameters provide runtime configuration:

- **Dynamic Reconfiguration**: Parameters can be changed at runtime
- **Type Safety**: Parameters have defined types (bool, int, double, string, lists)
- **Node Namespacing**: Parameters are associated with specific nodes
- **Callback Interface**: Nodes can be notified when parameters change
- **Declarative Interface**: Parameters can be declared with default values and constraints

## Practical Examples

### Example 1: Services for Robot Control

Implementing services for robot operations:

```python
# Service definition: srv/SetMode.srv
# string mode
# ---
# bool success
# string message

import rclpy
from rclpy.node import Node
from your_package.srv import SetMode  # Custom service
from std_msgs.msg import String

class RobotServiceServer(Node):
    def __init__(self):
        super().__init__('robot_service_server')

        # Create service server
        self.srv = self.create_service(SetMode, 'set_robot_mode', self.set_mode_callback)

        # Publisher for status updates
        self.status_pub = self.create_publisher(String, 'robot_status', 10)

        # Internal state
        self.current_mode = 'idle'
        self.modes = ['idle', 'navigation', 'manipulation', 'calibration']

    def set_mode_callback(self, request, response):
        """Callback for setting robot mode"""
        requested_mode = request.mode.lower()

        if requested_mode in self.modes:
            old_mode = self.current_mode
            self.current_mode = requested_mode

            # Publish status update
            status_msg = String()
            status_msg.data = f'Mode changed from {old_mode} to {requested_mode}'
            self.status_pub.publish(status_msg)

            response.success = True
            response.message = f'Mode set to {requested_mode}'

            self.get_logger().info(f'Mode changed: {old_mode} -> {requested_mode}')
        else:
            response.success = False
            response.message = f'Invalid mode: {requested_mode}. Valid modes: {self.modes}'

        return response

class RobotServiceClient(Node):
    def __init__(self):
        super().__init__('robot_service_client')

        # Create client for the service
        self.cli = self.create_client(SetMode, 'set_robot_mode')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = SetMode.Request()

    def send_request(self, mode):
        """Send a request to set robot mode"""
        self.req.mode = mode
        future = self.cli.call_async(self.req)
        return future

def main(args=None):
    rclpy.init(args=args)

    # Create client node
    client = RobotServiceClient()

    # Send request
    future = client.send_request('navigation')

    # Wait for response
    rclpy.spin_until_future_complete(client, future)

    response = future.result()
    client.get_logger().info(f'Response: success={response.success}, message={response.message}')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 2: Actions for Navigation Tasks

Implementing actions for navigation with feedback:

```python
# Action definition: action/NavigateToPose.action
# # Goal
# geometry_msgs/PoseStamped target_pose
# float64 tolerance
# ---
# # Result
# bool success
# string message
# geometry_msgs/PoseStamped final_pose
# float64 distance_traveled
# ---
# # Feedback
# geometry_msgs/PoseStamped current_pose
# float64 distance_remaining
# float64 progress_percentage

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from your_package.action import NavigateToPose  # Custom action
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import threading
import time

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        # Subscribe to odometry for current pose
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        self.current_pose = None
        self._goal_handle = None
        self._goal_lock = threading.Lock()

    def goal_callback(self, goal_request):
        """Accept or reject a goal"""
        self.get_logger().info('Received navigation goal')
        # Check if we can accept the goal
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = PoseStamped()
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose

    def distance_2d(self, pose1, pose2):
        """Calculate 2D distance between two poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return (dx*dx + dy*dy)**0.5

    async def execute_callback(self, goal_handle):
        """Execute the navigation goal"""
        self.get_logger().info('Executing navigation goal...')

        goal = goal_handle.request.target_pose
        tolerance = goal_handle.request.tolerance

        # Store goal handle for cancellation
        with self._goal_lock:
            self._goal_handle = goal_handle

        # Initialize result
        result = NavigateToPose.Result()
        result.success = False
        result.message = "Navigation failed"

        # Navigation loop
        feedback_msg = NavigateToPose.Feedback()
        start_pose = self.current_pose.pose if self.current_pose else goal.pose

        while goal_handle.is_active:
            if goal_handle.is_canceling():
                result.success = False
                result.message = "Navigation canceled"
                goal_handle.canceled()
                return result

            if not self.current_pose:
                await rclpy.sleep_until_future_complete(self, rclpy.create_timer(0.1))
                continue

            # Calculate distance to goal
            distance_to_goal = self.distance_2d(self.current_pose.pose, goal.pose)

            # Update feedback
            feedback_msg.current_pose = self.current_pose
            feedback_msg.distance_remaining = distance_to_goal
            feedback_msg.progress_percentage = max(0.0, min(100.0,
                (1.0 - distance_to_goal/self.distance_2d(start_pose, goal.pose)) * 100.0))

            goal_handle.publish_feedback(feedback_msg)

            # Check if we've reached the goal
            if distance_to_goal <= tolerance:
                result.success = True
                result.message = "Navigation successful"
                result.final_pose = self.current_pose
                result.distance_traveled = self.distance_2d(start_pose, self.current_pose.pose)

                goal_handle.succeed()
                break

            # Simulate navigation progress (in real system, this would send commands)
            await rclpy.sleep_until_future_complete(self, rclpy.create_timer(0.1))

        return result

class NavigationActionClient(Node):
    def __init__(self):
        super().__init__('navigation_action_client')

        # Create action client
        self._action_client = self.create_client(NavigateToPose, 'navigate_to_pose')

        # Wait for action server
        self.get_logger().info('Waiting for action server...')
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Action server not available, waiting again...')

    def send_goal(self, x, y, z=0.0):
        """Send navigation goal to action server"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.target_pose.header.frame_id = 'map'
        goal_msg.target_pose.pose.position.x = x
        goal_msg.target_pose.pose.position.y = y
        goal_msg.target_pose.pose.position.z = z
        goal_msg.target_pose.pose.orientation.w = 1.0  # No rotation
        goal_msg.tolerance = 0.1  # 10cm tolerance

        self.get_logger().info(f'Sending goal: ({x}, {y})')

        # Send goal
        future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        future.add_done_callback(self.goal_response_callback)

        return future

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle result"""
        result = future.result().result
        self.get_logger().info(f'Result: {result.message}')
        self.get_logger().info(f'Success: {result.success}')

    def feedback_callback(self, feedback_msg):
        """Handle feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Progress: {feedback.progress_percentage:.1f}% - '
            f'Distance remaining: {feedback.distance_remaining:.2f}m')

def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    server_node = NavigationActionServer()
    client_node = NavigationActionClient()

    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(server_node)
    executor.add_node(client_node)

    # Send a goal from client
    import threading
    def send_goal_later():
        time.sleep(2)  # Wait for everything to initialize
        client_node.send_goal(2.0, 3.0)  # Navigate to (2, 3)

    goal_thread = threading.Thread(target=send_goal_later)
    goal_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        goal_thread.join(timeout=1.0)
        server_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 3: Parameters for Dynamic Configuration

Using parameters for runtime configuration:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64
import math

class ParameterizedController(Node):
    def __init__(self):
        super().__init__('parameterized_controller')

        # Declare parameters with default values and descriptions
        self.declare_parameter('kp', 1.0,
            ParameterDescriptor(description='Proportional gain for PID controller'))
        self.declare_parameter('ki', 0.1,
            ParameterDescriptor(description='Integral gain for PID controller'))
        self.declare_parameter('kd', 0.05,
            ParameterDescriptor(description='Derivative gain for PID controller'))
        self.declare_parameter('max_velocity', 1.0,
            ParameterDescriptor(description='Maximum velocity limit'))
        self.declare_parameter('control_frequency', 50,
            ParameterDescriptor(description='Control loop frequency in Hz'))

        # Initialize controller with current parameter values
        self.update_parameters()

        # Subscribe to setpoint and current value
        self.setpoint_sub = self.create_subscription(Float64, 'setpoint',
            self.setpoint_callback, 10)
        self.current_sub = self.create_subscription(Float64, 'current_value',
            self.current_callback, 10)

        # Publish control output
        self.output_pub = self.create_publisher(Float64, 'control_output', 10)

        # Timer for control loop
        control_period = 1.0 / self.get_parameter('control_frequency').value
        self.control_timer = self.create_timer(control_period, self.control_loop)

        # Callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Internal state for PID
        self.setpoint = 0.0
        self.current_value = 0.0
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = self.get_clock().now()

    def setpoint_callback(self, msg):
        self.setpoint = msg.data

    def current_callback(self, msg):
        self.current_value = msg.data

    def update_parameters(self):
        """Update controller parameters from node parameters"""
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.control_frequency = self.get_parameter('control_frequency').value

    def parameter_callback(self, params):
        """Callback for parameter changes"""
        for param in params:
            if param.name in ['kp', 'ki', 'kd', 'max_velocity', 'control_frequency']:
                self.get_logger().info(f'Parameter {param.name} changed to {param.value}')

        # Update parameters
        self.update_parameters()

        # Return success
        return SetParametersResult(successful=True)

    def control_loop(self):
        """Main PID control loop"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        if dt > 0:
            error = self.setpoint - self.current_value

            # Proportional term
            p_term = self.kp * error

            # Integral term
            self.integral += error * dt
            i_term = self.ki * self.integral

            # Derivative term
            derivative = (error - self.previous_error) / dt
            d_term = self.kd * derivative

            # Calculate output
            output = p_term + i_term + d_term

            # Apply velocity limit
            output = max(-self.max_velocity, min(self.max_velocity, output))

            # Publish control output
            output_msg = Float64()
            output_msg.data = output
            self.output_pub.publish(output_msg)

            # Update state
            self.previous_error = error
            self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    controller = ParameterizedController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## System Integration Perspective

The choice between services, actions, parameters, and topics affects system design:

**Service Integration**: Services are best for:
- Operations with clear start and end
- Request-response patterns
- Infrequent operations
- Operations that return a result immediately

**Action Integration**: Actions are best for:
- Long-running operations
- Tasks requiring feedback
- Operations that can be canceled
- Goal-oriented behaviors

**Parameter Integration**: Parameters are best for:
- Runtime configuration
- Dynamic reconfiguration
- System-wide settings
- Values that don't change frequently

**Topic Integration**: Topics are best for:
- Continuous data streams
- Real-time sensor data
- Frequent updates
- Decoupled communication

## Summary

- Services provide synchronous request-response communication
- Actions enable goal-oriented communication with feedback and status
- Parameters offer runtime configuration management
- Each pattern serves different communication needs
- Choosing the right pattern is crucial for system design

## Exercises

1. **Pattern Selection**: For the following scenarios, identify whether services, actions, parameters, or topics would be most appropriate: (a) requesting a map from a mapping node, (b) commanding a robot to navigate to a location, (c) adjusting the maximum speed of a robot, (d) sharing sensor data between nodes.

2. **Service Design**: Design a service for a robot that can request the identification of objects in a camera image. Define the request and response messages.

3. **Action Implementation**: Design an action for a robot arm to pick up an object. Define the goal, result, and feedback messages, and explain the state transitions.

4. **Parameter Architecture**: Design a parameter system for a mobile robot that allows configuration of its navigation behavior (speed limits, obstacle avoidance sensitivity, etc.). How would these parameters be organized?

5. **Integration Challenge**: Design a system architecture that uses all four communication patterns (topics, services, actions, parameters) for a robot that performs autonomous cleaning. Identify specific use cases for each pattern.
