---
title: "Human-Robot Interaction Using Unity"
sidebar_position: 3
---

# 3. Human-Robot Interaction Using Unity

## Introduction

Human-Robot Interaction (HRI) is a critical aspect of robotics that focuses on the design and implementation of interfaces that enable effective communication between humans and robots. Unity, a powerful game development platform, has emerged as an excellent tool for creating interactive HRI applications, virtual environments, and user interfaces for robotic systems. This chapter explores the principles and implementation of HRI systems using Unity, leveraging its real-time rendering capabilities and extensive development tools.

## Learning Objectives

- Understand the principles of Human-Robot Interaction design
- Identify Unity's capabilities for HRI applications
- Design intuitive interfaces for robot control and monitoring
- Implement communication protocols between Unity and robotic systems
- Recognize best practices for HRI system development

## Conceptual Foundations

Human-Robot Interaction using Unity builds on several key principles:

**Intuitive Interfaces**: HRI systems should provide clear, intuitive ways for humans to communicate with robots. Unity's visual development environment enables rapid prototyping of user interfaces.

**Real-time Feedback**: Effective HRI requires immediate feedback from the robot to the human operator, which Unity's real-time rendering engine can provide.

**Immersive Environments**: Unity enables the creation of immersive environments where humans can visualize robot behavior and environment in real-time.

**Multi-modal Interaction**: Modern HRI systems support various interaction modes including visual, auditory, and haptic feedback, all of which can be implemented in Unity.

**Safety Considerations**: HRI systems must ensure safe interaction between humans and robots, with proper safeguards and feedback mechanisms.

## Technical Explanation

### Unity for Robotics

Unity provides several features that make it suitable for HRI applications:

**Real-time Rendering**: Unity's rendering pipeline provides high-quality, real-time visualization of robot environments and states.

**Cross-platform Deployment**: Unity applications can be deployed to various platforms including desktop, mobile, VR, and AR systems.

**Component-Based Architecture**: Unity's GameObject-Component system provides a flexible way to structure HRI applications.

**Extensive Asset Ecosystem**: The Unity Asset Store provides numerous tools and assets for robotics applications.

### Communication Protocols

Unity can communicate with robotic systems through various protocols:

**ROS/ROS 2 Integration**: Unity can interface with ROS/ROS 2 systems using packages like Unity Robotics Hub.

**WebSocket Communication**: Real-time bidirectional communication with robotic systems.

**TCP/UDP Sockets**: Direct network communication with robots.

**Serial Communication**: Direct communication with robot controllers.

### HRI Design Principles

**Visibility of System Status**: Users should always be aware of the robot's current state and behavior.

**Match Between System and Real World**: The interface should use familiar concepts and representations.

**User Control and Freedom**: Users should have clear ways to control the robot and undo actions.

**Consistency and Standards**: Interfaces should be consistent and follow established conventions.

**Error Prevention**: The system should prevent errors where possible.

## Practical Examples

### Example 1: Basic Robot Teleoperation Interface

Creating a Unity interface for robot teleoperation:

```csharp
using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class RobotTeleopController : MonoBehaviour
{
    [Header("Robot Connection")]
    public string robotTopic = "/cmd_vel";
    public float linearSpeed = 1.0f;
    public float angularSpeed = 1.0f;

    [Header("Input Settings")]
    public KeyCode forwardKey = KeyCode.W;
    public KeyCode backwardKey = KeyCode.S;
    public KeyCode leftKey = KeyCode.A;
    public KeyCode rightKey = KeyCode.D;

    private ROSConnection ros;
    private TwistStampedMsg lastCommand;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistStampedMsg>(robotTopic);

        // Initialize command
        lastCommand = new TwistStampedMsg();
        lastCommand.header = new StdMsgs.HeaderMsg();
    }

    void Update()
    {
        // Check for user input
        Vector3 linear = Vector3.zero;
        Vector3 angular = Vector3.zero;

        if (Input.GetKey(forwardKey))
            linear.z = linearSpeed;
        else if (Input.GetKey(backwardKey))
            linear.z = -linearSpeed;

        if (Input.GetKey(leftKey))
            angular.y = angularSpeed;
        else if (Input.GetKey(rightKey))
            angular.y = -angularSpeed;

        // Create and send command if input has changed
        if (linear != Vector3.zero || angular != Vector3.zero)
        {
            SendCommand(linear, angular);
        }
    }

    void SendCommand(Vector3 linear, Vector3 angular)
    {
        // Update header
        lastCommand.header.stamp = new TimeStamp(Time.time);
        lastCommand.header.frame_id = "base_link";

        // Set velocities
        lastCommand.twist.linear = new Vector3Msg(linear.x, linear.y, linear.z);
        lastCommand.twist.angular = new Vector3Msg(angular.x, angular.y, angular.z);

        // Send to robot
        ros.Publish(robotTopic, lastCommand);
    }

    // Method to stop the robot
    public void StopRobot()
    {
        SendCommand(Vector3.zero, Vector3.zero);
    }
}
```

### Example 2: Robot Visualization and State Monitoring

Creating a Unity scene that visualizes robot state and environment:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Nav;
using RosMessageTypes.Std;
using System.Collections.Generic;

public class RobotVisualizer : MonoBehaviour
{
    [Header("ROS Topics")]
    public string laserScanTopic = "/scan";
    public string odometryTopic = "/odom";
    public string imageTopic = "/camera/image_raw";

    [Header("Visualization Settings")]
    public GameObject robotModel;
    public Material obstacleMaterial;
    public Material freeSpaceMaterial;

    [Header("Display Settings")]
    public Transform laserScanParent;
    public float maxLaserDistance = 10.0f;

    private ROSConnection ros;
    private List<GameObject> laserRays = new List<GameObject>();
    private OdomMsg lastOdom;
    private LaserScanMsg lastScan;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to topics
        ros.Subscribe<LaserScanMsg>(laserScanTopic, LaserScanCallback);
        ros.Subscribe<OdomMsg>(odometryTopic, OdometryCallback);

        // Initialize laser visualization
        InitializeLaserVisualization();
    }

    void InitializeLaserVisualization()
    {
        // Create laser ray visualizations
        for (int i = 0; i < 360; i++)
        {
            GameObject ray = new GameObject($"LaserRay_{i}");
            ray.transform.SetParent(laserScanParent);
            ray.transform.position = transform.position;
            ray.AddComponent<LineRenderer>();

            LineRenderer lineRenderer = ray.GetComponent<LineRenderer>();
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            lineRenderer.widthMultiplier = 0.02f;
            lineRenderer.positionCount = 2;

            laserRays.Add(ray);
        }
    }

    void LaserScanCallback(LaserScanMsg scan)
    {
        lastScan = scan;
        UpdateLaserVisualization();
    }

    void OdometryCallback(OdomMsg odom)
    {
        lastOdom = odom;
        UpdateRobotPose();
    }

    void UpdateLaserVisualization()
    {
        if (lastScan == null) return;

        for (int i = 0; i < lastScan.ranges.Length; i++)
        {
            float distance = lastScan.ranges[i];
            if (distance > lastScan.range_max || distance < lastScan.range_min)
            {
                // Invalid range, hide ray
                LineRenderer lineRenderer = laserRays[i].GetComponent<LineRenderer>();
                lineRenderer.enabled = false;
                continue;
            }

            // Calculate angle
            float angle = lastScan.angle_min + i * lastScan.angle_increment;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            Vector3 endPosition = transform.position + direction * distance;

            // Update line renderer
            LineRenderer lineRenderer = laserRays[i].GetComponent<LineRenderer>();
            lineRenderer.enabled = true;
            lineRenderer.SetPosition(0, transform.position);
            lineRenderer.SetPosition(1, endPosition);

            // Color based on distance
            float colorValue = Mathf.InverseLerp(0, maxLaserDistance, distance);
            lineRenderer.startColor = Color.Lerp(Color.red, Color.green, colorValue);
            lineRenderer.endColor = Color.Lerp(Color.red, Color.green, colorValue);
        }
    }

    void UpdateRobotPose()
    {
        if (lastOdom == null || robotModel == null) return;

        // Update robot position
        Vector3 position = new Vector3(
            (float)lastOdom.pose.pose.position.x,
            (float)lastOdom.pose.pose.position.z, // Using z as y for visualization
            (float)lastOdom.pose.pose.position.y  // Using y as z for visualization
        );
        robotModel.transform.position = position;

        // Update robot rotation
        Quaternion rotation = new Quaternion(
            (float)lastOdom.pose.pose.orientation.x,
            (float)lastOdom.pose.pose.orientation.z,
            (float)lastOdom.pose.pose.orientation.y,
            (float)lastOdom.pose.pose.orientation.w
        );
        robotModel.transform.rotation = rotation;
    }

    void OnDestroy()
    {
        // Clean up laser rays
        foreach (GameObject ray in laserRays)
        {
            if (ray != null)
                DestroyImmediate(ray);
        }
        laserRays.Clear();
    }
}
```

### Example 3: Advanced HRI Interface with Multiple Views

Creating a comprehensive HRI interface with multiple visualization panels:

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Nav;
using RosMessageTypes.Std;

public class AdvancedHRIInterface : MonoBehaviour
{
    [Header("UI References")]
    public Canvas mainCanvas;
    public RawImage cameraFeed;
    public Text statusText;
    public Text batteryLevel;
    public Text robotPosition;
    public Slider linearSpeedSlider;
    public Slider angularSpeedSlider;
    public Button emergencyStopButton;
    public Toggle autonomousModeToggle;

    [Header("ROS Topics")]
    public string cameraTopic = "/camera/image_raw";
    public string batteryTopic = "/battery_level";
    public string cmdVelTopic = "/cmd_vel";
    public string poseTopic = "/robot_pose";

    [Header("Robot Control")]
    public float maxLinearSpeed = 2.0f;
    public float maxAngularSpeed = 1.0f;

    private ROSConnection ros;
    private Dictionary<string, System.Action<Message>> subscribers = new Dictionary<string, System.Action<Message>>();

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Initialize UI
        InitializeUI();

        // Subscribe to topics
        SubscribeToTopics();

        // Setup event listeners
        SetupEventListeners();
    }

    void InitializeUI()
    {
        // Set up speed sliders
        linearSpeedSlider.minValue = 0f;
        linearSpeedSlider.maxValue = maxLinearSpeed;
        linearSpeedSlider.value = maxLinearSpeed / 2;

        angularSpeedSlider.minValue = 0f;
        angularSpeedSlider.maxValue = maxAngularSpeed;
        angularSpeedSlider.value = maxAngularSpeed / 2;

        // Set up emergency stop button
        emergencyStopButton.onClick.AddListener(EmergencyStop);
    }

    void SubscribeToTopics()
    {
        // Subscribe to various topics
        ros.Subscribe<ImageMsg>(cameraTopic, CameraCallback);
        ros.Subscribe<Float32Msg>(batteryTopic, BatteryCallback);
        ros.Subscribe<PoseStampedMsg>(poseTopic, PoseCallback);
    }

    void SetupEventListeners()
    {
        // Add listeners for UI elements
        linearSpeedSlider.onValueChanged.AddListener(OnLinearSpeedChanged);
        angularSpeedSlider.onValueChanged.AddListener(OnAngularSpeedChanged);
        autonomousModeToggle.onValueChanged.AddListener(OnAutonomousModeChanged);
    }

    void CameraCallback(ImageMsg image)
    {
        // Process camera image and display in UI
        // Note: This is a simplified example - real implementation would need to
        // convert ROS image format to Unity texture
        StartCoroutine(ProcessCameraImage(image));
    }

    IEnumerator ProcessCameraImage(ImageMsg image)
    {
        // In a real implementation, you would convert the ROS image data
        // to a Unity texture and assign it to the cameraFeed RawImage
        // For this example, we'll just simulate the process

        // Simulate processing delay
        yield return new WaitForSeconds(0.033f); // ~30 FPS

        // Update UI
        statusText.text = "Camera: Active";
    }

    void BatteryCallback(Float32Msg battery)
    {
        // Update battery level display
        batteryLevel.text = $"Battery: {battery.data:F1}%";

        // Change color based on battery level
        if (battery.data < 20)
            batteryLevel.color = Color.red;
        else if (battery.data < 50)
            batteryLevel.color = Color.yellow;
        else
            batteryLevel.color = Color.green;
    }

    void PoseCallback(PoseStampedMsg pose)
    {
        // Update robot position display
        robotPosition.text = $"Position: ({pose.pose.position.x:F2}, {pose.pose.position.y:F2}, {pose.pose.position.z:F2})";
    }

    void OnLinearSpeedChanged(float value)
    {
        // Update linear speed based on slider
        statusText.text = $"Linear Speed: {value:F2} m/s";
    }

    void OnAngularSpeedChanged(float value)
    {
        // Update angular speed based on slider
        statusText.text = $"Angular Speed: {value:F2} rad/s";
    }

    void OnAutonomousModeChanged(bool isAutonomous)
    {
        if (isAutonomous)
        {
            statusText.text = "Mode: Autonomous";
            // Disable manual controls
            linearSpeedSlider.interactable = false;
            angularSpeedSlider.interactable = false;
        }
        else
        {
            statusText.text = "Mode: Manual";
            // Enable manual controls
            linearSpeedSlider.interactable = true;
            angularSpeedSlider.interactable = true;
        }
    }

    public void EmergencyStop()
    {
        // Send emergency stop command
        var twist = new TwistStampedMsg();
        twist.header = new StdMsgs.HeaderMsg();
        twist.header.stamp = new TimeStamp(Time.time);
        twist.twist.linear = new Vector3Msg(0, 0, 0);
        twist.twist.angular = new Vector3Msg(0, 0, 0);

        ros.Publish(cmdVelTopic, twist);

        statusText.text = "EMERGENCY STOP ACTIVATED";
    }

    void Update()
    {
        // Handle keyboard input for manual control
        if (!autonomousModeToggle.isOn)
        {
            float linear = 0, angular = 0;

            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))
                linear = linearSpeedSlider.value;
            else if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
                linear = -linearSpeedSlider.value;

            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
                angular = angularSpeedSlider.value;
            else if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
                angular = -angularSpeedSlider.value;

            if (linear != 0 || angular != 0)
            {
                SendVelocityCommand(linear, angular);
            }
        }
    }

    void SendVelocityCommand(float linear, float angular)
    {
        var twist = new TwistStampedMsg();
        twist.header = new StdMsgs.HeaderMsg();
        twist.header.stamp = new TimeStamp(Time.time);
        twist.twist.linear = new Vector3Msg(0, 0, linear);
        twist.twist.angular = new Vector3Msg(0, angular, 0);

        ros.Publish(cmdVelTopic, twist);
    }

    void OnDestroy()
    {
        // Clean up subscribers
        ros = null;
    }
}
```

## System Integration Perspective

Unity-based HRI systems require integration across multiple system components:

**Robot Communication**: Establishing reliable communication between Unity and robotic systems:
- Real-time data exchange with minimal latency
- Proper message serialization and deserialization
- Error handling and reconnection mechanisms

**User Experience Design**: Creating intuitive and effective interfaces:
- Human-centered design principles
- Accessibility considerations
- Visual feedback and status indicators
- Responsive control interfaces

**Performance Optimization**: Ensuring smooth operation:
- Efficient rendering pipelines
- Optimized network communication
- Resource management for complex scenes
- Multi-threading for non-blocking operations

**Safety Systems**: Implementing safety measures:
- Emergency stop functionality
- Safety boundary visualization
- Operator authentication
- Command validation and limits

**Data Visualization**: Presenting complex information clearly:
- Multi-view displays
- Real-time data visualization
- Historical data analysis
- Sensor fusion visualization

## Summary

- Unity provides powerful tools for creating HRI interfaces with real-time visualization
- Effective HRI systems require intuitive interfaces and real-time feedback
- Communication protocols enable integration between Unity and robotic systems
- Proper design considers user experience, performance, and safety
- Advanced HRI interfaces can provide comprehensive robot monitoring and control

## Exercises

1. **Interface Design**: Design a Unity interface for controlling a robotic arm with 6 degrees of freedom. What UI elements would you include for position and orientation control?

2. **Multi-Modal Interaction**: Implement a Unity HRI system that provides visual, auditory, and haptic feedback. How would you integrate these different modalities?

3. **Safety System**: Design and implement a safety system for a Unity-based robot control interface that includes emergency stops, boundary limits, and collision warnings.

4. **VR Integration**: How would you modify the teleoperation interface to work in a VR environment? What advantages and challenges would this present?

5. **Performance Optimization**: For a complex HRI system with multiple robots and sensors, identify potential performance bottlenecks in Unity and suggest optimization strategies.
