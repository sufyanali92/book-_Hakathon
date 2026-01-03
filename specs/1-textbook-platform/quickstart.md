# Quickstart Guide: AI-Native Technical Textbook Platform

## Prerequisites

- Node.js (version 18 or higher)
- npm or yarn package manager
- Git
- GitHub account (for deployment)

## Setup Instructions

### 1. Initialize Docusaurus Project

```bash
# Create a new Docusaurus project
npx create-docusaurus@latest textbook-ai-robotics classic

# Navigate to the project directory
cd textbook-ai-robotics
```

### 2. Install Dependencies

```bash
# Install required dependencies
npm install

# Or if using yarn
yarn install
```

### 3. Set up the Project Structure

```bash
# Create the directory structure for the 4 parts and 16 chapters
mkdir -p docs/part-i-foundations
mkdir -p docs/part-ii-ros
mkdir -p docs/part-iii-digital-twins
mkdir -p docs/part-iv-vision-language-action
```

### 4. Configure Sidebar Navigation

Create or update `sidebars.js`:

```javascript
// sidebars.js
module.exports = {
  textbook: [
    {
      type: 'category',
      label: 'Part I – Foundations of Physical AI',
      items: [
        'part-i-foundations/understanding-physical-ai',
        'part-i-foundations/core-principles-robotic-systems',
        'part-i-foundations/intelligence-physical-world',
        'part-i-foundations/humanoid-robotics-software-stack',
      ],
    },
    {
      type: 'category',
      label: 'Part II – The Robotic Nervous System (ROS 2)',
      items: [
        'part-ii-ros/ros2-backbone-modern-robotics',
        'part-ii-ros/robot-communication-nodes-topics-messages',
        'part-ii-ros/services-actions-parameters-ros2',
        'part-ii-ros/building-ros2-nodes-python',
      ],
    },
    {
      type: 'category',
      label: 'Part III – Digital Twins & Robot Brains',
      items: [
        'part-iii-digital-twins/physics-simulation-gazebo',
        'part-iii-digital-twins/simulated-sensors-perception',
        'part-iii-digital-twins/human-robot-interaction-unity',
        'part-iii-digital-twins/high-fidelity-simulation-nvidia-isaac',
      ],
    },
    {
      type: 'category',
      label: 'Part IV – Vision-Language-Action & Autonomy',
      items: [
        'part-iv-vision-language-action/vision-language-action-models',
        'part-iv-vision-language-action/speech-understanding-whisper',
        'part-iv-vision-language-action/language-action-planning',
        'part-iv-vision-language-action/autonomous-reasoning-decision-making',
      ],
    },
  ],
};
```

### 5. Create Sample Chapter Content

Create a sample chapter file at `docs/part-i-foundations/understanding-physical-ai.md`:

```markdown
---
title: Understanding Physical AI
sidebar_position: 1
---

# Understanding Physical AI

## Introduction

This chapter introduces the fundamental concepts of Physical AI, which bridges the gap between artificial intelligence and real-world robotic systems.

## Learning Objectives

- Define Physical AI and its significance
- Understand the relationship between AI and robotics
- Identify key challenges in Physical AI
- Recognize applications of Physical AI in humanoid robotics

## Core Concepts

Physical AI represents the intersection of artificial intelligence and physical systems. Unlike traditional AI that operates in virtual environments, Physical AI must contend with:

- Real-world physics and constraints
- Sensor and actuator limitations
- Uncertainty and noise in measurements
- Safety and reliability requirements

## Practical Examples

### Example 1: Object Manipulation
[Practical example content here]

### Example 2: Navigation in Dynamic Environments
[Practical example content here]

## Python Code Examples

```python
# Example: Basic robot movement control
import rospy
from geometry_msgs.msg import Twist

def move_robot():
    rospy.init_node('robot_mover')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Move forward for 1 second
    move_cmd = Twist()
    move_cmd.linear.x = 0.5  # Move forward at 0.5 m/s
    pub.publish(move_cmd)
    rospy.sleep(1.0)

    # Stop the robot
    move_cmd.linear.x = 0.0
    pub.publish(move_cmd)

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
```

## Diagram Placeholders

![Robot Control Architecture](/img/robot-control-architecture.png)

## Summary

Physical AI is a critical field that combines AI algorithms with real-world physical systems. Understanding its principles is essential for developing effective humanoid robots.

## Exercises

### Exercise 1: Conceptual Understanding
Explain in your own words the difference between traditional AI and Physical AI. What are the unique challenges that arise when AI interacts with physical systems?

### Exercise 2: Practical Application
Design a simple control algorithm for a robot that needs to navigate around obstacles. Consider the sensor inputs and control outputs required.

## Solutions

### Exercise 1 Solution:
[Solution content here]

### Exercise 2 Solution:
[Solution content here]
```

### 6. Local Development

```bash
# Start the development server
npm run start

# The site will be available at http://localhost:3000
```

### 7. Build and Deploy

```bash
# Build the static files
npm run build

# Deploy to GitHub Pages
GIT_USER=<your-github-username> npm run deploy
```

## Configuration

### Docusaurus Configuration

Update `docusaurus.config.js` with project-specific settings:

```javascript
// docusaurus.config.js
module.exports = {
  title: 'AI-Native Technical Textbook: Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive guide to Physical AI and Humanoid Robotics',
  url: 'https://<your-username>.github.io',
  baseUrl: '/textbook-ai-robotics/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: '<your-username>',
  projectName: 'textbook-ai-robotics',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/<your-username>/textbook-ai-robotics/edit/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  themeConfig: {
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'doc',
          docId: 'intro',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/<your-username>/textbook-ai-robotics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Content',
          items: [
            {
              label: 'Part I - Foundations',
              to: '/docs/part-i-foundations/understanding-physical-ai',
            },
            {
              label: 'Part II - ROS 2',
              to: '/docs/part-ii-ros/ros2-backbone-modern-robotics',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: require('prism-react-renderer/themes/github'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
      additionalLanguages: ['python', 'bash'],
    },
  },
};
```

## Next Steps

1. Create all 16 chapter files following the sample structure
2. Add Python and ROS code examples to each chapter
3. Include exercises at the end of each chapter
4. Add diagram placeholders and actual diagrams
5. Review content for technical accuracy
6. Test the build process
7. Deploy to GitHub Pages