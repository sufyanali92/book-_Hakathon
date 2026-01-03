// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Part I – Foundations of Physical AI',
      items: [
        'part-i-foundations/understanding-physical-ai',
        'part-i-foundations/core-principles-robotic-systems',
        'part-i-foundations/intelligence-physical-world',
        'part-i-foundations/humanoid-robotics-software-stack'
      ],
    },
    {
      type: 'category',
      label: 'Part II – Ros',
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
        'part-iii-digital-twins/physics-based-simulation-gazebo',
        'part-iii-digital-twins/simulated-sensors-perception',
        'part-iii-digital-twins/human-robot-interaction-using-unity',
        'part-iii-digital-twins/high-fidelity-simulation-nvidia-isaac',
      ],
    },
    {
      type: 'category',
      label: 'Part IV – Vision-Language-Action & Autonomy',
      items: [
        'part-iv-vision-language-action/vision-language-action-models-explained',
        'part-iv-vision-language-action/speech-understanding-whisper',
        'part-iv-vision-language-action/from-language-to-action-planning',
        'part-iv-vision-language-action/autonomous-reasoning-and-decision-making',
      ],
    },
  ],
};

export default sidebars;
