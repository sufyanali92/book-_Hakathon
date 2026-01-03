---
title: "From Language to Action Planning"
sidebar_position: 3
---

# 3. From Language to Action Planning

## Introduction

The transformation of natural language instructions into executable robot actions represents one of the most challenging aspects of human-robot interaction. This process, known as language-to-action planning, involves understanding linguistic meaning, grounding abstract concepts in physical reality, and generating executable plans that achieve the intended goals. This chapter explores the principles, architectures, and methodologies for converting natural language commands into effective robot action sequences.

## Learning Objectives

- Understand the challenges in translating natural language to robot actions
- Identify the components of language-to-action planning systems
- Recognize the role of world modeling and spatial reasoning
- Explain the integration of natural language processing with action planning
- Evaluate the effectiveness of different planning approaches

## Conceptual Foundations

Language-to-action planning is built on several foundational concepts:

**Semantic Grounding**: The process of connecting linguistic symbols to physical entities and actions in the environment. This involves mapping abstract language concepts to concrete objects, locations, and behaviors.

**Compositional Understanding**: The ability to understand complex instructions by composing simpler elements. For example, "pick up the red cup and place it on the table" combines "pick up," "red cup," and "place on table."

**World Modeling**: The representation of the environment, objects, and their relationships. This model enables the robot to understand spatial relationships and object affordances.

**Hierarchical Planning**: The decomposition of complex tasks into subtasks and primitive actions that can be executed by the robot.

**Context Awareness**: Understanding the current situation and environment to interpret instructions appropriately.

## Technical Explanation

### Language Understanding Pipeline

The process of converting language to actions typically involves several stages:

**Natural Language Processing**: Initial processing of the input text to extract:
- Syntactic structure (parsing)
- Semantic roles (who does what to whom)
- Named entities (objects, locations, people)
- Temporal and spatial references

**Semantic Parsing**: Converting natural language to formal representations:
- Logical forms (first-order logic, lambda calculus)
- Abstract meaning representations
- Action schemas with parameters
- Goal specifications

**Grounding**: Connecting abstract concepts to the physical world:
- Object identification and localization
- Spatial relationship resolution
- Action parameter specification
- Constraint extraction

**Planning**: Generating executable action sequences:
- Task decomposition
- Constraint satisfaction
- Resource allocation
- Execution ordering

### Planning Architectures

**Symbolic Planning**: Uses formal logic and symbolic representations:
- STRIPS-style planning
- Hierarchical Task Networks (HTNs)
- Planning Domain Definition Language (PDDL)
- Logic-based reasoning

**Neural Planning**: Uses neural networks to learn planning patterns:
- End-to-end learning of language-to-action mappings
- Neural symbolic integration
- Attention mechanisms for focus
- Memory-augmented networks

**Hybrid Approaches**: Combines symbolic and neural methods:
- Neural networks for perception and language
- Symbolic reasoning for planning
- Probabilistic models for uncertainty
- Reinforcement learning for adaptation

### Key Technical Challenges

**Ambiguity Resolution**: Natural language is often ambiguous, requiring context and world knowledge to resolve references.

**Symbol Grounding**: Connecting abstract symbols to physical entities and actions.

**Scalability**: Handling complex, multi-step instructions efficiently.

**Robustness**: Operating in dynamic environments with uncertain information.

## Practical Examples

### Example 1: Symbolic Language-to-Action Planner

A symbolic planning system that converts natural language to action sequences:

```python
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    MOVE_TO = "move_to"
    PICK_UP = "pick_up"
    PLACE = "place"
    GRASP = "grasp"
    RELEASE = "release"
    GREET = "greet"
    FOLLOW = "follow"

@dataclass
class Action:
    type: ActionType
    parameters: Dict[str, Any]
    description: str

@dataclass
class Object:
    name: str
    color: str
    size: str
    location: Tuple[float, float, float]
    properties: List[str]

@dataclass
class Location:
    name: str
    coordinates: Tuple[float, float, float]
    properties: List[str]

class WorldModel:
    def __init__(self):
        self.objects = {}
        self.locations = {}
        self.robot_position = (0, 0, 0)
        self.held_object = None

    def add_object(self, obj: Object):
        self.objects[obj.name] = obj

    def add_location(self, loc: Location):
        self.locations[loc.name] = loc

    def find_object(self, name: str = None, color: str = None, size: str = None) -> List[Object]:
        """Find objects matching criteria"""
        candidates = list(self.objects.values())

        if name:
            candidates = [obj for obj in candidates if name.lower() in obj.name.lower()]
        if color:
            candidates = [obj for obj in candidates if obj.color.lower() == color.lower()]
        if size:
            candidates = [obj for obj in candidates if obj.size.lower() == size.lower()]

        return candidates

    def find_location(self, name: str) -> Location:
        """Find location by name"""
        return self.locations.get(name.lower())

class LanguageParser:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.action_patterns = [
            (r'pick up the (?P<color>\w+)?\s*(?P<object>\w+)', self.parse_pickup),
            (r'pick up (?P<color>\w+)?\s*(?P<object>\w+)', self.parse_pickup),
            (r'grasp the (?P<color>\w+)?\s*(?P<object>\w+)', self.parse_pickup),
            (r'place the (?P<object>\w+) on the (?P<location>\w+)', self.parse_place),
            (r'put the (?P<object>\w+) on the (?P<location>\w+)', self.parse_place),
            (r'move to the (?P<location>\w+)', self.parse_move_to),
            (r'go to the (?P<location>\w+)', self.parse_move_to),
            (r'go to (?P<location>\w+)', self.parse_move_to),
            (r'bring the (?P<color>\w+)?\s*(?P<object>\w+) to the (?P<location>\w+)', self.parse_transport),
        ]

    def parse_command(self, command: str) -> List[Action]:
        """Parse a natural language command into actions"""
        command = command.lower().strip()
        actions = []

        for pattern, handler in self.action_patterns:
            match = re.search(pattern, command)
            if match:
                actions.extend(handler(match))
                break

        if not actions:
            # If no pattern matched, try to extract general intent
            actions.extend(self.parse_general_command(command))

        return actions

    def parse_pickup(self, match: re.Match) -> List[Action]:
        """Parse pickup commands"""
        color = match.group('color')
        object_name = match.group('object')

        # Find the object in the world
        objects = self.world_model.find_object(object_name, color)

        if not objects:
            return [Action(ActionType.GRASP, {}, f"Could not find {color or ''} {object_name}")]

        target_object = objects[0]  # Take the first match

        # Move to object, then grasp it
        actions = [
            Action(ActionType.MOVE_TO, {
                'target': target_object.location,
                'object_name': target_object.name
            }, f"Move to {target_object.name}"),
            Action(ActionType.GRASP, {
                'object': target_object.name,
                'location': target_object.location
            }, f"Grasp {target_object.name}")
        ]

        return actions

    def parse_place(self, match: re.Match) -> List[Action]:
        """Parse place commands"""
        object_name = match.group('object')
        location_name = match.group('location')

        # Find the location
        location = self.world_model.find_location(location_name)
        if not location:
            return [Action(ActionType.RELEASE, {}, f"Could not find location {location_name}")]

        # Release the object at the location
        actions = [
            Action(ActionType.MOVE_TO, {
                'target': location.coordinates,
                'location_name': location.name
            }, f"Move to {location.name}"),
            Action(ActionType.RELEASE, {
                'location': location.coordinates,
                'location_name': location.name
            }, f"Release object at {location.name}")
        ]

        return actions

    def parse_move_to(self, match: re.Match) -> List[Action]:
        """Parse move to commands"""
        location_name = match.group('location')

        location = self.world_model.find_location(location_name)
        if not location:
            return [Action(ActionType.MOVE_TO, {}, f"Could not find location {location_name}")]

        return [
            Action(ActionType.MOVE_TO, {
                'target': location.coordinates,
                'location_name': location.name
            }, f"Move to {location.name}")
        ]

    def parse_transport(self, match: re.Match) -> List[Action]:
        """Parse transport commands (pick up and place)"""
        color = match.group('color')
        object_name = match.group('object')
        location_name = match.group('location')

        # Find the object
        objects = self.world_model.find_object(object_name, color)
        if not objects:
            return [Action(ActionType.GRASP, {}, f"Could not find {color or ''} {object_name}")]

        target_object = objects[0]
        location = self.world_model.find_location(location_name)
        if not location:
            return [Action(ActionType.MOVE_TO, {}, f"Could not find location {location_name}")]

        # Sequence: move to object, grasp, move to location, release
        return [
            Action(ActionType.MOVE_TO, {
                'target': target_object.location,
                'object_name': target_object.name
            }, f"Move to {target_object.name}"),
            Action(ActionType.GRASP, {
                'object': target_object.name,
                'location': target_object.location
            }, f"Grasp {target_object.name}"),
            Action(ActionType.MOVE_TO, {
                'target': location.coordinates,
                'location_name': location.name
            }, f"Move to {location.name}"),
            Action(ActionType.RELEASE, {
                'location': location.coordinates,
                'location_name': location.name
            }, f"Release object at {location.name}")
        ]

    def parse_general_command(self, command: str) -> List[Action]:
        """Handle commands that don't match specific patterns"""
        if 'hello' in command or 'hi' in command:
            return [Action(ActionType.GREET, {}, "Greet user")]
        elif 'follow' in command or 'come' in command:
            return [Action(ActionType.FOLLOW, {}, "Follow user")]
        else:
            return [Action(ActionType.MOVE_TO, {}, f"Unknown command: {command}")]

class SymbolicPlanner:
    def __init__(self):
        self.world_model = WorldModel()
        self.parser = LanguageParser(self.world_model)

        # Add some example objects and locations
        self.world_model.add_object(Object(
            name="red cup",
            color="red",
            size="medium",
            location=(1.0, 0.5, 0.0),
            properties=["graspable", "drinkable"]
        ))
        self.world_model.add_object(Object(
            name="blue box",
            color="blue",
            size="large",
            location=(2.0, 1.0, 0.0),
            properties=["graspable", "container"]
        ))
        self.world_model.add_location(Location(
            name="kitchen",
            coordinates=(3.0, 2.0, 0.0),
            properties=["cooking", "food"]
        ))
        self.world_model.add_location(Location(
            name="table",
            coordinates=(1.5, 0.5, 0.0),
            properties=["surface", "flat"]
        ))

    def plan_actions(self, command: str) -> List[Action]:
        """Plan actions for a given command"""
        return self.parser.parse_command(command)

    def execute_plan(self, actions: List[Action]) -> bool:
        """Simulate execution of planned actions"""
        print(f"Executing plan with {len(actions)} actions:")
        for i, action in enumerate(actions):
            print(f"  {i+1}. {action.description}")
            # In a real system, this would execute the action on the robot
            # For simulation, we'll just print and update the world model
            self._simulate_action(action)
        return True

    def _simulate_action(self, action: Action):
        """Simulate the effect of an action on the world model"""
        if action.type == ActionType.GRASP:
            obj_name = action.parameters.get('object')
            if obj_name:
                self.world_model.held_object = obj_name
                print(f"    -> Robot now holds {obj_name}")
        elif action.type == ActionType.RELEASE:
            if self.world_model.held_object:
                held = self.world_model.held_object
                self.world_model.held_object = None
                print(f"    -> Robot released {held}")
        elif action.type == ActionType.MOVE_TO:
            target = action.parameters.get('target')
            location_name = action.parameters.get('location_name')
            if target:
                self.world_model.robot_position = target
                print(f"    -> Robot moved to {location_name or target}")

# Example usage
def demo_symbolic_planning():
    """Demonstrate symbolic language-to-action planning"""
    planner = SymbolicPlanner()

    # Test commands
    commands = [
        "pick up the red cup",
        "place the cup on the table",
        "move to the kitchen",
        "pick up the blue box and place it on the table"
    ]

    for command in commands:
        print(f"\nProcessing command: '{command}'")

        # Plan actions
        actions = planner.plan_actions(command)

        print(f"Planned actions:")
        for action in actions:
            print(f"  - {action.description}")

        # Execute plan
        success = planner.execute_plan(actions)
        print(f"Plan execution: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    demo_symbolic_planning()
```

### Example 2: Neural-Symbolic Integration

A hybrid approach combining neural language understanding with symbolic planning:

```python
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Any
import json

class NeuralLanguageEncoder(nn.Module):
    """Neural encoder for natural language commands"""
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

        # Additional layers for command classification
        self.action_classifier = nn.Linear(768, 10)  # 10 different action types
        self.parameter_extractor = nn.Linear(768, 128)  # For extracting parameters

    def forward(self, command: str):
        # Tokenize and encode the command
        inputs = self.tokenizer(command, return_tensors="pt", padding=True, truncation=True)

        # Get BERT embeddings
        outputs = self.bert(**inputs)
        pooled_output = outputs.pooler_output  # [batch_size, 768]

        # Classify action type
        action_logits = self.action_classifier(pooled_output)

        # Extract parameters
        parameters = self.parameter_extractor(pooled_output)

        return {
            'action_logits': action_logits,
            'parameters': parameters,
            'embeddings': pooled_output
        }

class SymbolicActionPlanner:
    """Symbolic planner that works with neural language understanding"""
    def __init__(self):
        self.action_templates = {
            'move': ['go to', 'move to', 'navigate to', 'walk to'],
            'grasp': ['pick up', 'grasp', 'take', 'lift'],
            'place': ['place on', 'put on', 'set on', 'place at'],
            'follow': ['follow', 'come with', 'accompany'],
            'greet': ['hello', 'hi', 'greet', 'say hi'],
            'transport': ['bring to', 'carry to', 'move to']
        }

        # Action parameters schema
        self.parameter_schema = {
            'move': ['target_location', 'target_coordinates'],
            'grasp': ['object_name', 'object_color', 'object_size'],
            'place': ['object_name', 'target_location'],
            'follow': ['target_person'],
            'transport': ['object_name', 'target_location']
        }

    def ground_neural_output(self, neural_output: Dict[str, torch.Tensor], command: str):
        """Ground neural output in symbolic form"""
        # Convert neural action logits to action type
        action_probs = torch.softmax(neural_output['action_logits'], dim=-1)
        action_id = torch.argmax(action_probs, dim=-1).item()

        # Map to action name (simplified mapping)
        action_names = list(self.action_templates.keys())
        if action_id < len(action_names):
            action_type = action_names[action_id]
        else:
            action_type = 'unknown'

        # Extract parameters based on command content
        parameters = self._extract_parameters(command, action_type)

        return {
            'action_type': action_type,
            'parameters': parameters,
            'confidence': action_probs[0][action_id].item()
        }

    def _extract_parameters(self, command: str, action_type: str) -> Dict[str, Any]:
        """Extract parameters from command using rule-based approach"""
        command_lower = command.lower()
        params = {}

        # Extract object names (simple approach)
        object_keywords = ['cup', 'box', 'bottle', 'book', 'ball', 'object']
        for keyword in object_keywords:
            if keyword in command_lower:
                params['object_name'] = keyword
                break

        # Extract colors
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
        for color in colors:
            if color in command_lower:
                params['object_color'] = color
                break

        # Extract locations
        locations = ['table', 'kitchen', 'room', 'desk', 'shelf']
        for location in locations:
            if location in command_lower:
                params['target_location'] = location
                break

        # Extract target person for follow commands
        if 'me' in command_lower or 'my' in command_lower:
            params['target_person'] = 'user'

        return params

class NeuralSymbolicPlanner:
    """Hybrid neural-symbolic planner"""
    def __init__(self):
        self.neural_encoder = NeuralLanguageEncoder()
        self.symbolic_planner = SymbolicActionPlanner()

        # World model (simplified)
        self.world_model = {
            'objects': [
                {'name': 'red cup', 'type': 'cup', 'color': 'red', 'location': [1.0, 0.5, 0.0]},
                {'name': 'blue box', 'type': 'box', 'color': 'blue', 'location': [2.0, 1.0, 0.0]},
            ],
            'locations': [
                {'name': 'table', 'coordinates': [1.5, 0.5, 0.0]},
                {'name': 'kitchen', 'coordinates': [3.0, 2.0, 0.0]},
            ]
        }

    def plan_from_command(self, command: str) -> Dict[str, Any]:
        """Plan actions from natural language command using hybrid approach"""
        # Step 1: Neural processing
        neural_output = self.neural_encoder(command)

        # Step 2: Ground in symbolic form
        symbolic_command = self.symbolic_planner.ground_neural_output(neural_output, command)

        # Step 3: Generate executable plan
        plan = self._generate_executable_plan(symbolic_command)

        return {
            'command': command,
            'neural_output': neural_output,
            'symbolic_command': symbolic_command,
            'plan': plan,
            'confidence': symbolic_command['confidence']
        }

    def _generate_executable_plan(self, symbolic_command: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate executable action plan from symbolic command"""
        action_type = symbolic_command['action_type']
        params = symbolic_command['parameters']

        # Generate plan based on action type
        if action_type == 'move':
            target_location = params.get('target_location', 'unknown')
            # Find coordinates for the location
            location_coords = self._get_location_coordinates(target_location)

            return [
                {
                    'action': 'navigate',
                    'target': location_coords,
                    'description': f'Move to {target_location}'
                }
            ]

        elif action_type == 'grasp':
            object_name = params.get('object_name', 'unknown')
            object_color = params.get('object_color')

            # Find object in world model
            target_object = self._get_object_info(object_name, object_color)

            if target_object:
                return [
                    {
                        'action': 'navigate',
                        'target': target_object['location'],
                        'description': f'Move to {object_name}'
                    },
                    {
                        'action': 'grasp',
                        'target_object': target_object['name'],
                        'description': f'Grasp {object_name}'
                    }
                ]
            else:
                return [
                    {
                        'action': 'search',
                        'target_object': object_name,
                        'description': f'Search for {object_name}'
                    }
                ]

        elif action_type == 'place':
            object_name = params.get('object_name', 'unknown')
            target_location = params.get('target_location', 'unknown')

            # Find location coordinates
            location_coords = self._get_location_coordinates(target_location)

            return [
                {
                    'action': 'navigate',
                    'target': location_coords,
                    'description': f'Move to {target_location} to place {object_name}'
                },
                {
                    'action': 'place',
                    'target_object': object_name,
                    'target_location': target_location,
                    'description': f'Place {object_name} at {target_location}'
                }
            ]

        elif action_type == 'transport':
            object_name = params.get('object_name', 'unknown')
            object_color = params.get('object_color')
            target_location = params.get('target_location', 'unknown')

            # Find object
            target_object = self._get_object_info(object_name, object_color)
            location_coords = self._get_location_coordinates(target_location)

            plan = []

            # Navigate to object
            if target_object:
                plan.append({
                    'action': 'navigate',
                    'target': target_object['location'],
                    'description': f'Move to {object_name}'
                })
                # Grasp object
                plan.append({
                    'action': 'grasp',
                    'target_object': target_object['name'],
                    'description': f'Grasp {object_name}'
                })

            # Navigate to target location
            plan.append({
                'action': 'navigate',
                'target': location_coords,
                'description': f'Move to {target_location}'
            })

            # Place object
            plan.append({
                'action': 'place',
                'target_object': object_name,
                'target_location': target_location,
                'description': f'Place {object_name} at {target_location}'
            })

            return plan

        else:
            # Default plan for unknown actions
            return [
                {
                    'action': 'unknown',
                    'description': f'Unknown action: {action_type}',
                    'original_command': symbolic_command['action_type']
                }
            ]

    def _get_location_coordinates(self, location_name: str) -> List[float]:
        """Get coordinates for a named location"""
        for location in self.world_model['locations']:
            if location['name'].lower() == location_name.lower():
                return location['coordinates']
        return [0.0, 0.0, 0.0]  # Default coordinates

    def _get_object_info(self, object_name: str, color: str = None) -> Dict[str, Any]:
        """Get information about an object"""
        for obj in self.world_model['objects']:
            if object_name in obj['name'].lower():
                if color is None or color in obj['name'].lower():
                    return obj
        return None

# Example usage
def demo_neural_symbolic_planning():
    """Demonstrate neural-symbolic planning"""
    planner = NeuralSymbolicPlanner()

    # Test commands
    commands = [
        "pick up the red cup",
        "place the cup on the table",
        "move to the kitchen",
        "bring the blue box to the table"
    ]

    for command in commands:
        print(f"\nProcessing command: '{command}'")

        # Plan using hybrid approach
        result = planner.plan_from_command(command)

        print(f"Action type: {result['symbolic_command']['action_type']}")
        print(f"Parameters: {result['symbolic_command']['parameters']}")
        print(f"Confidence: {result['confidence']:.3f}")

        print("Generated plan:")
        for i, action in enumerate(result['plan']):
            print(f"  {i+1}. {action['description']}")
            print(f"     Action: {action['action']}")

if __name__ == "__main__":
    demo_neural_symbolic_planning()
```

### Example 3: Context-Aware Planning System

A planning system that maintains context and handles multi-turn interactions:

```python
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

class ContextManager:
    """Manages conversation and world context for planning"""
    def __init__(self):
        self.conversation_history = []
        self.world_state = {
            'robot_position': [0, 0, 0],
            'held_object': None,
            'object_locations': {},
            'recent_observations': []
        }
        self.user_preferences = {}
        self.context_variables = {}

    def update_world_state(self, observation: Dict[str, Any]):
        """Update world state with new observations"""
        self.world_state['recent_observations'].append({
            'timestamp': datetime.now().isoformat(),
            'data': observation
        })

        # Keep only recent observations (last 10)
        self.world_state['recent_observations'] = self.world_state['recent_observations'][-10:]

    def get_context_for_command(self, command: str) -> Dict[str, Any]:
        """Get relevant context for processing a command"""
        return {
            'world_state': self.world_state,
            'conversation_history': self.conversation_history[-5:],  # Last 5 exchanges
            'user_preferences': self.user_preferences,
            'current_time': datetime.now().isoformat()
        }

    def add_conversation_turn(self, user_input: str, system_response: str):
        """Add a turn to the conversation history"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'system': system_response
        })

        # Keep only recent history (last 10 turns)
        self.conversation_history = self.conversation_history[-10:]

class ActionRefinementSystem:
    """Refines planned actions based on context and constraints"""
    def __init__(self):
        self.safety_constraints = [
            'avoid obstacles',
            'maintain safe distance',
            'respect personal space'
        ]
        self.execution_constraints = [
            'reachable locations',
            'graspable objects',
            'valid action sequences'
        ]

    def refine_plan(self, plan: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refine a plan based on context and constraints"""
        refined_plan = []

        for action in plan:
            # Check safety constraints
            safe_action = self._apply_safety_constraints(action, context)
            if safe_action:
                # Check execution constraints
                executable_action = self._apply_execution_constraints(safe_action, context)
                if executable_action:
                    refined_plan.append(executable_action)

        return refined_plan

    def _apply_safety_constraints(self, action: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply safety constraints to an action"""
        # For navigation actions, check for obstacles
        if action['action'] == 'navigate':
            target = action.get('target', [0, 0, 0])
            robot_pos = context['world_state']['robot_position']

            # Check if path is clear (simplified)
            # In a real system, this would check against a map
            if self._is_path_clear(robot_pos, target, context):
                return action
            else:
                # Modify action to avoid obstacles
                safe_path = self._find_safe_path(robot_pos, target, context)
                if safe_path:
                    action['target'] = safe_path[-1]  # Use last safe point
                    return action

        return action  # Return original if no safety issues

    def _apply_execution_constraints(self, action: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply execution constraints to an action"""
        # Check if object is graspable for grasp actions
        if action['action'] == 'grasp':
            obj_name = action.get('target_object')
            if obj_name:
                if self._is_object_graspable(obj_name, context):
                    return action
                else:
                    # Object not graspable, return None to skip this action
                    return None

        return action

    def _is_path_clear(self, start: List[float], end: List[float], context: Dict[str, Any]) -> bool:
        """Check if path is clear of obstacles (simplified)"""
        # In a real system, this would check against a map
        # For now, assume path is clear
        return True

    def _find_safe_path(self, start: List[float], end: List[float], context: Dict[str, Any]) -> Optional[List[List[float]]]:
        """Find a safe path around obstacles (simplified)"""
        # In a real system, this would use path planning algorithms
        # For now, return direct path
        return [end]

    def _is_object_graspable(self, obj_name: str, context: Dict[str, Any]) -> bool:
        """Check if an object is graspable"""
        # In a real system, this would check object properties
        # For now, assume all objects are graspable
        return True

class ContextAwarePlanner:
    """Context-aware language-to-action planner"""
    def __init__(self):
        self.context_manager = ContextManager()
        self.action_refinement = ActionRefinementSystem()
        self.symbolic_planner = SymbolicPlanner()  # From previous example

    def plan_with_context(self, command: str) -> Dict[str, Any]:
        """Plan actions with full context awareness"""
        # Get current context
        context = self.context_manager.get_context_for_command(command)

        # Plan initial actions
        initial_plan = self.symbolic_planner.plan_actions(command)

        # Refine plan based on context
        refined_plan = self.action_refinement.refine_plan(initial_plan, context)

        # Create result
        result = {
            'command': command,
            'initial_plan': [a.description for a in initial_plan],
            'refined_plan': [a.description for a in refined_plan],
            'context_used': {
                'world_state': context['world_state'],
                'conversation_turns': len(context['conversation_history'])
            }
        }

        # Add to conversation history
        self.context_manager.add_conversation_turn(
            command,
            f"Planned {len(refined_plan)} actions: {[a.description for a in refined_plan]}"
        )

        return result

    def update_world_with_observation(self, observation: Dict[str, Any]):
        """Update world state with new observations"""
        self.context_manager.update_world_state(observation)

    def get_current_context(self) -> Dict[str, Any]:
        """Get current context information"""
        return self.context_manager.get_context_for_command("")

# Example usage
def demo_context_aware_planning():
    """Demonstrate context-aware planning"""
    planner = ContextAwarePlanner()

    # Simulate a conversation
    commands = [
        "pick up the red cup",  # First command
        "place it on the table",  # Follow-up command (refers to "it")
        "move to the kitchen",  # Simple navigation
        "bring the blue box to me"  # Complex command with "me" reference
    ]

    for i, command in enumerate(commands):
        print(f"\n--- Turn {i+1}: '{command}' ---")

        # Plan with context
        result = planner.plan_with_context(command)

        print(f"Initial plan: {result['initial_plan']}")
        print(f"Refined plan: {result['refined_plan']}")
        print(f"Context used {result['context_used']['conversation_turns']} previous turns")

        # Simulate execution
        if result['refined_plan']:
            print("Executing refined plan...")
            # In a real system, this would execute the plan
            # For simulation, just update world state
            planner.update_world_with_observation({
                'action_executed': result['refined_plan'][-1] if result['refined_plan'] else 'none',
                'timestamp': datetime.now().isoformat()
            })
        else:
            print("No executable plan generated")

if __name__ == "__main__":
    demo_context_aware_planning()
```

## System Integration Perspective

Language-to-action planning systems require integration across multiple system components:

**Natural Language Understanding**: Processing and interpreting user commands:
- Speech-to-text conversion
- Syntactic and semantic parsing
- Entity and relation extraction
- Intent classification

**World Modeling**: Maintaining and updating environmental knowledge:
- Object detection and tracking
- Spatial relationship mapping
- Dynamic environment updates
- Uncertainty representation

**Action Planning**: Generating executable action sequences:
- Task decomposition and scheduling
- Constraint satisfaction
- Resource allocation
- Plan validation and verification

**Execution Control**: Managing plan execution and adaptation:
- Low-level motion control
- Execution monitoring
- Failure detection and recovery
- Plan adjustment based on feedback

**Human-Robot Interaction**: Managing the interaction loop:
- Confirmation and clarification requests
- Progress reporting
- Error handling and recovery
- Multi-turn dialogue management

## Summary

- Language-to-action planning bridges natural language and robot execution
- Symbolic approaches provide interpretability and constraint handling
- Neural approaches offer flexibility and learning capabilities
- Context awareness enables more sophisticated interactions
- System integration requires coordination across multiple components

## Exercises

1. **Planning Architecture**: Design a planning architecture that can handle both simple and complex multi-step instructions. What components would you include for scalability?

2. **Ambiguity Resolution**: How would you handle ambiguous references like "that one" or "it" in natural language commands? Design a resolution system.

3. **Context Management**: Design a context management system that can maintain information across multiple interactions. What information would you track?

4. **Error Recovery**: How would your planning system handle execution failures? Design a recovery mechanism.

5. **Multi-Modal Integration**: How would you integrate visual information (object recognition, spatial understanding) with language understanding for more robust planning?
