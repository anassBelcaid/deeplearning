---
type: assignment
date: 2024-12-21T4:00:00+4:30
title: 'Assignment #3 - Reinforcement Learning'
# pdf: /assign1/
#attachment: /assignments/assignment2_colab.zip/
# solutions: /static_files/assignments/asg_solutions.pdf
due_event: 
    type: due
    date: 2024-12-15T23:59:00+3:30
    description: 'Assignment #2 due'
---


# Reinforcement Learning Homework: Q-Learning with OpenAI Gym

## Objective
The primary goal of this homework assignment is to gain hands-on experience with:
- OpenAI Gym environments
- Implementing basic reinforcement learning algorithms
- Specifically focusing on Value Iteration and Q-Learning techniques

## Background
Reinforcement Learning (RL) is a machine learning approach where an agent learns to make decisions by interacting with an environment. The key components include:
- **Agent**: The decision-maker
- **Environment**: The world in which the agent operates
- **Actions**: Possible moves the agent can make
- **States**: Different situations the agent can be in
- **Rewards**: Feedback mechanism to guide learning

## Setup Requirements
### Prerequisites
- Python 3.7+
- NumPy
- OpenAI Gym
- Matplotlib (for visualization)

### Installation
```bash
pip install gym numpy matplotlib
```

## Assignment Components

### Part 1: Environment Exploration
1. Choose an OpenAI Gym environment (recommended: CartPole-v1 or FrozenLake-v1)
2. Implement functions to:
   - Initialize the environment
   - Render the environment
   - Understand state and action spaces

### Part 2: Value Iteration
- Implement the Value Iteration algorithm
- Key steps:
  1. Initialize value function
  2. Iteratively update value estimates
  3. Compute optimal policy
- Convergence criteria and maximum iterations

### Part 3: Q-Learning Implementation
- Develop a Q-Learning algorithm
- Core implementation requirements:
  - Q-table initialization
  - Epsilon-greedy action selection
  - Q-value update rule
  - Learning rate and discount factor exploration

## Deliverables
1. Commented Python script with implementations
2. Visualization of learning progress
3. Short report explaining:
   - Algorithm details
   - Performance analysis
   - Challenges encountered

## Evaluation Criteria
- Correct implementation of algorithms
- Code readability and documentation
- Visualization of learning process
- Understanding of RL concepts demonstrated

## Bonus Challenges
- Experiment with different exploration strategies
- Compare performance across multiple environments
- Implement function approximation for Q-Learning

## Recommended Resources
- OpenAI Gym Documentation
- Sutton & Barto's Reinforcement Learning: An Introduction
- David Silver's RL Course

**Note**: Focus on understanding the core concepts and iterative nature of reinforcement learning algorithms.

