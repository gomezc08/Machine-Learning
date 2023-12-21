"""
1. Write in 1-2 lines about each of these components:

• Environment: The world/setting through which the agent moves and learn from. The video uses a cool example if you were dropped off on an island, the island would be your environment.
    
• Agent: The RL algorithm that we are creating. The agent learns through trial and error and positive and negative reinforcements.

• State: The current conditionb returned by the environment.
    
• Actions: All the possible steps that an agent can do in the enviornment. The video uses the example of moving from room 1 to either room 3 or 5.

• Rewards: Appraise given to the agent based on what action they take in their given enviornment.
    
• Policy: How the agent determines the next best action to take.
    
• Value: Agent uses what they already know to maximize amount of rewards
    
• Action-value: The expected return if the agent chooses action a according to a policy  

2. What is the difference between Exploration and Exploitation, and how is it associated with gamma?

• Exploitation is when the agent uses what they know to guide their actions and exploration is more the agent exploring different actions in search for better ways to maximize rewards (and learn more all in all)
• Gamma reports back a value between 0 and 1 for the agent. 
    If gamma = 0, that means that the agent will likely choose the next action that they're already familiar with; more on exploitation
    If gamma = 1, the agent is more curious and wants to learn more and will likely choose an unknown kind of path; more on exploration
    
3. What is Q-Learning? How is it implemented?

•  Q-Learning is a type of RL algorithm performed on an agent in a particular enviornment, with intentions to learn what actions are the most rewarding and negative values.
•  There are a couple of important matrices for building the memory of the agent: The Reward Matrix and the Q matrix. And this is how the Q learning is implemented...
•  The reward matrix is basically, as the agent is training in their environment, the reward matrix shows the cost/reward of a particular action given your current state. 
•  The Q matrix represents the actual memory! It stores all the Q-values we've caluclated 

4. What is the significance of  this formula

Q(State, Action) = R (State, Action) + Gamma * Max [Q(next state, all actions)] 
We want to calculate the q matrix based on the action you want to take given your current state. That's represented as Q(State, Action).
R (State, Action) represents the actual value in the reward matrix if you were gonna take a certain action given your current state. 
We've already touched on gamma, but it plays a role in the level of which the agent is in the state of exploitation or exploration.
Max [Q(next state, all actions)] is the highest Q-table value based on the already calculated state and action rewards.
The goal of this is for to train the agent to predict what how to navigate around their enviornment with the highest reward as possible.
""" 
# 5. Implement the Q-learning on mentioned example in the video.

import numpy as np
rewardMatrix = np.matrix([[-1,-1,-1,-1,0,-1],
                          [-1,-1,-1,0,-1,100],
                          [-1,-1,-1,0,-1,-1],
                          [-1,0,0,-1,0,-1],
                          [-1,0,0,-1,-1,100],
                          [-1,0,-1,-1,0,100]])

qMatrix = np.matrix(np.zeros([6,6]))     # hasn't been trained so need to set values to 0

gamma = 0.8

initialState = 1

def availableActions(state):
    currentStateRow = rewardMatrix[state,]    # find row 
    actualAction = np.where(currentStateRow >= 0)[1]    # find actual action you can take (>=0; -1 = can't take that path)
    return actualAction

available_action = availableActions(initialState) 


def randomNextAction(availableActionsRange):
    return int(np.random.choice(available_action, 1))

action = randomNextAction(available_action)

# Equation...
def update(currentState, action, gamma):
    maxIndex = np.where(qMatrix[action,] == np.max(qMatrix[action,]))[1]
    if(maxIndex.shape[0] > 1):
        maxIndex = int(np.random.choice(maxIndex, size = 1))
    else:
        maxIndex = int(maxIndex)
    
    maxValue = qMatrix[action, maxIndex]
    
    qMatrix[currentState, action] = rewardMatrix[currentState, action] + gamma * maxValue


update(initialState, action, gamma)

# TRAINING TIME...

for i in range(10000):
    currentState = np.random.randint(0, int(qMatrix.shape[0]))
    available_action = availableActions(currentState)
    action = randomNextAction(available_action)
    update(currentState, action, gamma)
    
print("Trained Q matrix: ")
print(qMatrix / np.max(qMatrix) * 100)

# TESTING TIME...
currentState = 1
steps = [currentState]

while currentState != 5:
    nextStep = np.where(qMatrix[currentState, ] == np.max(qMatrix[currentState]))[1]
    if nextStep.shape[0] > 1:
        nextStep = int(np.random.choice(nextStep, size = 1))
    else:
        nextStep = int(nextStep)
    steps.append(nextStep)
    currentState = nextStep

print("Selected path: ")
print(steps)