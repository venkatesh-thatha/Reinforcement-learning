


import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self,k_arms=10):
        self.true_probs = np.random.rand(k_arms)
        self.k_arms = k_arms
    
    def pull(self, arm_index):
        if np.random.rand()<self.true_probs[arm_index]:
            return 1
        else:
            return 0
        
env=Bandit(k_arms=5)

# print(f'Hidden True Win Rates: {np.round(env.true_probs,2)}')
# # Pull Arm 0 a few times
# print(f"Pull 1: {env.pull(0)}")
# print(f"Pull 2: {env.pull(0)}")
# print(f"Pull 3: {env.pull(0)}")

# print("\nTesting Arm 1 ten times:")
# for _ in range(10):
#     reward = env.pull(1)
#     print(f"Reward: {reward}", end=" | ")

class Agent:
    def __init__(self, k_arms=10, epsilon=0.1):
        self.k_arms = k_arms
        self.epsilon = epsilon
        self.counts = np.zeros(k_arms)
        self.values = np.zeros(k_arms)

    def choose_action(self):
        if np.random.rand()<self.epsilon:
            return np.random.randint(0,self.k_arms)
        else:
            return np.argmax(self.values)
    
    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        # Incremental update of average
        new_value = value + (reward - value) / n
        self.values[action] = new_value


agent=Agent(k_arms=5, epsilon=0.1)
total_rewards=[]

n=int(input("Enter the number of episodes: "))
for episode in range(n):
    action=agent.choose_action()
    reward=env.pull(action)
    agent.update(action,reward)
    total_rewards.append(reward)
print(f'Total Reward after {n} episodes: {sum(total_rewards)}')
print(f"True Values:      {np.round(env.true_probs, 2).tolist()}")
print(f"Estimated Values: {np.round(agent.values, 2).tolist()}")
print(f"Counts:           {agent.counts.astype(int).tolist()}")