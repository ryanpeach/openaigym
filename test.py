from envs.aigym import *
from QLearner import *
from qffnetwork import *

# Get the envrionment
import gym
env = gym.make('CartPole-v0')

# Create the agent
states = env.observation_space
actions = env.action_space
print("States: {}, Actions: {}".format(states, actions))
agent = RealtimeQLearner(states, actions, q_ff_network, 
                 learn_rate = 1e-6, future_weight = .99, memsize = 1e6,
                 start_conf = .1, end_conf = .9, conf_period = 1e6)

# Create the learning loop
n_episodes = 20
t_max = 100
training_steps_per_episode = 100
number_samples = t_max

# Over n_episodes episodes
for i_episode in range(n_episodes):
    observation = env.reset()  # Each episode reset the environment and get the first observation
    reward      = None         # Initialize reward to None
    done        = False        # Initialize done to False
    
    # Each episode lasting at maximum t_max iterations
    for t in range(t_max):
        # Display state
        env.render()
        print("Observation t={}: {}".format(t,observation))
        
        # Select action
        action, _ = agent.act(observation, reward, done)
        
        # Get new observation and reward
        observation, reward, done, info = env.step(action)
        
        if done:
            agent.act(observation, reward, done) # A dummy action to record the last reward and done variable
            print("Episode finished after {} timesteps".format(t+1))
            break
        
    # Train after each episode
    print("Training")
    for i in range(training_steps_per_episode):
        # Print for debugging
        if i % (training_steps_per_episode // 10) == 0:
            print("Training Step: {}/{}".format(i, training_steps_per_episode))
        
        # Train agent once
        agent.train_step(number_samples)