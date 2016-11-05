from QLearner import *
from lib.nn import hotone

# Get the envrionment
import gym
import scipy.optimize
env = gym.make('CartPole-v0')
assert isinstance(env.observation_space, gym.spaces.box.Box)
assert isinstance(env.action_space, gym.spaces.discrete.Discrete)
states_n = len(env.observation_space.high)
actions_n = env.action_space.n
goal_per_100 = 195

def tuning(learn_rate, future_weight = .9, training_period = 1000, testing_period = 200, debug=True):
    learn_rate = float(learn_rate)
    net = QNetwork(states_n, actions_n, learn_rate, debug=False)
    
    # Create the agent
    agent = RealtimeQLearner(states_n, actions_n, net, 
                     learn_rate = learn_rate, future_weight = future_weight, memsize = int(1e3),
                     start_conf = 0.0, end_conf = 1.0, conf_period = int(training_period-training_period*.1))
    
    # Run the training
    #print(" ------ Training... ------ ")
    try:
        assert future_weight < 1., "Future Weight must be less than 1.0"
        train_rewards = run(agent, training_period, 300, 1, 200, debug=debug)
        agent.end_conf = 1.0
    
        # Run the testing
        #print(" ------ Testing... ------ ")
        test_rewards = run(agent, testing_period, 300, 0, 0, debug=debug)
        score = np.mean(test_rewards)
    except Exception as e:
        print(e)
        score = 0.
        train_rewards = [0.]
        
    #if debug: print("Final Q: {}".format(net.run(states)[0]))
    print("Training Rewards: {}, Learn Rate: {}, Future Weight: {}, Score: {}".format(sum(train_rewards), learn_rate, future_weight, score))
    return score
    
def run(agent, n_episodes, t_max, training_steps_per_episode, number_samples, debug=True):
    # Over n_episodes episodes
    total = 0
    all_rewards = []
    for i_episode in range(n_episodes):
        observation = env.reset()  # Each episode reset the environment and get the first observation
        observation = np.array([observation]) # Normalize and put in proper 2d format
        reward      = None         # Initialize reward to None
        done        = False        # Initialize done to False
        
        # Render every n episodes
        if debug:
            if i_episode % 100 == 0:
                render = True
            else:
                render = False
        else:
            render = False
        
        # Each episode lasting at maximum t_max iterations
        if render: print("====== Episode {} ======".format(i_episode))
        episode_total_reward = 0
        for t in range(t_max):
            # Display state
            #if render: env.render()
            if render: print("Observation t={}: {}".format(t,observation))
            
            # Select action
            #print(observation, observation.shape)
            action, conf = agent.act(observation, reward, done)
            if render: print("Action: {}, Confidence: {}".format(action, conf))
            
            # Get new observation and reward
            observation, reward, done, info = env.step(action)
            observation = np.array([observation]) # Normalize and put in proper 2d format
            episode_total_reward += reward
            total += reward
            if done:
                reward = 0
                agent.act(observation, reward, done) # A dummy action to record the last reward and done variable
                if render: print("Episode finished after {} timesteps".format(t+1))
                break
        
        # Record Reward
        if render: print("Episode {}, Reward: {}, Total: {}".format(i_episode, episode_total_reward, total))
        all_rewards.append(episode_total_reward)
           
        # Train after each episode
        if total > 0:  # Training diverges to infinity when training without any positive samples
            if render: print("Training")
            for i in range(training_steps_per_episode):
                # Print for debugging
                if debug and render:
                    m = (training_steps_per_episode // 10)
                    if m == 0 or i % m == 0:
                        print("Training Step: {}/{}".format(i, training_steps_per_episode))
                    
                # Train agent once
                agent.train_step(number_samples)
            
    if debug: print("Rewards: {}".format(all_rewards))
    return all_rewards

# Settings
init_period, max_period, dp = 500, 1000, 100  # The smallest period to test at
tune_free_parameters = True
done = False
guess = np.array([2e-3, .9])

# Main loop
while not done:
    period = init_period # The amount of training_period we will use in tuning
    
    # Minimize free parameters: learn_rate, future_weight
    direc = guess*.5 # Initial direction is half of the guess
    bounds = [(1e-5,1e-2),(.8,0.999)]
    if tune_free_parameters:
        res = scipy.optimize.minimize(lambda x: -tuning(*x, training_period=period, debug=False), guess, method='Powell', options = {'maxfev': 100, 'disp': True, 'direc':direc})
        learn_rate, future_weight = res.x
        guess = res.x # Reset guess incase this fails
    else:
        learn_rate, future_weight = guess
    
    # Starting at the smallest period, gradually increase it until the required score is met.
    score = 0.
    print("Learn Rate: {}, Future Weight: {}".format(learn_rate, future_weight))
    while score < goal_per_100:
        print("=========== Period {} ===========".format(period))
        #env.monitor.start('./output/CartPole-v0/', force=True)
        
        # Check if period is out of bounds
        if period > max_period:
            break
        
        # Tune and score
        score = tuning(learn_rate, future_weight = future_weight, training_period = period, testing_period = int(1e2), debug = False)
        
        # Check for errors
        if score == 0.:    # Indicates an Error
            print("Error") 
            break       # It worked during training, so it has to work *sometimes*
        
        period += 100  # Try a longer training period, assumes that the other parameters will still be better with longer training time
        
        print("=========== Score {} ===========".format(score))
        #env.monitor.close()
    
    # Did we succeed, or break?
    if score >= goal_per_100:
        done = True
        print("Best! Period: {}, Score: {}, Learn Rate: {}, Future Weight: {}".format(period-100, score, learn_rate, future_weight))
    else:
        init_period += dp
        max_period += dp