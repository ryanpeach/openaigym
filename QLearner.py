# ref: http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
# ref: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
# ref: Playing Atari with Deep Reinforcement Learning

import numpy as np
import tensorflow as tf
from lib.nn import *

class QNetwork(object):
    def __init__(self, states_n, actions_n, learn_rate = 1e-6, save_path = None, debug = True):
        self.states_n, self.actions_n = states_n, actions_n
        self.path = save_path
        savefile = self._get_savefile()
        self.debug, self.T = debug, 0
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Running Info
            # Inputs Required
            self._state_input = tf.placeholder(tf.float32, shape=(None, states_n))
            
            # Get the Q output vector-list given the state
            self._q_output, _ = ffBranch(self._state_input, [256, 512, 256, actions_n])
            
            # Training Info
            # Inputs Required
            self._expected_reward_input = tf.placeholder(tf.float32, shape=(None, actions_n))      # Used to input the predicted reward based on the NN given the next state
            
            # Calculation of expected vs predicted values
            # Need to mask q value out by action input
            
            # Loss and optimize
            self._loss = (self._q_output - self._expected_reward_input)**2.     # Seeks the convergance of the predicted Q given this state, and the expected Q given the Q formula and real rewards
            self._optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self._loss)
            self._init_op = tf.initialize_all_variables()
            self._saver = tf.train.Saver()
            
            self._sess = tf.Session()
            if savefile is None: self._sess.run(self._init_op)
            else: self.load(savefile)
    
    def train(self, states, action_mask, expected_rewards):
        _, loss, _ = self._sess.run([self._q_output, self._loss, self._optimizer],
                                     feed_dict={ self._state_input      : states,
                                                 self._expected_reward_input : expected_rewards})
        #print("Loss: {}".format(loss))
        self.T += 1
        return loss
        
    def run(self, states):
        """ Given a state, returns the q for each possible action,
            the index of the action with the best expected reward,
            and the value of the expected reward """
        if self.debug: print("State: {}".format(state))
        q_vec = np.vstack(self._sess.run([self._q_output], 
                                feed_dict={self._state_input: states}))
        assert not np.any(np.isnan(q_vec)), "Q Vector includes NaN values."
        
        # Get the best (highest reward) action
        # Index q_vec by the action (Tested: Works)
        expected_action = np.argmax(q_vec, axis=1)
        expected_reward = q_vec[np.arange(len(q_vec)),expected_action]
        
        if self.debug: print("Q: {}, \neA: {}, \neR: {} \n-------".format(q_vec, expected_action, expected_reward))
        return q_vec, expected_action, expected_reward
                                                 
    def save(self):
        self._saver.save(self._sess, self.path, global_step=self.T)
    def load(self, savefile):
        self._saver.restore(self._sess, savefile)
    def _get_savefile(self):
        """ Gets the newest savefile from self.path.
            Returns None if no checkpoints in directory."""
        files = []
        for file in os.listdir(self.path):
            if file.endswith(".ckpt"):
                files.append(file)
        if len(files) == 0:
            return None
        else:
            return list(sorted(files))[0]
            
class QLearner(object):
    """ A network-agnostic Q Learner """
    def __init__(self, states_n, actions_n, network,
                       future_weight = .90, memsize = 1e6):
        # Set constants
        self.MEMORY  = RollingMemory(memsize)
        self.NSTATES  = states_n
        self.NACTIONS = actions_n
        self.future_weight = future_weight
        self.Network = network

    # Outside Callable Functions
    def record(self, state, action, next_state, reward, done = False):
        self.MEMORY.add((state, action, reward, next_state, done))

    def action_randomize(self, action_index, conf = None):
        """ Select random action if dice comes up greater than confidence. """
        if conf is None: conf = self.confidence_time_decay()
        
        if np.random.rand() < conf:
            return action_index
        else:
            out = np.random.randint(0, self.NACTIONS)
            #print("Random {}, Conf: {}".format(out, conf))
            return out

    def train_step(self, Ns):
        """ Runs one single training step on network based on random sample from memory.
            Parameters:
              Ns   : int; The number of memory samples to train off of. Automatically reduced if not enough in memory.
              
            Returns:
              loss : vector; The list of losses for each sample. """
              
        # Get a random sample of the memory
        if Ns > len(self.MEMORY):
            Ns = len(self.MEMORY)
        mem_sample = random.sample(list(self.MEMORY.Mem), Ns)    # Get random sample of indexes
        states, action_index, rewards, results, terminal = zip(*mem_sample)
        action_mask = (np.arange(self.NACTIONS)[np.newaxis,:] == np.array(action_index)[:,np.newaxis])
        states, results = np.vstack(states), np.vstack(results)
        
        # Get predicted rewards for each result as is
        rewards_calculated, _, _  = self.Network.run(states)
        _, _, reward_predictions  = self.Network.run(results)
        expected_rewards = []
        for i in range(Ns):
            a = rewards_calculated[i]
            if terminal[i]:
                a[action_index[i]] = rewards[i]
            else:
                a[action_index[i]] = rewards[i] + self.future_weight * reward_predictions[i]
            expected_rewards.append(a) # This is the Bellman Equation
        
        return self.Network.train(states, action_mask, expected_rewards)
        
        # Print & Save   
        # self.train_writer.add_summary(summ, self.T)
        # save checkpoints for later
        # if self.T % self.save_freq == 0:
        #     self._saver.save(self._session, self._path + '/network', global_step=self.T)

class RealtimeQLearner(QLearner):
    def __init__(self, states, actions, Q_network, 
                 learn_rate = 1e-7, future_weight = .90, memsize = 1e6,
                 start_conf = .1, end_conf = .9, conf_period = 1e6):
       super(RealtimeQLearner, self).__init__(states, actions, Q_network, future_weight, memsize)
       self.last_state = None
       self.last_action = None
       assert start_conf >= 0. and end_conf <= 1.0 and start_conf < end_conf, "0. <= a < b <= 1.0"             # A must be less that b and both must be between 0. and 1. inclusive
       self.start_conf, self.end_conf, self.conf_period = start_conf, end_conf, conf_period
    
    def confidence_training_decay(self):
        """ Generate a confidence linearly from start_conf to end_conf over conf_period iterations. 
            Returns:
              conf : float; The current confidence value based on training iterations. """
        m = (self.end_conf-self.start_conf)/self.conf_period                    # Slope is positive over the period
        if self.Network.T < self.conf_period:
            conf = m*self.Network.T + self.start_conf
        else:
            conf = self.end_conf                                                # Confidence is capped at end_conf
        #if self.T % 100 == 0:
        #    print("Confidence: {0}".format(conf))
        return conf
       
    def act(self, state, reward, done = False, randomize = True):
        """ Generates an action from the given state, and records past states.
            Must be run in order with the environment.
            Sometimes produces random actions dependent on the training confidence.
            
            Parameters:
              state  : vector; The state vector for the current state.
              reward : float;  The reward from going from self.last_state to state.
              done   : bool;   Whether or not this is a terminal state for an episodic learner.
            
            Returns:
              action_index      : int;   The index of the action selected between 0 and self.NACTIONS.
              action_confidence : float; The probability between 0 to 1 of this being the best action given all prior knowledge.
            """
        # Update memory given last and current state/reward
        if self.last_state is not None and self.last_action is not None and reward is not None:
            self.record(self.last_state, self.last_action, state, reward, done)
        
        state = np.array(state)
        # Get the predicted best action given the state
        q_vec, action_index, expected_reward = self.Network.run(state)
        
        #print("Q: {}, Ai: {}, eR: {}".format(q_vec, action_index, expected_reward))
        q_vec, action_index, expected_reward = q_vec[0], action_index[0], expected_reward[0] # We are working in a single item batch
        
        # Get the confidence
        # Randomize action based on training confidence
        training_confidence = self.confidence_training_decay()
        if randomize:
            action_index = self.action_randomize(action_index, conf = training_confidence)
        #print(q_vec, action_index, expected_reward)
        # Muffle the expected reward softmax intensity based on the training decay confidence
        #print(q_vec, training_confidence, softmax(q_vec*training_confidence, axis=1))
        action_confidence = softmax(q_vec*training_confidence)[action_index]
        
        # Update needed memory for next action
        if not done:
            self.last_state = state
            self.last_action = action_index
        else: # Erase episodically
            self.last_state = None
            self.last_action = None
            
        return action_index, action_confidence
        
if __name__=="__main__":
    pass
