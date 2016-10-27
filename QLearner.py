# ref: http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
# ref: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
# ref: Playing Atari with Deep Reinforcement Learning

import numpy as np
import tensorflow as tf
from nn import *

class QLearner(object):
    """ A network-agnostic Q Learner """
    def __init__(self, states, actions, Q_network,
                       learn_rate = 1e-6, future_weight = .99, memsize = 1e6):
        # Set constants
        self.MEMORY  = RollingMemory(memsize)
        self.STATES  = States
        self.ACTIONS = Actions
        self.future_weight = future_weight
        self.learn_rate    = learn_rate

        # Create the network based on the Q_Network given
        self._create_network(Q_network)
        self.T = 0  # The number of training steps taken
    
    def _create_network(self, Q_network):
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Running Info
            # Inputs Required
            self._state_input = tf.placeholder(tf.float32, shape=(None, len(self.STATES)))
            
            # Get the Q output vector-list given the state
            self._q_output = Q_network(self._state_input, len(self.STATES), len(self.ACTIONS), self.learn_rate)
            
            # Training Info
            # Inputs Required
            self._action_input = tf.placeholder(tf.int, shape(None, 1))                # Index of the action taken
            self._expected_reward_input = tf.placeholder(tf.float32, shape=(None, 1))  # Used to input the predicted reward based on the NN given the next state
            
            # Calculation of expected vs predicted values
            self._predicted_reward = self._q_output[self._action_input]                # The reward based on input state data and action taken
            
            # Loss and optimize
            self._loss = (self._predicted_reward - self._expected_reward_input)**2.    # Seeks the convergance of the predicted Q given this state, and the expected Q given the Q formula and real rewards
            self._optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self._loss)

    # Outside Callable Functions
    def record(self, state, action, next_state, reward, done = False):
        self.MEMORY.add((state, action, reward, next_state, done))
    
    def Q(self, state):
        """ Given a state, returns the q for each possible action,
            the index of the action with the best expected reward,
            and the value of the expected reward """
        q_vec = self._session.run([self._q_output], 
                                   feed_dict={self._state_input: state})
        expected_action = np.argmax(q_vec)
        expected_reward = q_vec[expected_action]
        return q_vec, expected_action, expected_reward
    
    def action_randomize(self, action_index, conf = None):
        """ Select random action if dice comes up greater than confidence. """
        if conf is None: conf = self.confidence_time_decay()
        
        if np.random.rand() > conf:
            return action_index
        else:
            return np.random.randint(0, self.Na)

    def train_step(self, Ns):
        """ Runs one single training step on network based on random sample from memory.
            Parameters:
              Ns   : int; The number of memory samples to train off of. Automatically reduced if not enough in memory.
              
            Returns:
              loss : vector; The list of losses for each sample. """
              
        # Get a random sample of the memory
        if Ns > len(self.memory):
            Ns = len(self.memory)
        mem_sample = random.sample(list(self.memory.Mem), Ns)    # Get random sample of indexes
        states, actions, rewards, results, terminal = zip(*mem_sample)
            
        # Get predicted rewards for each result as is
        _, _, reward_predictions = Q(self, states)
        expected_rewards = []
        for i in range(Ns):
            if terminal[i]:
                expected_rewards.append(rewards[i])
            else:
                expected_rewards.append(rewards[i] + self.future_weight * 
                    np.max(reward_predictions[i], axis = 1)) # This is the Bellman Equation
            
        # learn that these actions in these states lead to these rewards
        _, _, loss, _ = self._session.run([self._q_output, self._loss, self._optimizer],
                                           feed_dict={ self._state_input      : states,
                                                       self._action_input     : actions,
                                                       self._expected_reward_input : expected_rewards })
        
        self.T += 1 # Update training count
        return loss
        
        # Print & Save   
        # self.train_writer.add_summary(summ, self.T)
        # save checkpoints for later
        # if self.T % self.save_freq == 0:
        #     self._saver.save(self._session, self._path + '/network', global_step=self.T)

class RealtimeQLearner(QLearner):
    def __init__(self, states, actions, Q_network, 
                 learn_rate = 1e-6, future_weight = .99, memsize = 1e6,
                 start_conf = .1, end_conf = .9, conf_period = 1e6):
       super(RealtimeQLearner, self).__init__(states, actions, Q_network, learn_rate, future_weight, memsize)
       self.last_state = None
       self.last_action = None
       self.start_conf, self.end_conf, self.conf_period = start_conf, end_conf, conf_period
    
    def confidence_training_decay(self):
        """ Generate a confidence linearly from start_conf to end_conf over conf_period iterations. 
            Returns:
              conf : float; The current confidence value based on training iterations. """
        assert a >= 0. and b <= 1.0 and a < b, "0. <= a < b <= 1.0"             # A must be less that b and both must be between 0. and 1. inclusive
        m = (self.end_conf-self.start_conf)/self.conf_period                    # Slope is positive over the period
        if self.T < self.conf_period:
            conf = m*self.T + self.start_conf
        else:
            conf = self.end_conf                                                # Confidence is capped at end_conf
        if self.MEMORY.N % 100 == 0:
            print("Confidence: {0}".format(conf))
        return conf
       
    def act(self, state, reward, done = False):
        """ Generates an action from the given state, and records past states.
            Must be run in order with the environment.
            Sometimes produces random actions dependent on the training confidence.
            
            Parameters:
              state  : vector; The state vector for the current state.
              reward : float;  The reward from going from self.last_state to state.
              done   : bool;   Whether or not this is a terminal state for an episodic learner.
            
            Returns:
              action_index      : int;   The index of the action selected in self.ACTIONS.
              action_confidence : float; The probability between 0 to 1 of this being the best action given all prior knowledge.
            """
        # Update memory given last and current state/reward
        if self.last_state is not None and self.last_reward is not None:
            self.record(self.last_state, self.last_action, state, reward, done)
    
        # Get the predicted best action given the state
        q_vec, action_index, expected_reward = self.Q(state)

        # Get the confidence
        # Randomize action based on training confidence
        training_confidence = confidence_training_decay()
        action_index = self.action_randomize(action_index, conf = training_confidence)
        
        # Muffle the expected reward softmax intensity based on the training decay confidence
        action_confidence = softmax(expected_reward*training_confidence)[action_index]
        
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
