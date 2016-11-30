from QLearner import *
import unittest
from lib.nn import hotone, allhot

# Initialize Variables
DEBUG = False

def get_q_map(net):
    q_vector, _, _ = net.run(all_states)
    return q_vector
            
# Q Learner Testing

    
# Get all states
all_states = allhot(NSTATES)
print("All States: \n{}\n".format(all_states))
        
class TestQ(unittest.TestCase):

    def test_q_network(self):
        """ Tests the Q Network for running, loss, training, and copying. """
        
        # Initialize Variables
        NSTATES, NACTIONS = 5, 2
        ACTION_MAP = [0,1,0,1,0] # The mapping (one hot) from a state to a proper action
        
        # Initialize the network
        NETWORK = QNetwork(NSTATES, NACTIONS, learn_rate = .1, save_path = './', debug = False)
        
        # Generate the target reward mapping
        TARGET = np.array([hotone(v, NACTIONS) for v in ACTION_MAP]).astype("float32")
        if DEBUG: print("Target: \n{}\n".format(TARGET))
        
        # Print the initial map
        current = get_q_map(NETWORK)
        if DEBUG: print("Initial: \n{}\n".format(current))
        
        # Get the initial difference
        prev_loss = NETWORK.loss(all_states, TARGET)
        if DEBUG: print("Initial Loss: \n{}\n".format(prev_loss))
        
        # Train some
        if DEBUG: print("Loss:")
        for i in range(10):
            next_loss = NETWORK.train(all_states, TARGET)
            if DEBUG: print(next_loss)
        
        # Was it better?
        test1 = next_loss < prev_loss
        if DEBUG: print("\nBetter? {}, {}".format(test1, prev_loss-next_loss))
        self.assertTrue(test1)
        
        # Now copy the network
        if DEBUG: print("----- Copying -----")
        N1 = NETWORK
        N2 = NETWORK.deepcopy()
        
        # Are the Q's the same
        Q1, _, _ = N1.run(all_states)
        Q2, _, _ = N2.run(all_states)
        test2 = np.all(np.isclose(Q1,Q2))
        if DEBUG: print("Q1: \n{}\n".format(Q1))
        if DEBUG: print("Q2: \n{}\n".format(Q2))
        if DEBUG: print("Successful copy? {}\n".format(test2))
        self.assertTrue(test2)
        
        # Are they indeed seperate networks?
        # Train one of them
        for i in range(10):
            next_loss = N1.train(all_states, TARGET)
            
        # Compare again
        Q1, _, _ = N1.run(all_states)
        Q2, _, _ = N2.run(all_states)
        test3 = not np.all(np.isclose(Q1,Q2))
        if DEBUG: print("Q1: \n{}\n".format(Q1))
        if DEBUG: print("Q2: \n{}\n".format(Q2))
        if DEBUG: print("Successfully Seperate? {}\n".format(test3))
        self.assertTrue(test3)
        
    def test_q_learner(self):
        # In this world there are three states, and it loops. Actions are move left and right.
        REWARD_MAP = [0,0,1] # The reward for a given state 
        RESULT_MAP = np.array([[2,1],[0,2],[1,0]]) # The mapping of what state you will go to for each action in each state
        QTARGET    = np.array([[1,0],[0,1],[0,0]]) # The ideal Q matrix (calculated by hand from the other two)
        NSTATES, NACTIONS = 3, 2
    
        def env(state, action):
            """ Given a state and action, returns a reward and new state """
            next_state = RESULT_MAP[state,action]
            reward = REWARD_MAP[next_state]
            return next_state, reward
    
        # Create the agent
        agent = QLearner(NSTATES, NACTIONS, QNetwork, future_weight = .99, memsize = 1e6)
        
        # Get the initial Q map
        Q0 = get_q_map(agent.Network)
        
        # Run all state action pairs and record them
        for s in range(NSTATES):
            for a in range(NACTIONS):
                s1, r = env(s,a)
                record(self, s, a, s1, r, done = (r == 1))
        
        # Train N number of times
        loss0 = agent.train_step(1)
        loss1 = agent.train_step(10)
        
        # Has the loss decreased
        self.assertLessThan(loss1,loss0)
        
        # Get the new network Q
        Q1 = get_q_map(agent.Network)
        
        # Has the true loss decreased?
        true_loss0 = np.mean(np.argmax(Q0,axis=1)==np.argmax(QTARGET,axis=1))
        true_loss1 = np.mean(np.argmax(Q1,axis=1)==np.argmax(QTARGET,axis=1))
        self.assertLessThan(true_loss1, true_loss0)
        
if __name__ == '__main__':
    unittest.main()