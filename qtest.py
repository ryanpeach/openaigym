from QLearner import *
import unittest
from lib.nn import hotone, allhot

DEBUG = False

class TestQ(unittest.TestCase):

    def test_q_network(self):
        """ Tests the Q Network for running, loss, training, and copying. """
        # Initialize the network
        NSTATES, NACTIONS = 5, 2
        NETWORK = QNetwork(NSTATES, NACTIONS, learn_rate = .1, save_path = './', debug = False)
        ACTION_MAP = [0,1,0,1,0] # The mapping (one hot) from a state to a proper action
        
        # Get all states
        all_states = allhot(NSTATES)
        if DEBUG: print("All States: \n{}\n".format(all_states))
        
        # Generate the target reward mapping
        TARGET = np.array([hotone(v, NACTIONS) for v in ACTION_MAP]).astype("float32")
        if DEBUG: print("Target: \n{}\n".format(TARGET))
        
        # Get the Q Map of the network
        def get_q_map(net):
            q_vector, _, _ = net.run(all_states)
            return q_vector
            
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
        
if __name__ == '__main__':
    unittest.main()