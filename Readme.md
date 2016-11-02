The algorithm uses a custom built Q Learning agent in Python 3.

## Dependences:
1. Python 3
2. Tensorflow
3. Numpy
4. Scipy

## Description of Classes and Functions:
QLearner.py
  * QNetwork - Contains the tensorflow network and easily usable run, train, load, and save methods.
  * QLearner - The meat of the Q learner, creates the functions specialized for training Q agents using updated network predictions.
  * RealtimeQLearner - Assumes the Q learner gets it's data in real time. Creates a rolling memory and associated functions for training.
  
test.py
  * tuning - Creates the network from scratch using the given free parameters such as future_weight and learn_rate.
           Runs a training period # of episodes where training takes place, then a verifcation or test period where no training takes place where the agent is scored.
           The score is returned in a manner suitable for running through a minimization algorithm for free parameter tuning and network evaluation.
  * run - Self explanitory, runs the network with gym. Returns the final scores for each episode in a list. Trains each step a given number of steps in the parameters, no training if that number is 0.
  
**To test, simply download and run "python3 test.py"**

Test.py will automatically tune the algorithm's free parameters based on best guess, then slowly increase the training period until the desired score is reached.

Only the training which returns the best score is returned, but it is clear that monitor is monitoring a fresh, randomized network, not one that has any pre-training.

*Only free parameters have been trained in the background.*