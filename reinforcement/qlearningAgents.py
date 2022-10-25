# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pickletools import UP_TO_NEWLINE
from pyexpat import features
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random
import util
import math
import copy


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        temp_qValue = util.Counter()
        temp_max = float("-inf")
        temp_action = 'None'
        if len(self.getLegalActions(state)) == 0:
            return 0.0
        for action in self.getLegalActions(state):
            qValue = self.getQValue(state, action)
            if qValue > temp_max:
                temp_max = qValue
        return temp_max

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # No legal actions
        if len(self.getLegalActions(state)) == 0:
            return None
        else:
            temp_qValue = util.Counter()
            temp_max_value = float("-inf")
            max_action_list = []
            for action in self.getLegalActions(state):
                qValue = self.getQValue(state, action)
                if qValue >= temp_max_value:
                    if qValue > temp_max_value:
                        max_action_list = []
                        max_action_list.append(action)
                        temp_max_value = qValue
                    else:
                        max_action_list.append(action)
                        # temp_max_value = temp_qValue[action]
            max_action = random.choice(max_action_list)
            return max_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        randomAction = random.choice(self.getLegalActions(state))
        bestAction = self.computeActionFromQValues(state)
        if len(self.getLegalActions(state)) == 0:
            return action
        else:
            if util.flipCoin(self.epsilon):
                return randomAction
            else:
                return bestAction

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * \
            self.computeValueFromQValues(nextState)
        updatedQ = (1 - self.alpha) * \
            self.getQValue(state, action) + self.alpha * sample
        self.qValues[(state, action)] = updatedQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        value = 0
        features = self.featExtractor.getFeatures(state, action)
        for feature in features.keys():
            value = value + (self.weights[feature] * features[feature])
            # print(self.weights[feature])
        return value

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        maxQValue = self.getValue(nextState)
        difference = (reward + self.discount *
                      maxQValue) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        # print("Features", features)

        for feature in features.keys():
            # print("BEFORE", self.weights[feature], "FEATURE:", feature)
            self.weights[feature] = self.weights[feature] + self.alpha * \
                difference * features[feature]
            # print("AFTER", self.weights[feature], "FEATURE:", feature)

        # print("Weights", self.weights)

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
# class ApproximateQAgent(PacmanQAgent):
#     """
#        ApproximateQLearningAgent
#        You should only have to overwrite getQValue
#        and update.  All other QLearningAgent functions
#        should work as is.
#     """

#     def __init__(self, extractor='IdentityExtractor', **args):
#         self.featExtractor = util.lookup(extractor, globals())()
#         PacmanQAgent.__init__(self, **args)
#         self.weights = util.Counter()

#     def getWeights(self):
#         return self.weights

#     def getQValue(self, state, action):
#         """
#           Should return Q(state,action) = w * featureVector
#           where * is the dotProduct operator
#         """
#         "*** YOUR CODE HERE ***"
#         feats = self.featExtractor.getFeatures(state, action)
#         qVal = 0.0
#         for f, v in feats.items():
#             qVal += self.weights[f] * v
#         return qVal

#     def update(self, state, action, nextState, reward: float):
#         """
#            Should update your weights based on transition
#         """
#         "*** YOUR CODE HERE ***"
#         feats = self.featExtractor.getFeatures(state, action)
#         oldQVal = self.getQValue(state, action)
#         newQVal = reward + self.discount * \
#             self.computeValueFromQValues(nextState)
#         update = self.alpha * (newQVal-oldQVal)
#         for f, v in feats.items():
#             self.weights[f] += update * v
#         print(self.weights)

#     def final(self, state):
#         """Called at the end of each game."""
#         # call the super-class final method
#         PacmanQAgent.final(self, state)

#         # did we finish training?
#         if self.episodesSoFar == self.numTraining:
#             # you might want to print your weights here for debugging
#             "*** YOUR CODE HERE ***"
#             pass
