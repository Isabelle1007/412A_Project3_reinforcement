# valueIterationAgents.py
# -----------------------
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

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        for i in range(self.iterations):        
            newValues = util.Counter()
            for state in mdp.getStates():
                if mdp.isTerminal(state):
                    continue
                maxValue = float("-inf")
                for action in mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    maxValue = max(maxValue, q_value)
                newValues[state] = maxValue
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        q_value = 0
        for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
            reward = mdp.getReward(state, action, nextState)
            q_value += prob * (reward + self.discount * self.values[nextState])
        return q_value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        maxValue = float("-inf")
        bestAction = None
        for action in mdp.getPossibleActions(state):
            currValue = 0
            for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                reward = mdp.getReward(state, action, nextState)
                currValue += prob * (reward + self.discount * self.values[nextState])
            maxValue = max(maxValue, currValue)
            if(maxValue == currValue):
                bestAction = action
        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        newValues = util.Counter()
        for i in range(self.iterations):        
            state = mdp.getStates()[i % len(mdp.getStates())]
            if mdp.isTerminal(state):
                    continue
            maxValue = float("-inf")
            for action in mdp.getPossibleActions(state):
                q_value = self.computeQValueFromValues(state, action)
                maxValue = max(maxValue, q_value)
            if(len(mdp.getPossibleActions(state)) != 0):
                newValues[state] = maxValue
            self.values = newValues
                

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        
        # Compute predecessors of all states
        predecessors = {}
        for state in mdp.getStates():
            predecessors[state] = set()
        for state in mdp.getStates():
            for action in mdp.getPossibleActions(state):
                for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)
                        
        # Initialize an empty priority queue
        queue = util.PriorityQueue()
        
        # For non-terminal state s, find diff and assign priority
        for state in mdp.getStates():
            if not mdp.isTerminal(state):
                maxValue = float("-inf")
                for action in mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    maxValue = max(maxValue, q_value)
                diff = abs(self.values[state] - maxValue)
                queue.push(state, diff*(-1))
        
        # For each iteration, pop the state with the highest priority from the priority queue
        for i in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
            # Update the value of s (if it is not a terminal state) in self.values
            if not mdp.isTerminal(state):
                maxValue = float("-inf")
                for action in mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    maxValue = max(maxValue, q_value)
                self.values[state] = maxValue
            # For each predecessor p of s, find diff and assign priority
            for predecessor in predecessors[state]:
                if not mdp.isTerminal(predecessor):
                    maxValue = float("-inf")
                    for action in mdp.getPossibleActions(predecessor):
                        q_value = self.computeQValueFromValues(predecessor, action)
                        maxValue = max(maxValue, q_value)
                    diff = abs(self.values[predecessor] - maxValue)
                    # If diff > theta, push p into the priority queue with new priority
                    if diff > self.theta:
                        queue.update(predecessor, diff*(-1))
                        

