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

        gamma = self.discount
        currStates = self.mdp.getStates()
        lenStates = len(currStates)
        k = self.iterations
        for i in range(k):
            VCurr = [0]*lenStates
            for j in range(lenStates):
                if self.mdp.isTerminal(currStates[j]):
                    continue
                maxi=-pow(10,5)
                actionList = self.mdp.getPossibleActions(currStates[j])
                for action in actionList:
                    TransitionList = self.mdp.getTransitionStatesAndProbs(currStates[j], action) # ((0, 1), 0.8)
                    sumi=0;
                    for newSnP in TransitionList:
                        newState = newSnP[0]
                        prob = newSnP[1]
                        reward = self.mdp.getReward(currStates[j], action, newState)
                        Vk = self.values[newState]
                        sumi=sumi+ (prob*((reward)+(Vk*gamma)))
                    maxi = max(maxi, sumi)
                VCurr[j] = maxi
            
            for j in range(lenStates):
                self.values[currStates[j]] = VCurr[j]
            
        "*** YOUR CODE HERE ***"


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
#       mdp.getStates()
#       mdp.getPossibleActions(state)
#       mdp.getTransitionStatesAndProbs(state, action)
#       mdp.getReward(state, action, nextState)
#       mdp.isTerminal(state)
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        gamma = self.discount
        currState = state
        currAction = action
        sumi=0
        TransitionList = self.mdp.getTransitionStatesAndProbs(currState, currAction)
        for newSnP in TransitionList:
            newState = newSnP[0]
            prob = newSnP[1]
            reward = self.mdp.getReward(currState, currAction, newState)
            Vk = self.values[newState]
            sumi=sumi+ (prob*((reward)+(Vk*gamma)))
        return sumi
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
#       mdp.getStates()
#       mdp.getPossibleActions(state)
#       mdp.getTransitionStatesAndProbs(state, action)
#       mdp.getReward(state, action, nextState)
#       mdp.isTerminal(state)
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        listActions = self.mdp.getPossibleActions(state)
        maxi = -pow(10,5)
        ansAct=listActions[0]
        for action in listActions:
            temp = self.computeQValueFromValues(state, action)
            if (temp>maxi):
                maxi=temp
                ansAct=action
        return ansAct
    
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
        
        gamma = self.discount
        currStates = self.mdp.getStates()
        lenStates = len(currStates)
        k = self.iterations
        for i in range(k):
            VCurr = [0]*lenStates
            State = currStates[i%lenStates]
            if self.mdp.isTerminal(State):
                continue
            maxi=-pow(10,5)
            actionList = self.mdp.getPossibleActions(State)
            for action in actionList:
                TransitionList = self.mdp.getTransitionStatesAndProbs(State, action) # ((0, 1), 0.8)
                sumi=0;
                for newSnP in TransitionList:
                    newState = newSnP[0]
                    prob = newSnP[1]
                    reward = self.mdp.getReward(State, action, newState)
                    Vk = self.values[newState]
                    sumi=sumi+ (prob*((reward)+(Vk*gamma)))
                maxi = max(maxi, sumi)
            self.values[State] = maxi

        "*** YOUR CODE HERE ***"

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
#         mdp.getStates()
#         mdp.getPossibleActions(state)
#         mdp.getTransitionStatesAndProbs(state, action)
#         mdp.getReward(state)
#         mdp.isTerminal(state)
        def maxQvalueFromValues(s):
            act=self.mdp.getPossibleActions(s)
            maxi=-pow(10,5)
            for a in act:
                maxi = max(maxi, self.computeQValueFromValues(s, a))
            return maxi
        
        
        
        pred = util.Counter()
        k = self.iterations
        stateList = self.mdp.getStates()
        PQ = util.PriorityQueue()
        
        for state in stateList:
            pred[state]= set()
            
        for state in stateList:
            if self.mdp.isTerminal(state):
                continue
            ActionList = self.mdp.getPossibleActions(state)
            maxi=-pow(10,5)
            for action in ActionList:
                maxi = max(maxi, self.getQValue(state, action))
                TransitionList = self.mdp.getTransitionStatesAndProbs(state, action)
                for newState, prob in TransitionList:
                    if (prob!=0):
                        pred[newState].add(state)
            
            diff = abs(self.values[state]-maxi)
            PQ.update(state, -diff)
        
        for it in range(k):
            if PQ.isEmpty():
                break
            curr = PQ.pop();
            if (self.mdp.isTerminal(curr)!=True):
                self.values[curr] = maxQvalueFromValues(curr)
            for p in pred[curr]:
                if self.mdp.isTerminal(p):
                    continue
                diff = abs(self.values[p] - maxQvalueFromValues(p))
                if (diff>self.theta):
                    PQ.update(p, -diff)
            
        
        
        
            
        "*** YOUR CODE HERE ***"

