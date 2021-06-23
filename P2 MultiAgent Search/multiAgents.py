# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import searchAgents

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPos = [ghost.getPosition() for ghost in newGhostStates]
        currFoodList = currentGameState.getFood().asList()
        succFoodList = newFood.asList()
        ans=0

#         if(len(currFoodList)>len(succFoodList)):
#             return 10
        
        scared = min(newScaredTimes)
        if (scared==0) and (newPos in ghostPos):
            return -10
#         if (scared>0) and (newPos in ghostPos):
#             return 10

        if(len(currFoodList)>len(succFoodList)):
            return 10
        
#         if (scared>0) and (newPos in ghostPos):
#             return 10

        mini=1000
        for food in succFoodList:
            mini=min(mini, manhattanDistance(food,newPos))
            if (mini==1):
                break
        ans+= 10.0/mini
        
        mini=1000
        for ghost in ghostPos:
            mini=min(mini, manhattanDistance(ghost,newPos))
            if (mini==1):
                break
#         if (scared>0):
#             ans+= 10.0/mini
#         else:
        ans-= 10.0/mini
        return ans
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        dep = self.depth;
        
        def minValue(gameState, idx, d):
            
            if (gameState.isWin() or gameState.isLose() or d==dep):
                return self.evaluationFunction(gameState)
            
            v = pow(10,9)
            for action in gameState.getLegalActions(idx):
                succ= gameState.generateSuccessor(idx, action)
                if (idx==gameState.getNumAgents()-1):
                    v = min(v, maxValue(succ, d+1))
                else:
                    v = min(v, minValue(succ, idx+1, d))
            return v
        
        def maxValue(gameState, d):
            
            if (gameState.isWin() or gameState.isLose() or d==dep):
                return self.evaluationFunction(gameState)
            
            v = -pow(10,9)
            for action in gameState.getLegalActions(0):
                succ = gameState.generateSuccessor(0, action)
                v=max(v, minValue(succ, 1, d))
            return v
        
        ans=gameState.getLegalActions(0)[0]
        maxi=-pow(10,9)
        for action in gameState.getLegalActions(0):
            temp = minValue(gameState.generateSuccessor(0, action), 1, 0)
            if (temp>maxi):
                maxi=temp
                ans=action

        return ans
            
            
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        dep = self.depth;
        
        def minValue(gameState, idx, d, alpha, beta):
            
            if (gameState.isWin() or gameState.isLose() or d==dep):
                return self.evaluationFunction(gameState)
            
            v = pow(10,9)
            for action in gameState.getLegalActions(idx):
                succ= gameState.generateSuccessor(idx, action)

                if (idx==gameState.getNumAgents()-1):
                    v = min(v, maxValue(succ, d+1, alpha, beta))
                else:
                    v = min(v, minValue(succ, idx+1, d, alpha, beta))
                
#                 if (idx==1):
                if (v<alpha):
                    return v
#                 else:
#                     if (v>beta):
#                         return v
                beta = min(beta, v)    
            return v
        
        def maxValue(gameState, d, alpha, beta):
            
            if (gameState.isWin() or gameState.isLose() or d==dep):
                return self.evaluationFunction(gameState)
            
            v = -pow(10,9)
            for action in gameState.getLegalActions(0):
                succ = gameState.generateSuccessor(0, action)
                v=max(v, minValue(succ, 1, d, alpha, beta))
                if (v>beta):
                    return v
                alpha = max(alpha, v)
            return v
        
        alpha = -pow(10,9)
        beta = pow(10,9)
        ans=gameState.getLegalActions(0)[0]
        maxi=-pow(10,9)
        for action in gameState.getLegalActions(0):
            temp = minValue(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if (temp>maxi):
                maxi=temp
                ans=action
            alpha = max(alpha, temp) 

        return ans
            
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        dep = self.depth;
        
        def expValue(gameState, idx, d):
            
            if (gameState.isWin() or gameState.isLose() or d==dep):
                return self.evaluationFunction(gameState)
            
            v = 0
            for action in gameState.getLegalActions(idx):
                succ= gameState.generateSuccessor(idx, action)
                p = 1.0/(len(gameState.getLegalActions(idx)))
                if (idx==gameState.getNumAgents()-1):
                    v += (p*maxValue(succ, d+1))
                else:
                    v +=(p*expValue(succ, idx+1, d))
#                 p = 1.0/()
            return v
        
        def maxValue(gameState, d):
            
            if (gameState.isWin() or gameState.isLose() or d==dep):
                return self.evaluationFunction(gameState)
            
            v = -pow(10,9)
            for action in gameState.getLegalActions(0):
                succ = gameState.generateSuccessor(0, action)
                v=max(v, expValue(succ, 1, d))
            return v
        
        ans=gameState.getLegalActions(0)[0]
        maxi=-pow(10,9)
        for action in gameState.getLegalActions(0):
            temp = expValue(gameState.generateSuccessor(0, action), 1, 0)
            if (temp>maxi):
                maxi=temp
                ans=action

        return ans
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    
    # Did not check this function!!!!
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    ghostPos = [ghost.getPosition() for ghost in GhostStates]
    FoodList = Food.asList()
    ans=0
    if(currentGameState.isLose()):
        return -pow(10,5);
    scared = min(ScaredTimes)

    if (scared==0) and (Pos in ghostPos):
        return -100000
    if (scared>0) and (Pos in ghostPos):
        ans+=500
    mini=1000
    maxi=0
    for food in FoodList:
        temp = manhattanDistance(food,Pos)
        mini=min(mini, temp)
        maxi = max(maxi, temp)
        if (mini==0):
            break
    
    ans+= min(1000, 100.0/mini)
    ans-= 500.0/maxi
    
    mini=1000
    for ghost in ghostPos:
        mini=min(mini, manhattanDistance(ghost,Pos))
        if (mini<=3):
            break
    if(scared>0):        
        ans-= 1000.0
    else:
        ans+= min(1000, 100.0/mini)

    return ans

    util.raiseNotDefined()

    
# Abbreviation
better = betterEvaluationFunction
