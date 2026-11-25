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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        bonus = 0.0
        foodList = newFood.asList()
        if foodList:
            nearestFood = min(manhattanDistance(newPos, f) for f in foodList)
            bonus += 8.0 / (nearestFood + 1)
            bonus += 2.0 / (len(foodList) + 1)

        for ghostState, scaredTimer in zip(newGhostStates, newScaredTimes):
            gDist = manhattanDistance(newPos, ghostState.getPosition())
            if scaredTimer > 0:
                if gDist <= 1:
                    bonus += 200.0
                else:
                    bonus += 12.0 / (gDist + 1)
            else:
                if gDist <= 1:
                    bonus -= 500.0
                elif gDist <= 2:
                    bonus -= 50.0 / gDist
                else:
                    bonus -= 2.0 / gDist

        successorGameState.data.score += bonus
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        evalFn = self.evaluationFunction

        def minimax(state, agentIdx, depthLeft):
            if state.isWin() or state.isLose() or depthLeft == 0:
                return evalFn(state)
            actions = state.getLegalActions(agentIdx)
            if not actions:
                return evalFn(state)

            if agentIdx == 0:
                best = float('-inf')
                for a in actions:
                    v = minimax(state.generateSuccessor(0, a), 1, depthLeft)
                    if v > best:
                        best = v
                return best
            else:
                nxt = (agentIdx + 1) % state.getNumAgents()
                nd = depthLeft - 1 if nxt == 0 else depthLeft
                best = float('inf')
                for a in actions:
                    v = minimax(state.generateSuccessor(agentIdx, a), nxt, nd)
                    if v < best:
                        best = v
                return best

        rootActions = gameState.getLegalActions(0)
        bestValue = float('-inf')
        bestAction = rootActions[0]
        for a in rootActions:
            v = minimax(gameState.generateSuccessor(0, a), 1, self.depth)
            if v > bestValue:
                bestValue = v
                bestAction = a
        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        evalFn = self.evaluationFunction

        def ab(state, agentIdx, depthLeft, alpha, beta):
            if state.isWin() or state.isLose() or depthLeft == 0:
                return evalFn(state)
            actions = state.getLegalActions(agentIdx)
            if not actions:
                return evalFn(state)

            if agentIdx == 0:
                val = float('-inf')
                for a in actions:
                    v = ab(state.generateSuccessor(0, a), 1, depthLeft, alpha, beta)
                    if v > val:
                        val = v
                    if val > beta:
                        return val
                    if val > alpha:
                        alpha = val
                return val
            else:
                nxt = (agentIdx + 1) % state.getNumAgents()
                nd = depthLeft - 1 if nxt == 0 else depthLeft
                val = float('inf')
                for a in actions:
                    v = ab(state.generateSuccessor(agentIdx, a), nxt, nd, alpha, beta)
                    if v < val:
                        val = v
                    if val < alpha:
                        return val
                    if val < beta:
                        beta = val
                return val

        rootActions = gameState.getLegalActions(0)
        alpha, beta = float('-inf'), float('inf')
        bestValue = float('-inf')
        bestAction = rootActions[0]
        for a in rootActions:
            v = ab(gameState.generateSuccessor(0, a), 1, self.depth, alpha, beta)
            if v > bestValue:
                bestValue = v
                bestAction = a
            if bestValue > alpha:
                alpha = bestValue
        return bestAction


        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        evalFn = self.evaluationFunction

        def expectimax(state, agentIdx, depthLeft):
            if state.isWin() or state.isLose() or depthLeft == 0:
                return evalFn(state)
            actions = state.getLegalActions(agentIdx)

            if agentIdx == 0:
                if not actions:
                    return evalFn(state)
                best = float('-inf')
                for a in actions:
                    v = expectimax(state.generateSuccessor(0, a), 1, depthLeft)
                    if v > best:
                        best = v
                return best
            else:
                nxt = (agentIdx + 1) % state.getNumAgents()
                nd = depthLeft - 1 if nxt == 0 else depthLeft
                if not actions:
                    return expectimax(state, nxt, nd)
                total = 0.0
                for a in actions:
                    total += expectimax(state.generateSuccessor(agentIdx, a), nxt, nd)
                return total / len(actions)

        rootActions = gameState.getLegalActions(0)
        bestValue = float('-inf')
        bestAction = rootActions[0]
        for a in rootActions:
            v = expectimax(gameState.generateSuccessor(0, a), 1, self.depth)
            if v > bestValue:
                bestValue = v
                bestAction = a
        return bestAction


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    totalScore = currentGameState.getScore()

    foodList = foodGrid.asList()
    if foodList:
        nearestFood = min(manhattanDistance(pacPos, fp) for fp in foodList)
        totalScore += 15.0 / (nearestFood + 1)
        totalScore += 3.0 / (len(foodList) + 1)

    for ghost in ghostStates:
        gDist = manhattanDistance(pacPos, ghost.getPosition())
        if ghost.scaredTimer == 0:
            if gDist <= 1:
                totalScore -= 800.0
            elif gDist <= 3:
                totalScore -= 20.0 / gDist
            else:
                totalScore -= 1.5 / gDist
        else:
            if ghost.scaredTimer > gDist + 1:
                totalScore += 30.0 / (gDist + 1)
            else:
                totalScore += 3.0 / (gDist + 1)

    if capsules:
        nearestCap = min(manhattanDistance(pacPos, c) for c in capsules)
        threatened = any(
            g.scaredTimer == 0 and manhattanDistance(pacPos, g.getPosition()) < 7
            for g in ghostStates
        )
        if threatened:
            totalScore += 20.0 / (nearestCap + 1)

    return totalScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
