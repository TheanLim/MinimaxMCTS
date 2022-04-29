from prototype import Search, State, Action
from typing import Callable, Optional, Any, List, Union
import time
import random
import math
import pickle
from multiprocessing import Manager, Process


'''
A Prototype of a Node
'''
class Node:
  def __init__(self, state:State, parent=None):
    self.state = state
    self.parent = parent
    self.children = {} # {action:Node(stateAfterAction, self)}
    self.numVisits = 0
    self.utilities = None
  
  def isLeaf(self)->bool:
    # A terminal state is considered a leaf node
    return len(self.children)==0 or self.state.isTerminal()
  
  def __str__(self, level=0):
    ret = "\t"*level+repr(self)+"\n"
    for action, node in self.children.items():
        ret += node.__str__(level+1)
    return ret

  def __repr__(self):
    #return '<tree node representation>'
    s=[]
    s.append("rewards: "+str(self.utilities))
    s.append("numVisits: "+str(self.numVisits))
    s.append("children/actions: " + str(list(self.children.keys())))
    return str(self.__class__.__name__)+": {"+", ".join(s)+"}"
  '''
  def __str__(self)->str:
    s=[]
    s.append("rewards: "+str(self.utilities))
    s.append("numVisits: "+str(self.numVisits))
    s.append("children/actions: " + str(list(self.children.keys())))
    return str(self.__class__.__name__)+": {"+", ".join(s)+"}"
  '''

'''
A Monte Carlo Tree Search Object.
It samples the search space and expands the search tree according to promising nodes.
Less promising nodes are visited from time to time.
'''
class MCTS(Search):
  def __init__(self, 
               selectionPolicy:Callable[[Node],Node], 
               expansionPolicy:Callable[[State], List[Action]], 
               rollOutPolicy:Callable[[State],Any],  
               utilitySumFunc:Callable[[Any, Any], Any]=sum, 
               utilityIdx:Optional[List[int]]=None
               ):
    '''
    selectionPolicy: Given the current node, which child node should be selected to traverse to?
    expansionPolicy: Given the current (leaf) node, which child node should be expanded (grown) first?
    rollOutPolicy: Given the current node/state, how should a playout be completed? What's the sequence of action to take?
    utilitySumFunc: function used to sum two rewards. The default is sum()
    utilityIdx: Applicable if the utilities are encoded with multiple elements, each representing different agents' utility
                  For example utility =(0,1,1). utilityIdx:=2 means that only utility[utilityIdx] is considered.
    '''
    self.selectionPolicy = selectionPolicy
    self.expansionPolicy = expansionPolicy # function that returns a seq of actions
    self.rollOutPolicy = rollOutPolicy
    self.utilitySumFunc = utilitySumFunc
    self.utilityIdx = utilityIdx
  
  def search(self, 
             state:State, 
             maxIteration:Callable=(lambda: 1000000),
             maxTimeSec:Callable=(lambda: 1000),
             simPerIter:Callable=(lambda:1),
             breakTies:Callable[[List[Action]],Action]=random.choice
             )->Action:
    '''
    Search for the best action to take given a state.
    The search is stopped when the maxIteration or maxTimeSec is hitted. 
    Args:
      simPerIter: number of simulation(rollouts) from the chosen node.
      breakTies: Function used to choose an node from multiple equally good node.
    '''
    self.root = Node(state, None)
    self.simPerIter = simPerIter()
    maxTime = maxTimeSec()
    self.timeMax = time.time()+maxTime
    self.maxIter = maxIteration()
    self.breakTies = breakTies

    # Spawn a process to IDS for an action
    # Kill the process when time is up and return the latest action found
    with Manager() as manager:
      # Using a queue to share objects
      q = manager.Queue()
      p = Process(target=self._search, args=[q])
      p.start()
      # Usage: join([timeout in seconds])
      p.join(maxTime)
      if p.is_alive():
          p.terminate()
          p.join()
      # Get the latest chosen action
      action = None
      while not q.empty(): action = q.get()
    
    # If the search doesn't give any action, choose the first available action as the default
    if not action:
      action = self.expansionPolicy(state)
      print("Fail to search for an action - return the first possible action found.")
    #print("Player take", state.getCurrentPlayerSign(), " action ", action)
    return action

  def _search(self, queueOfActions):
    # Loop while have remaining iterations or time
    iterCnt = 0
    while iterCnt<self.maxIter and time.time()<self.timeMax:
      self.oneIteration()
      iterCnt+=1

      ########## Select the Best Action in this iteration #######
      ########## Select the best action based on its expected utilities ##########
      if not self.root.children: continue
      bestExpectedUtilities, bestActions = float("-inf"), []
      epsilon = 0.00001 # Prevent numeric overflow
      # The sequence of action follows the expansion policy used
      for action, child in self.root.children.items():
        if not child.utilities:
          childUtilities=0
        else:
          childUtilities = sum([child.utilities[idx] for idx in self.utilityIdx]) if self.utilityIdx else sum(child.utilities)

        expectedUtilities = childUtilities/(child.numVisits+epsilon)
        if expectedUtilities>bestExpectedUtilities:
          bestActions = [action]
          bestExpectedUtilities = expectedUtilities
        elif expectedUtilities==bestExpectedUtilities:
          bestActions.append(action)

      action = self.breakTies(bestActions)
      queueOfActions.put(action)
      ####### End Selecting the Best Action in this iteration #######
    # Nothing to return, just end the execution
    return 
  
  def oneIteration (self)->None:
    '''
    Perform one iteration of leaf node selection, expansion (if applicable), simulation, and backpropagation.
    Only expand a node if it was visited before. Otherwise, perform simulation on the node that wasn't visited.
    Simulation is performed `self.simPerIter` times
    '''
    node = self.selection()
    # If the node was visited, and expandable (not terminal)
    if node.numVisits>0 and not node.state.isTerminal():
      node = self.expansion(node)
    for i in range(self.simPerIter):
      utility = self.simulation(node)
      self.backpropagation(node, utility, self.utilitySumFunc)
  
  def selection(self)->Node:
    '''
    Select and returns a leaf node.
    Traverse from the root node to the leaf node, following self.selectionPolicy
    '''
    # Select a leaf node starting from the root node
    node = self.root
    depth = 0
    while not node.isLeaf():
      node = self.selectionPolicy(node, depth)
      depth+=1
    return node
  
  def expansion(self, node:Node)->Node:
    '''
    Fully expands a node and return one of its child node.
    Expands a node following self.expansionPolicy. 
    Returns the first children node.
    '''
    # Fully expand the tree ahead of time
    actions = self.expansionPolicy(node.state)
    for action in actions:
      # Add a new state to the tree
      stateAfterAction = node.state.takeAction(action)
      newNode = Node(stateAfterAction, node)
      node.children[action] = newNode
    # Choose the firstAction newNode to return
    return node.children[actions[0]]
  
  def simulation(self, node:Node)->Any:
    '''
    Returns the rewards received from this simulation
    '''
    return self.rollOutPolicy(node.state)
  
  def backpropagation(self, node:Node, utility:Any, utilitySumFunc:Callable=sum)->None:
    '''
    BackPropagate results to parent nodes.
    Update a node's Utility and Number of being visited.

    utilitySumFunc: function used to sum two utilities. The default is sum()
    '''
    while node:
      node.numVisits+=1
      if node.utilities:
        node.utilities = utilitySumFunc(node.utilities,utility)
      else:
        node.utilities = utility
      node = node.parent

def linearExpansion(state:State)->List[Action]:
  '''
  Returns a list of actions in a sequence 
  that are encoded by the state.
  '''
  return state.getActions()

def randomRollout(state:State)->Any:
  '''
  Starting from the provided state, randomly take actions
  until the terminal state. 
  Returns the terminal state utility.
  '''
  # Deep copy to perform takeAction that doesnt preserve state
  # This is done to speed up rollout
  state = pickle.loads(pickle.dumps(state))
  while not state.isTerminal():
    actions = state.getActions()
    '''
    n = len(actions)
    t = int(str(time.time())[-1])
    chosenIdx = t%n
    action = actions[chosenIdx]
    '''
    action = random.choice(actions)
    state = state.takeAction(action, preserveState=False)
  return state.getUtility()

class UCB:
  '''
  Given a parent node, returns a child node according to UCB1 quantity.
  utilityIdx: Applicable it the utilities are encoded with multiple elements, each representing different agents' utility
            For example utility =(0,1,1). utilityIdx:=2 means that only utility[utilityIdx] is considered.
  breakTies: Function used to choose an node from multiple equally good node.
  '''
  def __init__( self, 
                utilityIdx:Optional[List[int]]=None,
                explorationConstant:Union[float, int] = math.sqrt(2), 
                breakTies:Callable[[List[Action]],Action]=random.choice
                )->Node:
    self.utilityIdx = utilityIdx
    self.explorationConstant = explorationConstant
    self.breakTies =breakTies
  
  def __call__(self, node:Node, depth:int)->Node:
    bestUCB, bestChildNodes = float("-inf"), []
    epsilon = 0.00001
    
    # The sequence of action follows the expansion policy used
    for _, child in node.children.items():
      if not child.utilities:
          childUtilities=0
      else:
        # Shift the utilityIdx correctly so that each player is maximizing it's gain
        numPlayers = len(child.utilities)
        # No shifts if depth 0, numPlayers, 2*numPlayers
        shift = depth%numPlayers
        if self.utilityIdx:
          shiftedUtilityIdx = [(idx + shift)%numPlayers for idx in self.utilityIdx]
          childUtilities = sum([child.utilities[idx] for idx in shiftedUtilityIdx])
        else:
          childUtilities = sum(child.utilities)
      
      childExpectedUtility = childUtilities / (child.numVisits+epsilon)
      ucb = childExpectedUtility + self.explorationConstant * math.sqrt(math.log(node.numVisits)/(child.numVisits+epsilon))
      if ucb>bestUCB:
        bestChildNodes = [child]
        bestUCB = ucb
      elif ucb==bestUCB:
        bestChildNodes.append(child)
    return self.breakTies(bestChildNodes)