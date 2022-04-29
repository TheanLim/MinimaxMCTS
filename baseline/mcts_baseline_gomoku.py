'''
Play againts a Random player - baseline
'''

from mnk import MNK
from mcts import MCTS, UCB, linearExpansion
from utils import sumTuple, gamePlay, transpose2DList, Random
import random

def utilityMNK(state:MNK, depth:int):
  '''
  This is a heuristic that calculates the utility of a state.
  It assumes a minimizer calling this function from odd depth,
  and a maximizer from the even depth.
  '''
  utility = 0
  weightPerSign = 1/state.k  # assuming each sign is equally important regardless of where it is placed
  board = state.getBoard()

  def calculateUtility(lst):
    LARGE_WIN_UTILITY = 1000000000
    utility = 0
    consecEmptyCells = 0            # Number of consecutive empty cells
    emptyCellsReqToWin = state.k    # Empty cells required to win. An empty cell implies an opportunity to place a sign to win
    candidateWinningPos = []        # This is wipe out when a possible winning path is interrupted by the other players. 
                                    #The length of it reflects the number of same signs placed uninterupted.
    prevSign = None
    for idx, sign in enumerate(lst):
      ## Empty Sign ##
      if sign==state.emptySign:
        consecEmptyCells+=1
        emptyCellsReqToWin-=1
      ## End Empty Sign ##
      else:
      ## Player Sign (nonEmpty Sign)##
        if sign!=prevSign: candidateWinningPos = []
        prevSign = sign
        emptyCellsReqToWin = emptyCellsReqToWin-1 if candidateWinningPos else (state.k-consecEmptyCells-1)
        consecEmptyCells = 0
        candidateWinningPos.append(idx+state.k-1)
      ## End Player Sign (nonEmpty Sign)##
      ## Update Utility if there's a way to win ##
      if emptyCellsReqToWin<=0:
        direction = 1 if prevSign==state.getCurrentPlayerSign() else -1
        utility+=direction*weightPerSign*len(candidateWinningPos)
        ## If WIN ##
        if len(candidateWinningPos)==state.k: return direction*LARGE_WIN_UTILITY
        ## End if WIN ##
        ## Additional Reward/Punishment for one-step win/lose ##
        if len(candidateWinningPos)+1 == state.k:
          # Else: I could still block but wasted a move
          bonus = 1*state.k if prevSign==state.getCurrentPlayerSign() else -2*state.k 
          utility+=bonus
      ## End Update Utility
      if candidateWinningPos and candidateWinningPos[0]==idx:
        # Pop it out because it no longer within the window of being helpful to win
        # in the next iteration
        candidateWinningPos.pop(0)
    return utility

  ## ROW utility ##
  for row in board:
    # Next iteration if no playerSign --> no need to count utility
    if not any(sign in row for sign in state.playerSigns): 
      continue
    ## At least one playerSign ##
    utility+=calculateUtility(row)

  ## COL utility ##
  boardTranspose = transpose2DList(board)
  for col in boardTranspose:  # col in the original board is now a row
    # Next iteration if no playerSign --> no need to count utility
    if not any(sign in col for sign in state.playerSigns): 
      continue
    ## At least one playerSign ##
    utility+=calculateUtility(col)

  ## Diag1 utility ##
  # Populate startPos. It helps to reduce computation on upper right ad lower left corners by excluding calculating utilties on them
  startRow, endRow, startCol, endCol= 0, state.m-state.k, 0, state.n-state.k  # starting a diag1 streak from endRow/endCol+1 is impossible to complete within bound
  startPos = [(i,0) for i in range(startRow, endRow+1)] + [(0,i) for i in range(startCol, endCol+1) if i!=0] #i!=0 because its already included when iterate row.
  for startI, startJ in startPos:
    # Construct a list signs to be evaluated
    diagList = []
    while startI<state.m and startJ<state.n:
      diagList.append(board[startI][startJ])
      startI, startJ = startI+1, startJ+1
    utility+=calculateUtility(diagList)

  ## Diag2 utility ##
  startRow, endRow, startCol, endCol= 0, state.m-state.k, state.k-1, state.n-1
  startPos = [(i,state.n-1) for i in range(startRow, endRow+1)] + [(0,i) for i in range(startCol, endCol+1) if i != state.n-1]
  for startI, startJ in startPos:
    # Construct a list signs to be evaluated
    diagList = []
    while startI<state.m and startJ>=0:
      diagList.append(board[startI][startJ])
      startI, startJ = startI+1, startJ-1
    utility+=calculateUtility(diagList)

  # Its a minimizer at odd depth
  return -utility if depth%2==1 else utility

def limitedRandomRollout(state):
  '''
  Randomly rollout two actions and then score the resulting state using heuristic
  '''
  # Assume the MCTS agent is "O"
  sign = 0 if state.getCurrentPlayerSign() == "O" else 1
  depth = 1
  while not state.isTerminal() and depth<=2:
    actions = state.getActions()
    action = random.choice(actions)
    state = state.takeAction(action)
    depth +=1
  score = utilityMNK(state, sign)
  return (-score, score)


def main():
  # Create a Tic Tac Toe Game State
  gomoku = MNK(15,15, 5, ["X", "O"])
  agent1 = Random()
  agent2 = MCTS(UCB(utilityIdx = [1]), linearExpansion, limitedRandomRollout, sumTuple, utilityIdx=[1]) # It is "O"
  agents = [agent1, agent2]
  agentKwargs=[{},{'maxTimeSec':lambda:1}]
  
  # StartGame
  random.seed(10) 
  winStatistics = gamePlay(rounds=100, initialState=gomoku, agentList = agents, 
                          agentKwargList=agentKwargs, utilitySumFunc=sumTuple, printDetails=False)
  print(winStatistics)
if __name__ == "__main__":
    main()
'''
Took  16 mins 23s, 
100 win , 0 draw
'''