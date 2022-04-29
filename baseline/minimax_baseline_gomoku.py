'''
Play againts a Random player - baseline
'''

from typing import List
from mnk import MNK
from utils import sumTuple, Random, gamePlay, transpose2DList
from minimax import MinimaxIDS, cacheExpansion
import random

def utilityMNK(state:MNK, depth:int):
  utility = 0
  weightPerSign = 1/state.k  # assuming each sign is equally important regardless of where it is placed
  board = state.getBoard()

  def calculateUtility(lst:List):
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
          # if it's my turn ==> I am winning!
          #if prevSign==state.getCurrentPlayerSign(): return direction*LARGE_WIN_UTILITY
          # Else: I could still block but wasted a move
          bonus = 1*state.k if prevSign==state.getCurrentPlayerSign() else -2*state.k 
          #bonus = -2*state.k
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
      #print("NExtRow")
      continue
    ## At least one playerSign ##
    #print("Row", calculateUtility(row))
    utility+=calculateUtility(row)

  ## COL utility ##
  boardTranspose = transpose2DList(board)
  for col in boardTranspose:  # col in the original board is now a row
    # Next iteration if no playerSign --> no need to count utility
    if not any(sign in col for sign in state.playerSigns): 
      #print("Next col")
      continue
    ## At least one playerSign ##
    #print("Col", calculateUtility(col))
    utility+=calculateUtility(col)

  ## Diag1 utility ##
  # Populate startPos. It helps to reduce computation on upper right ad lower left corners by excluding calculating utilties on them
  startRow, endRow, startCol, endCol= 0, state.m-state.k, 0, state.n-state.k  # starting a diag1 streak from endRow/endCol+1 is impossible to complete within bound
  startPos = [(i,0) for i in range(startRow, endRow+1)] + [(0,i) for i in range(startCol, endCol+1) if i!=0] #i!=0 because its already included when iterate row.
  #print("Diag1",startPos)
  for startI, startJ in startPos:
    # Construct a list signs to be evaluated
    diagList = []
    while startI<state.m and startJ<state.n:
      diagList.append(board[startI][startJ])
      startI, startJ = startI+1, startJ+1
    #print(diagList)
    #print("D1", calculateUtility(diagList))
    utility+=calculateUtility(diagList)

  ## Diag2 utility ##
  startRow, endRow, startCol, endCol= 0, state.m-state.k, state.k-1, state.n-1
  startPos = [(i,state.n-1) for i in range(startRow, endRow+1)] + [(0,i) for i in range(startCol, endCol+1) if i != state.n-1]
  #print("Diag2",startPos)
  for startI, startJ in startPos:
    # Construct a list signs to be evaluated
    diagList = []
    while startI<state.m and startJ>=0:
      diagList.append(board[startI][startJ])
      startI, startJ = startI+1, startJ-1
    #print(diagList)
    #print("D2", calculateUtility(diagList))
    utility+=calculateUtility(diagList)

  # Its a minimizer at odd depth
  return -utility if depth%2==1 else utility

def main():
  # Create a Tic Tac Toe Game State
  TTT = MNK(15,15, 5, ["O", "X"])
  agent1 = Random()
  agent2 = MinimaxIDS(time=1, maxDepth = 15, evaluationFunction=utilityMNK, expansionPolicy=cacheExpansion, toCache=True)
  agents = [agent1, agent2]
  
  random.seed(10)  ## for simulation
  winStatistics = gamePlay( rounds=100, initialState=TTT, agentList = agents,
                            utilitySumFunc=sumTuple, printDetails=False)
  print(winStatistics)
if __name__ == "__main__":
    main()
'''
maxDepth = 15 is used to cutoff search when there's no significant change of bestAction returned
Took  10 mins 41s, 100 win , 0 draw 
'''