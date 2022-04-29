from typing import List, Any

class Action:
  pass

'''
A Prototype of a State
'''
class State:
  def getActions(self)->List[Action]:
    pass
  def takeAction(self, action:Action, preserveState:bool=True)->'State':
    '''
    preserveState: if True, make a copy of the current state, act and return the copied state.
    '''
    pass
  def isTerminal(self)->bool:
    pass
  def getUtility(self)->Any:
    pass

'''
A Prototype of a Search object
'''
class Search:
  def search(self, state:State, *args, **kwargs)->Action:
    pass