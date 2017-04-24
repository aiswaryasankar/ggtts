#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np

class Item:

  def __init__(self, name, cls, weight, cost, val):
    self.name = name
    self.weight = weight
    self.cost = cost
    self.val = val
    self.cls = cls


"""
===============================================================================
  Please complete the following function.
===============================================================================
"""

def solve(P, M, N, C, items, constraints):
  """
  Write your amazing algorithm here.

  Return: a list of strings, corresponding to item names.
  """
  return ['1']


"""
===============================================================================
  No need to change any code below this line.
===============================================================================
"""

def sortMinCost(constraint, classItemMap):
  # Sort the constraint by the class that has minimum total cost for all items
  #constraint.sort(key = lambda x: sum(classItemMap[x], lambda item: item.cost), reverse=True)
  constraint.sort(key = lambda x: sum([item.cost for item in classItemMap[x]]))
  return (constraint)

def sortMinWeight(constraint, classItemMap):
  # Sort the constraint by the class that has minimum total weight for all items
  constraint.sort(key = lambda x: sum([item.cost for item in classItemMap[x]]))
  return (constraint)

def sortMaxProfit(constraint, classItemMap):
  # Sort the constraint by the class that has maximum total profit for all items
  constraint.sort(key = lambda x: sum([item.val - item.cost for item in classItemMap[x]]), reverse=True)
  return (constraint)

def sortNumItems(constraint, classItemMap):
  # This passses in each of the constraints in constraint as x and returns the length of the number of items in its list as the metric by which to sort the constraints
  constraint.sort(key = lambda x: len(classItemMap[x]), reverse=True)
  return (constraint)

def createItemSet(maxCanChooseSet, classItemMap):
  # This function needs to take in the constraints and add in the corresponding list from classItemMap
  allItemSet = set()
  allItemSetNames = set()

  for cls in maxCanChooseSet:
    for elem in classItemMap[cls]:
      allItemSet.add(elem)
      allItemSetNames.add(elem.name)

  print('allItemSet is ' + str(allItemSetNames))


def read_input(filename):
  """
  P: float
  M: float
  N: integer
  C: integer
  items: list of tuples
  constraints: list of sets
  """
  with open(filename) as f:
    P = float(f.readline())
    M = float(f.readline())
    N = int(f.readline())
    C = int(f.readline())
    items = []
    constraints = []

    canChoose = set()
    noChoose = set()
    classItemMap = {}

    for i in range(N):
       name, cls, weight, cost, val = f.readline().split("; ")
       temp = Item(name, int(cls.strip()), float(weight.strip()), float(cost.strip()), float(val.strip()))

       if int(cls.strip()) not in classItemMap:
         classItemMap[int(cls.strip())] = {temp}
       else:
         classItemMap[int(cls.strip())].add(temp)

       items.append((name, int(cls), float(weight), float(cost), float(val)))

    masterList = []

    for i in range(C):
      constraint = list(eval(f.readline()))
      masterList.append(constraint)

    funcNames = [sortMinCost, sortNumItems, sortMinWeight, sortMaxProfit]
    maxCanChoose = -1
    maxCanChooseSet = {}
    maxFunc = ''

    for func in funcNames:
      canChoose = set()
      noChoose = set()

      #print('function name is ' + str(func))
      for constraint in masterList:
        # Function that calls the different sorting algorithms and returns whichever one results in the most classes
        constraint = func(constraint, classItemMap)

        l = 0
        while (l < len(constraint) and (constraint[l] in noChoose or constraint[l] in canChoose)):
          l += 1
        if l < len(constraint):
          canChoose.add(constraint[l])
          noChoose.update(constraint[l+1:])
          canChoose = canChoose.difference(noChoose)

      # Go through and check if number you can choose from is greatest using this func
      if len(canChoose) > maxCanChoose:
        #print('len canChoose ' + str(len(canChoose)))
        #print('func ' + str(func))
        maxCanChoose = len(canChoose)
        maxCanChooseSet = canChoose
        maxFunc = func
      #constraint = set(eval(f.readline()))
      #constraints.append(constraint)
    print('max number of classes to choose from ' + str(len(canChoose)))
    #print('max canChoose set ' + str(maxCanChooseSet))
    print('max func ' + str(maxFunc))
    #print('number of classes no choose from ' + str(len(noChoose)))
    #print(constraints)

    # Now I need to go through and create a set of all the objects that are in the given classes in my canChoose set
    itemSet = createItemSet(maxCanChooseSet, classItemMap)

  return P, M, N, C, items, constraints

def write_output(filename, items_chosen):
  with open(filename, "w") as f:
    for i in items_chosen:
      f.write("{0}\n".format(i))

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="PickItems solver.")
  parser.add_argument("input_file", type=str, help="____.in")
  parser.add_argument("output_file", type=str, help="____.out")
  args = parser.parse_args()

  P, M, N, C, items, constraints = read_input(args.input_file)
  #print('constraints are ')
  #print(constraints)
  items_chosen = solve(P, M, N, C, items, constraints)

  write_output(args.output_file, items_chosen)


def createConstraints(classes):
  # Generate random number of elements for the constraint
  allConstraints = []
  for i in range(8):
    num = np.random.randint(2,10)
    a = [classes[np.random.randint(0, len(classes))] for i in range(num)]
    allConstraints.append(a)


## Pass in an array of items you can choose from
## Each item has a weight and a cost
## Iterate through the list of weights and costs
## We aren't iterating through the elements at all but instead all the values from 0 to max for both
## For each possible 
def knapsack(items):

    table = []

    for i in range(noItems):
        table[i][0] = 0
    for j in range(totalWeight):
        table[0][j] = 0
    
    for i in range(noItems):
        for j in range(totalWeight):
            if weights[j] + table[i][j-1] > totalWeight:
                table[i][j] = table[i][j-1]
            else:
                table[i][j] = max(table[i-1][j-weights[j]] + value[j], table[i-1][j])
    return table[noItems][totalWeight]
    









