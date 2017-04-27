#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np
import math as m
import random

class Item:

  def __init__(self, name, cls, weight, cost, val):
    self.name = name
    self.weight = weight
    self.cost = cost
    self.val = val
    self.cls = cls
    self.profit = self.val - self.cost


"""
===============================================================================
  Please complete the following function.
===============================================================================
"""

def greedyChoose(W, C, N, items):
  itemsList = list(items)
  itemsSorted = sorted(itemsList, key=lambda item: (item.val - item.cost), reverse=True)
  itemsChosen = []
  costSum = 0.0
  weightSum = 0.0
  index = 0

  item = itemsList[0]
  while costSum + item.cost < C and weightSum + item.weight < W and index < N:
    item = itemsList[index]
    if costSum + item.cost > C:
      print('costSum over ')
      break
    if weightSum + item.weight > W:
      print('weightSum over')
      break
    
    itemsChosen.append(item.name)
    costSum += item.cost
    weightSum += item.weight
    index += 1

  print('is cost used > max cost? ' + str(costSum > C))
  print('is weight used > max weight? ' + str(weightSum > W))
  #print('the cost of all the items used is ' + str(costSum))
  #print('items to choose are ' + str(len(itemsChosen)))
  return itemsChosen

def knapsackV2(W, C, N, items):
  """
    Here we run a knapsack where we are finding the minimum weight that will fit inside the maximum possible value.  We check that the cost contstraint is upheld at each step. We want to optimize over the profit so we store that as a parameter first. The parameters are the number of items, the max cost, the item list.
  """
  # This will ultimately be reduced into a smaller number and rounded appropriately
  maxProfit = int(sum([item.profit for item in items]))
  
  if maxProfit < 0:
    #print('maxProfit is ' + str(maxProfit))
    return []

  table = np.empty((N + 1, maxProfit + 1), dtype=list)

  # Initialize the first row to infinity since we are minimizing the total weight
  for x in range(1, maxProfit):
    table[0][x] = [float('inf'), 0, []]

  for index in range(1, N + 1):
    for profit in range(1, maxProfit + 1):

      # Check if you haven't reached the next bound
      if items[index].profit > profit:
        table[index][profit] = table[index - 1][profit]

      # Here I need to also check that the cost constraint is satisfied, and I need to save the list of objects used so far
      # [weight, cost, itemList]
      else:
        includeItem = table[profit - items[index - 1].profit][index - 1]

        includeItemUpdate = [prevElem[0] + items[index - 1].weight, prevElem[1] + items[index - 1].cost, prevElem[2] + items[index - 1].name]

        notIncludeItem = table[profit][index - 1]

        # Check that you haven't gone over the cost by adding the item

        # The lambda function will compute the min over the cost while still storing the additional info
        table[profit][index] = min([notIncludeItem, includeItemUpdate], lambda x: x[0])
        #print('table value is ' + str(table[profit][index]))
  for index in range(1, N + 1):
    for profit in range(1, maxProfit + 1):
      print('hello')
      #print(table[index][profit])

def knapsack(W, C, N, items):
  """
  W = total weight/pounds
  C = total cost/dollars
  N = total number of items
  items = list of items

  [name 0]; [class 1]; [weight 2]; [cost 3]; [resale 4]
  """

  K = np.empty((W + 1, C + 1, N + 1), dtype=list)

  # Initialize to having the weight dimension be n*max(Val-Cost) instead, this serves as an upper bound for the total profit of the items which is what we are optimizing over

  maxProfit = m.ceil(max([item.val - item.cost for item in items])) * N
  #print('maxProfit is ' + str(maxProfit))

  for w in range(maxProfit + 1):
    for c in range(C + 1):
        K[w][c][0] = [[], 0]

  for c in range(C + 1):
      for i in range(N + 1):
          K[0][c][i] = [[], 0]

  for w in range(maxProfit + 1):
      for i in range(N + 1):
          K[w][0][i] = [[], 0]

  for i in range(1, N+1):
      for w in range(1, maxProfit + 1):
          for c in range(1, C + 1):
              x = items[i-1]
              if m.ceil(x.weight) > w or m.ceil(x.cost) > c:
                  K[w][c][i] = K[w][c][i-1]
              else:
                  include = K[w-m.ceil(x.weight)][c-m.ceil(x.cost)][i-1]
                  if (include is None):
                      print("Nope")
                  profit = x.val - x.cost
                  itemListCopy = include[0][:]
                  itemListCopy.append(i)
                  # first in max: Don't include item
                  # second in max: Include item
                  K[w][c][i] = max(K[w][c][i-1], [itemListCopy, include[1] + profit], key=lambda x: x[1])

  print("Result: ", K[W][C][N])
  return K[W][C][N]


def solve(P, M, N, C, items, constraints):
  """
  Write your amazing algorithm here.

  Return: a list of strings, corresponding to item names.
  """
  lst = []
  #result = knapsackV2(int(P), int(M), N, items)
  result = greedyChoose(int(P), int(M), N, items)
  lst.append(result)
  return lst

"""
===============================================================================
  No need to change any code below this line.
===============================================================================
"""

def noSort (constraint, classItemMap):
  return constraint

def sortMinCost(constraint, classItemMap):
  # Sort the constraint by the class that has minimum total cost for all items
  #constraint.sort(key = lambda x: sum(classItemMap[x], lambda item: item.cost), reverse=True)
  print('constraint before ' + str(constraint))
  constraint.sort(key = lambda x: costSum(x, classItemMap))
  print('constraint after ' + str(constraint))
  return (constraint)

def costSum(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return float('inf')
    return sum([item.cost for item in classItemMap[x]])

def sortMinWeight(constraint, classItemMap):
  # Sort the constraint by the class that has minimum total weight for all items
  constraint.sort(key = lambda x: weightSum(x, classItemMap))
  return (constraint)

def weightSum(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return float('inf')
    return sum([item.weight for item in classItemMap[x]])

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
  
  for cls in maxCanChooseSet:
    for elem in classItemMap[cls]:
      allItemSet.add(elem)

  print('the number of Items we can choose from is ' + str(len(allItemSet)))
  return list(allItemSet)


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
    for cls in range(N+1):
        classItemMap[cls] = set()

    for i in range(N):
       name, cls, weight, cost, val = f.readline().split(";")
       temp = Item(name, int(cls.strip()), float(weight.strip()), float(cost.strip()), float(val.strip()))

       if int(cls.strip()) not in classItemMap:
         classItemMap[int(cls.strip())] = {temp}
       else:
         classItemMap[int(cls.strip())].add(temp)

       items.append((name, int(cls), float(weight), float(cost), float(val)))
    #print('classItemMap is ' +str(classItemMap))
    #print(classItemMap)
    masterList = []

    for i in range(C):
      constraint = list(eval(f.readline()))
      masterList.append(constraint)

    funcNames = [sortNumItems, sortMinWeight, sortMaxProfit, sortMinCost]
    #funcNames = [noSort]
    maxCanChoose = -1
    maxCanChooseSet = {}
    maxFunc = ''

    ###### NEW STUFF TO TEST WITH ######
    #masterList = [random.sample(range(100), 10) for i in range(10)]
    #for l in masterList:
    #  print(l)

    for func in funcNames:
      canChoose = set()
      noChoose = set()

      #print('function name is ' + str(func))
      for constraint in masterList:
        # Function that calls the different sorting algorithms and returns whichever one results in the most classes
        constraint = func(constraint, classItemMap)
        print('calling function ' + str(func))
        l = 0
        #while (l < len(constraint) and (constraint[l] in noChoose or constraint[l] in canChoose)):
        while (l < len(constraint) and (constraint[l] in noChoose)):
          l += 1
        if l < len(constraint):
          canChoose.add(constraint[l])
          #noChoose.update(constraint[l+1:])
          constraint.remove(constraint[l])
          noChoose.update(constraint)
          #print('before diff ' + str(canChoose & noChoose))
          if len(canChoose & noChoose) > 0:
            # print('before diff ' + str(canChoose & noChoose))
            # print(str(len(canChoose)))
            # print(str(len(noChoose)))
            canChoose = canChoose.difference(noChoose)
            # print(str(len(canChoose)))
            # print(str(len(noChoose)))
          print('updating difference')
          canChoose = canChoose.difference(noChoose)
          if len(canChoose & noChoose) > 0:
            print('after diff ' + str(canChoose & noChoose))

      # Go through and check if number you can choose from is greatest using this func
      if len(canChoose) > maxCanChoose:
        #print('len canChoose ' + str(len(canChoose)))
        #print('func ' + str(func))
        maxCanChoose = len(canChoose)
        maxCanChooseSet = canChoose
        maxFunc = func

    print('there should not be any overlap btwn canChoose and noChoose' + str(canChoose & noChoose))
    print('canChoose ' + str(len(canChoose)))
    print('above length should be less than ' + str(C))
    print('noChoose ' + str(len(noChoose)))
    print('max number of classes to choose from ' + str(len(canChoose)))
    #print('max canChoose set ' + str(maxCanChooseSet))
    print('max func ' + str(maxFunc))
    #print('number of classes no choose from ' + str(len(noChoose)))
    #print(constraints)

    itemSet = createItemSet(maxCanChooseSet, classItemMap)


  return P, M, N, C, itemSet, constraints

def write_output(filename, items_chosen):
  with open(filename, "w") as f:
    for i in items_chosen[0]:
      #print(i + '\n')
      #f.write("{0}\n".format(i))
      f.write(i + '\n')

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="PickItems solver.")
  parser.add_argument("input_file", type=str, help="____.in")
  parser.add_argument("output_file", type=str, help="____.out")
  args = parser.parse_args()

  P, M, N, C, items, constraints = read_input(args.input_file)
  items_chosen = solve(P, M, len(items), C, items, constraints)

  write_output(args.output_file, items_chosen)


def createConstraints(classes):
  # Generate random number of elements for the constraint
  allConstraints = []
  for i in range(8):
    num = np.random.randint(2,10)
    a = [classes[np.random.randint(0, len(classes))] for i in range(num)]
    allConstraints.append(a)
