#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np
import math as m

class Item:

  def __init__(self, name, cls, weight, cost, val):
    self.name = name
    self.weight = weight
    self.cost = cost
    self.val = val
    self.cls = cls
    self.profit = self.val = self.cost
    # self.profit = m.ceil((self.val - self.cost) / 100000000)
    # self.realProfit = self.val - self.cost

  def __str__(self):
      return ("{}, weight: {}, cost: {}, val: {}, prof: {}").format(self.cls, self.weight, self.cost, self.val, self.profit)


"""
===============================================================================
  Please complete the following function.
===============================================================================
"""

def sortingFunc(x):
    if x.cost == 0 and x.weight == 0:
        return x.profit

    if x.cost == 0:
        return x.profit/x.weight + x.profit

    if x.weight == 0:
        return x.profit/x.cost + x.profit

    return x.profit/x.cost + x.profit/x.weight

def greedyPick(W, C, N, items):
    items.sort(key=lambda x: sortingFunc(x), reverse=True)

    currCost = 0
    currWeight = 0
    result = []
    print("LEN INPUT: ", len(items))
    for x in items:
        if currCost + x.cost > C or currWeight + x.weight > W:
            continue
        else:
            currCost += x.cost
            currWeight += x.weight
            result.append(x.name)
    print("LEN OUTPUT: ", len(result))
    return ", ".join(result)

# def knapsack(W, C, N, items):
#     print("Started")
#     maxProfit = m.ceil(sum([item.profit for item in items]))
#
#     if maxProfit <= 0:
#         return []
#     print("MaxProfit: ", maxProfit)
#     print("N: ", N)
#
#     K = np.empty((N+1, maxProfit+1), dtype=list)
#
#     # list of items included, weight, cost, profit, realProfit
#
#     K[0][0] = [[], 0, 0, 0, 0]
#
#     for p in range(1, maxProfit+1):
#         K[0][p] = [[], m.inf, 0, 0, 0, 0]
#
#     for i in range(1, N+1):
#         K[i][0] = [[], m.inf, 0, 0, 0, 0]
#
#     for i in range(1, N+1):
#         for p in range(1, maxProfit+1):
#             x = items[i-1]
#             if x.profit > p:
#                 K[i][p] = K[i-1][p]
#             else:
#                 # include[0] = list of items
#                 # include[1] = weight
#                 # include[2] = cost
#                 include = K[i-1][p-x.profit]
#
#                 # could add to previous if case
#                 # if x.weight + include[1] > W or x.cost + include[2] > C:
#                 #     K[i][p] = K[i-1][p]
#                 # else:
#                 itemListCopy = include[0][:]
#                 itemListCopy.append(i)
#                 # first in max: Don't include item
#                 # second in max: Include item
#                 K[i][p] = min(K[i-1][p], [itemListCopy, x.weight + include[1], x.cost + include[2], include[3] + x.profit, include[4] + x.realProfit], key=lambda x: x[1])
#     # print("ARRIVED")
#     # print(K[N][maxProfit])
#     # return K[N][maxProfit]
#
#     for i in reversed(range(maxProfit+1)):
#         if K[i][maxProfit][1] <= W:
#             print("HI")
#             print(K[i][maxProfit])
#             return K[i][maxProfit]

#
# def knapsack(W, C, N, items):
#   """
#   W = total weight/pounds
#   C = total cost/dollars
#   N = total number of items
#   items = list of items
#
#   [name 0]; [class 1]; [weight 2]; [cost 3]; [resale 4]
#   """
#   K = np.empty((W + 1, C + 1, N + 1), dtype=list)
#
#   for w in range(W + 1):
#     for c in range(C + 1):
#         K[w][c][0] = [[], 0]
#
#   for c in range(C + 1):
#       for i in range(N + 1):
#           K[0][c][i] = [[], 0]
#
#   for w in range(W + 1):
#       for i in range(N + 1):
#           K[w][0][i] = [[], 0]
#
#   for i in range(1, N+1):
#       for w in range(1, W + 1):
#           for c in range(1, C + 1):
#               x = items[i-1]
#               if m.ceil(x.weight) > w or m.ceil(x.cost) > c:
#                   K[w][c][i] = K[w][c][i-1]
#               else:
#                   include = K[w-m.ceil(x.weight)][c-m.ceil(x.cost)][i-1]
#                   if (include is None):
#                       print("Nope")
#                   profit = x.val - x.cost
#                   itemListCopy = include[0][:]
#                   itemListCopy.append(i)
#                   # first in max: Don't include item
#                   # second in max: Include item
#                   K[w][c][i] = max(K[w][c][i-1], [itemListCopy, include[1] + profit], key=lambda x: x[1])
#   print("Result: ", K[W][C][N])
#   return K[W][C][N]


def solve(P, M, N, C, items, constraints):
  """
  Write your amazing algorithm here.

  Return: a list of strings, corresponding to item names.
  """
  lst = []
  result = greedyPick(P, M, N, items)
  # result = knapsack(int(P), int(M), N, items)
  return [result]
  # lst.append(result)
  # return lst

"""
===============================================================================
  No need to change any code below this line.
===============================================================================
"""

def sortMinCost(constraint, classItemMap):
  # Sort the constraint by the class that has minimum total cost for all items
  #constraint.sort(key = lambda x: sum(classItemMap[x], lambda item: item.cost), reverse=True)
  # print('constraint is ' + str(constraint))
  constraint.sort(key = lambda x: costSum(x, classItemMap))
  return (constraint)

def costSum(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return m.inf
    return sum([item.cost for item in classItemMap[x]])

def sortMinWeight(constraint, classItemMap):
  # Sort the constraint by the class that has minimum total weight for all items
  constraint.sort(key = lambda x: weightSum(x, classItemMap))
  return (constraint)

def weightSum(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return m.inf
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
  # print('maxCanChooseSet is ' + str(maxCanChooseSet))
  for cls in maxCanChooseSet:
    for elem in classItemMap[cls]:
      allItemSet.add(elem)

  # print('allItemSet is ' + str(allItemSet))
  # print(allItemSet)
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
    for cls in range(N):
        classItemMap[cls] = set()

    for i in range(N):
       name, cls, weight, cost, val = f.readline().split("; ")
       temp = Item(name, int(cls.strip()), float(weight.strip()), float(cost.strip()), float(val.strip()))

       if int(cls.strip()) not in classItemMap:
         classItemMap[int(cls.strip())] = {temp}
       else:
         classItemMap[int(cls.strip())].add(temp)

       items.append((name, int(cls), float(weight), float(cost), float(val)))
    # print('classItemMap is ' +str(classItemMap))
    masterList = []

    for i in range(C):
      constraint = list(eval(f.readline()))
      masterList.append(constraint)

    funcNames = [sortMinCost, sortNumItems, sortMinWeight, sortMaxProfit]
    # funcNames = [sortMinCost]
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
    # print('max number of classes to choose from ' + str(len(canChoose)))
    # print('max canChoose set ' + str(maxCanChooseSet))
    # print('max func ' + str(maxFunc))
    #print('number of classes no choose from ' + str(len(noChoose)))
    #print(constraints)

    # Now I need to go through and create a set of all the objects that are in the given classes in my canChoose set
    # print('classItemMap is ' +str(classItemMap))
    itemSet = createItemSet(maxCanChooseSet, classItemMap)
    # print("ItemSet")
    # print(type(itemSet))

  return P, M, N, C, itemSet, constraints

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
  items_chosen = solve(P, M, len(items), C, items, constraints)

  write_output(args.output_file, items_chosen)


def createConstraints(classes):
  # Generate random number of elements for the constraint
  allConstraints = []
  for i in range(8):
    num = np.random.randint(2,10)
    a = [classes[np.random.randint(0, len(classes))] for i in range(num)]
    allConstraints.append(a)
