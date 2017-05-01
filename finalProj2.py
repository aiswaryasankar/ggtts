#!/usr/bin/env python

from __future__ import division
import argparse
import numpy as np
import math as m
import random

#For given total items: do different greedy pick
#Keep track of best two constraint/class sets

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
problem_file_name = ""

sorts = {0: 'sortMinWeight', 1: 'sortMaxProfit', 2: 'sortMinCost', 3: 'sortProfitWCRatio', 4: 'sortProfitRatio1', 5: 'sortProfitRatio2', 6: 'sortNumItems', 7: 'linearWeightSort'}

def sortingFunc(x):
    if x.cost == 0 and x.weight == 0:
        return x.profit + x.profit

    if x.cost == 0:
        return x.profit/x.weight + x.profit

    if x.weight == 0:
        return x.profit/x.cost + x.profit

    return x.profit/x.cost + x.profit/x.weight

def sortingFunc2(x):
    if x.cost == 0 and x.weight == 0:
        return x.profit + x.profit

    if x.cost == 0:
        return x.profit/x.weight + x.profit

    if x.weight == 0:
        return x.profit/x.cost + x.profit

    return x.profit / (x.cost + x.weight)

def sortingFunc3(x):
    if x.cost == 0 and x.weight == 0:
        return x.profit + x.profit

    if x.cost == 0:
        return x.profit/x.weight + x.profit

    if x.weight == 0:
        return x.profit/x.cost + x.profit

    return x.profit/x.cost + x.profit/x.weight

def sortingFunc4(x):
    if x.cost == 0 and x.weight == 0:
        return x.profit + x.profit

    if x.cost == 0:
        return x.profit/x.weight + x.profit

    if x.weight == 0:
        return x.profit/x.cost + x.profit

    return x.profit/(x.cost * x.profit)

def sortingFunc5(x):
    if x.cost == 0 and x.weight == 0:
        return x.profit + x.profit

    if x.cost == 0:
        return x.profit/x.weight + x.profit

    if x.weight == 0:
        return x.profit/x.cost + x.profit

    return x.profit / np.log(x.cost * x.profit)

def greedyPick(W, C, items):
    sorts = [sortingFunc, sortingFunc2, sortingFunc3, sortingFunc4, sortingFunc5]
    maxProfit = 0.0
    masterResult = []
    maxFunc = ''

    for sort in sorts:
      items.sort(key=lambda x: sort(x), reverse=True)

      currCost = 0.0
      currWeight = 0.0
      profit = 0.0
      result = []
    #   print("LEN INPUT: ", len(items))
      for x in items:
        if currCost + x.cost > C or currWeight + x.weight > W:
            continue
        else:
            currCost += x.cost
            currWeight += x.weight
            profit += x.profit
            result.append(x.name)

      if profit > maxProfit:
        maxProfit = profit
        masterResult = result
        maxFunc = sort

    #   print("PROFIT WITH SORT " + sort.__name__ + ': ' + str(profit))
    #   print("LEN INPUT: ", len(result))
    #print('MAX FUNC ' + maxFunc.__name__)

    return (masterResult, maxProfit)

def greedyChoose(W, C, N, items):
  itemsList = list(items)
  itemsSorted = sorted(itemsList, key=lambda item: (item.val - item.cost), reverse=True)
  itemsChosen = []
  costSum = 0.0
  weightSum = 0.0
  profitSum = 0.0
  index = 0

  item = itemsList[0]
  while costSum + item.cost < C and weightSum + item.weight < W and index < N:
    item = itemsList[index]
    if costSum + item.cost > C:
    #   print('costSum over ')
      break
    if weightSum + item.weight > W:
    #   print('weightSum over')
      break

    itemsChosen.append(item.name)
    costSum += item.cost
    weightSum += item.weight
    profitSum += item.profit
    index += 1

  print('the profit sum is ' + str(profitSum))
  return (itemsChosen, profit)

def solve(P, M, items):
  """
  Write your amazing algorithm here.
  Return: a list of strings, corresponding to item names.
  """
  lst = []
  # result = greedyChoose(int(P), int(M), N, items)
  result, profit = greedyPick(P, M, items)
  lst.append(result)
  return (lst, profit)

"""
===============================================================================
  No need to change any code below this line.
===============================================================================
"""

def noSort (constraint, classItemMap):
  return constraint

## This function will compute the number of items, weight, cost, and the two ratios and apply a random number combination of the values. It will return this combination.  I will check if it is better by 
def linearWeightSort(constraint, classItemMap):
  constraint.sort(key = lambda x: linearWeights(x, classItemMap), reverse=True)
  return (constraint)

# This function takes in a class from the constraint and returns the linear combo generated from it
def linearWeights(x, classItemMap):
  ratio1 = profitRatio(x, classItemMap)
  ratio2 = sortRatio1(x, classItemMap)
  ratio3 = sortRatio2(x, classItemMap)
  cost = costSum(x, classItemMap)
  weight = weightSum(x, classItemMap)
  profit = 15 * maxProfit(x, classItemMap)
  return weight + cost + 6 * ratio1 + 4 * ratio2 + ratio3 + profit + 10 * len(classItemMap[x])

def sortProfitWCRatio(constraint, classItemMap):
    constraint.sort(key = lambda x: profitRatio(x, classItemMap), reverse=True)
    return (constraint)

def profitRatio(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return -1

    totalRatio = 0.0
    for item in classItemMap[x]:
        if item.cost == 0 and item.weight == 0:
            totalRatio += 2 * item.profit
        elif item.cost == 0:
            totalRatio += item.profit/item.weight + item.profit
        elif item.weight == 0:
            totalRatio +=  item.profit/item.cost + item.profit
        else:
            if (item.cost == 0 or item.weight == 0):
                print("Cost: ", item.cost)
                print("Weight: ", item.weight)
            totalRatio += item.profit/item.cost + item.profit/item.weight
    return totalRatio / len(classItemMap[x])

def sortMinCost(constraint, classItemMap):
  constraint.sort(key = lambda x: costSum(x, classItemMap))
  return (constraint)

def costSum(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return float('inf')
    return sum([item.cost for item in classItemMap[x]])/len(classItemMap[x])

def sortMinWeight(constraint, classItemMap):
  # Sort the constraint by the class that has minimum total weight for all items
  constraint.sort(key = lambda x: weightSum(x, classItemMap))
  return (constraint)

def weightSum(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return float('inf')
    return sum([item.weight for item in classItemMap[x]])/len(classItemMap[x])

def sortMaxProfit(constraint, classItemMap):
  # Sort the constraint by the class that has maximum total profit for all items
  constraint.sort(key = lambda x: maxProfit(x, classItemMap), reverse=True)
  return (constraint)

def maxProfit(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return float('-inf')
    return sum([item.profit for item in classItemMap[x]])/len(classItemMap[x])

def sortNumItems(constraint, classItemMap):
  # This passses in each of the constraints in constraint as x and returns the length of the number of items in its list as the metric by which to sort the constraints
  constraint.sort(key = lambda x: len(classItemMap[x]), reverse=True)
  return (constraint)

def sortRatio1(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return -1

    totalRatio = 0.0
    for item in classItemMap[x]:
        if item.cost == 0 and item.weight == 0:
            totalRatio += 2 * item.profit
        elif item.cost == 0:
            totalRatio += item.profit/item.weight + item.profit
        elif item.weight == 0:
            totalRatio +=  item.profit/item.cost + item.profit
        else:
            if (item.cost == 0 or item.weight == 0):
                print("Cost: ", item.cost)
                print("Weight: ", item.weight)
            totalRatio += item.profit / (item.cost + item.weight)
    return totalRatio / len(classItemMap[x])

def sortRatio2(x, classItemMap):
    if len(classItemMap[x]) == 0:
        return -1

    totalRatio = 0.0
    for item in classItemMap[x]:
        if item.cost == 0 and item.weight == 0:
            totalRatio += 2 * item.profit
        elif item.cost == 0:
            totalRatio += item.profit/item.weight + item.profit
        elif item.weight == 0:
            totalRatio +=  item.profit/item.cost + item.profit
        else:
            if (item.cost == 0 or item.weight == 0):
                print("Cost: ", item.cost)
                print("Weight: ", item.weight)
            totalRatio += item.profit / (item.cost * item.weight)
    return totalRatio / len(classItemMap[x])

def sortProfitRatio1(constraint, classItemMap):
    constraint.sort(key = lambda x: sortRatio1(x, classItemMap), reverse=True)
    return (constraint)

def sortProfitRatio2(constraint, classItemMap):
    constraint.sort(key = lambda x: sortRatio2(x, classItemMap), reverse=True)
    return (constraint)

def createItemList(maxCanChooseSet, classItemMap):
  # This function needs to take in the constraints and add in the corresponding list from classItemMap
  allItems = []

  for cls in maxCanChooseSet:
    for item in classItemMap[cls]:
      allItems.append(item)
  return allItems

def form_constraints(masterList, classItemMap, N):
    funcNames = [sortMinWeight, sortMaxProfit, sortMinCost, sortProfitWCRatio, sortProfitRatio1, sortProfitRatio2, sortNumItems, linearWeightSort]
    # funcNames = [sortNumItems]
    canChoose = set()
    noChoose = set()

    maxCanChoose = -1
    maxCanChooseSet = {}

    altCanChoose = -1
    altCanChooseSet = {}

    chooseSetList = []

    classConflicts = {}
    for cls in range(N):
        classConflicts[cls] = set()

    # map class x -> all classes that conflict with class x
    for constraint in masterList:
        for i in range(len(constraint)):
            cls = constraint[i]
            classConflicts[cls].update(constraint[:i] + constraint[i+1:])

    masterList.sort(key=lambda x: len(x), reverse=True)

    for func in funcNames:
      canChoose = set()
      noChoose = set()
      print("USING: ", func.__name__)

      for constraint in masterList:
        constraint = func(constraint, classItemMap)

        for cls in constraint:
            if cls in canChoose:
                constraint.remove(cls)
                noChoose.update(constraint)
                continue # Done processing this constraint, move on to next one

        # print("HEHE")
        for cls in constraint:
            if cls not in noChoose:
                canChoose.add(cls)
                noChoose.update(classConflicts[cls])
                break

      for cls in range(N):
        if cls not in noChoose:
            canChoose.add(cls)

      chooseSetList.append(canChoose)

    itemsList = []
    for s in chooseSetList:
        itemsList.append(createItemList(s, classItemMap))
    return itemsList

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

    # canChoose = set()
    # noChoose = set()

    # Sets up mapping from class -> items in the class
    classItemMap = {}
    for cls in range(N):
        classItemMap[cls] = set()

    for i in range(N):
       name, cls, weight, cost, val = f.readline().split(";")
       cls = int(cls.strip())
       weight = float(weight.strip())
       cost = float(cost.strip())
       val = float(val.strip())
       temp = Item(name, cls, weight, cost, val)

       if (val - cost > 0): #Don't add items with negative cost
           if cls not in classItemMap:
             classItemMap[cls] = {temp}
           else:
             classItemMap[cls].add(temp)

           items.append(temp)

    masterList = []

    # Grabs constraints and puts them in masterList
    for i in range(C):
      constraint = list(eval(f.readline()))
      masterList.append(constraint)

    itemList = form_constraints(masterList, classItemMap, N)
  return P, M, N, C, itemList, constraints

def write_scores_output(filename, profit, sort):
  sortsUsed = {}
  with open(filename, "a") as f:
    f.write('problem' + str(problem_file_name[-1]) + ": sort: " + str(sorts[sort]) + " profit: " + str(profit) + '\n')

def write_output(filename, items_chosen):
  with open(filename, "w") as f:
    for i in items_chosen[0]:
      f.write(i + '\n')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PickItems solver.")
  parser.add_argument("input_file", type=str, help="____.in")
  parser.add_argument("output_file", type=str, help="____.out")
  args = parser.parse_args()

  problem_file_name = args.input_file

  P, M, N, C, items, constraints = read_input(args.input_file)

  result_items = []
  result_profit = 0
  result_sort = -1
  index = 0
  
  sortsUsed = {}
  for item in items:
      print("Solving: ", index)
      items_chosen, profit = solve(P, M, item)
      print('Profit is ' + str(profit))

      if profit > result_profit:
          result_items = items_chosen
          result_profit = profit
          result_sort = index
      index += 1

  print("MAX PROFIT SORT " + str(result_sort) + ": " + str(result_profit))

  write_output(args.output_file, result_items)
  write_scores_output("score_tracker.txt", result_profit, result_sort)

def createConstraints(classes):
  # Generate random number of elements for the constraint
  allConstraints = []
  for i in range(8):
    num = np.random.randint(2,10)
    a = [classes[np.random.randint(0, len(classes))] for i in range(num)]
    allConstraints.append(a)