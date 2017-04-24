#!/usr/bin/env python
from __future__ import division
import argparse
import random
import numpy as np

def generate_items():
    items = []
    classes = set()
    maxWeight = 0
    maxCost = 0
    for i in range(20):
        item_name = "Item " + str(i)
        class_name = random.randint(0, 199999)
        weight = random.randint(0, 100000)
        cost = random.randint(0, 100000)
        resale = check_profit_positive(cost, random.randint(0, 100000))
        item_string = "{}; {}; {}; {}; {}".format(item_name, class_name, weight, cost, resale)
        items.append(item_string)
        classes.add(class_name)

        if maxWeight < weight:
            maxWeight = weight
        if maxCost < cost:
            maxCost = cost
    return (items, list(classes), maxWeight, maxCost)

def check_profit_positive(cost, resale):
    while resale < cost:
        resale = random.randint(0, 100000)
    return resale

def write_output(filename, items_chosen):
  with open(filename, "w") as f:
    for i in items_chosen:
      f.write("{0}\n".format(i))

#3 1's, 20 2's, 12 3's, 36 4's = 71
def generate_items_only_one_class_chosen():
    items = []
    classes = set()

    maxWeight = 0
    maxCost = 0

    for i in range(3): #Three items of class 1
        item_name = "Item " + str(i)
        class_name = 1
        weight = random.randint(0, 100000)
        cost = random.randint(0, 100000)
        resale = check_profit_positive(cost, random.randint(0, 100000))
        item_string = "{}; {}; {}; {}; {}".format(item_name, class_name, weight, cost, resale)
        items.append(item_string)
        classes.add(class_name)

        if maxWeight < weight:
            maxWeight = weight
        if maxCost < cost:
            maxCost = cost

    for i in range(20): #Twenty items of class 2
        item_name = "Item " + str(i + 3)
        class_name = 2
        weight = random.randint(0, 100000)
        cost = random.randint(0, 100000)
        resale = check_profit_positive(cost, random.randint(0, 100000))
        item_string = "{}; {}; {}; {}; {}".format(item_name, class_name, weight, cost, resale)
        items.append(item_string)
        classes.add(class_name)

        if maxWeight < weight:
            maxWeight = weight
        if maxCost < cost:
            maxCost = cost

    for i in range(12): #Twelve items of class 3
        item_name = "Item " + str(i + 23)
        class_name = 3
        weight = random.randint(0, 100000)
        cost = random.randint(0, 100000)
        resale = check_profit_positive(cost, random.randint(0, 100000))
        item_string = "{}; {}; {}; {}; {}".format(item_name, class_name, weight, cost, resale)
        items.append(item_string)
        classes.add(class_name)

        if maxWeight < weight:
            maxWeight = weight
        if maxCost < cost:
            maxCost = cost

    for i in range(36): #36 items of class 4
        item_name = "Item " + str(i + 35)
        class_name = 4
        weight = random.randint(0, 100000)
        cost = random.randint(0, 100000)
        resale = check_profit_positive(cost, random.randint(0, 100000))
        item_string = "{}; {}; {}; {}; {}".format(item_name, class_name, weight, cost, resale)
        items.append(item_string)
        classes.add(class_name)

        if maxWeight < weight:
            maxWeight = weight
        if maxCost < cost:
            maxCost = cost

    for i in range(29): #100 - 71 = 29
        item_name = "Item " + str(i + 71)
        class_name = i + 71
        weight = random.randint(0, 100000)
        cost = random.randint(0, 100000)
        resale = check_profit_positive(cost, random.randint(0, 100000))
        item_string = "{}; {}; {}; {}; {}".format(item_name, class_name, weight, cost, resale)
        items.append(item_string)
        classes.add(class_name)

        if maxWeight < weight:
            maxWeight = weight
        if maxCost < cost:
            maxCost = cost

    return (items, list(classes), maxWeight, maxCost)

def createConstraints(classes):
  # Generate random number of elements for the constraint
  allConstraints = []
  for i in range(8):
    num = np.random.randint(2,10)
    a = [classes[np.random.randint(0, len(classes))] for i in range(num)]
    allConstraints.append(a)

  constraints_formatted = []
  for constraint in allConstraints:
    constraints_formatted.append(', '.join(map(str, constraint)))
  return constraints_formatted

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="PickItems solver.")
  parser.add_argument("output_file", type=str, help="____.out")
  args = parser.parse_args()

  items_chosen, classes_chosen, maxWeight, maxCost = generate_items()
  #items_chosen, classes_chosen, maxWeight, maxCost = generate_items_only_one_class_chosen()
  constraints = createConstraints(classes_chosen)

  pounds = [maxWeight]
  dollars = [maxCost]
  num_items = [20]
  num_constraints = [8]

  setup_values = pounds + dollars + num_items + num_constraints
  write_to_file = setup_values + items_chosen + constraints

  for i in range(5):
      random.shuffle(constraints)
      write_to_file += ["", "New constraint " + str(i)] + constraints

  write_output(args.output_file, write_to_file)
