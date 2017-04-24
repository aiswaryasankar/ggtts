## Aiswarya Sankar
## 4/21/17
## Knapsack Implementation

import numpy as np 

def knapsack():
	noItems = int(input())
	totalWeight = int(input())
	weights = int(input().split(','))
	values = int(input().split(','))


	table = np.zeros(totalWeight * noItems).reshape((totalWeight, noItems))
	for item in noItems:
		for weight in totalWeight:
			pass


knapsack()



#code
import numpy as np

def knapsack():
    numTests = input()
    for num in numTests:
        noItems = int(input())
        totalWeight = int(input())
        weights = input().split(' ')
        values = input().split(' ')
    
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
    
knapsack()
