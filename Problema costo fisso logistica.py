#esempio estratto da Logistics Lecture Notes, Maria Grazia Scutellà, settembre 2015, università di Pisa, pag 27
#linearizzazione del vincolo condizionale con M pari a 2000000000, numero a caso minuscolo

from pulp import *

item = [1,2,3]

prob = LpProblem("MIX FAVOREVOLE", LpMaximize)

q_vars = LpVariable.dicts("QTA",item, 0)

a_vars = LpVariable.dicts("AUX",item, 0, cat='Integer')

cost = {1: 1000,
        2: 800,
        3: 900,
        }

profit = {1: 48,
          2: 55,
          3: 50,
          }

mach = {1: 1,
        2: 3,
        3: 6,
        }

grind = {1: 6,
         2: 3,
         3: 4,
        }

assem = {1: 5,
         2: 6,
         3: 2,
        }


prob += lpSum(profit[i] * q_vars[i] - cost[i] * a_vars[i] for i in item)


for i in item:
    prob += 0.0000000000002 * a_vars[i] < q_vars[i]

for i in item:
    prob += q_vars[i] <= 20000000000001 * a_vars[i]
    

prob += q_vars[1] - 50 * a_vars[1] <= 0

prob += q_vars[2] - 67 * a_vars[2] <= 0

prob += q_vars[3] - 75 * a_vars[3] <= 0

prob += lpSum(mach[i] * q_vars[i] for i in item) <= 600

prob += lpSum(grind[i] * q_vars[i] for i in item) <= 300

prob += lpSum(assem[i] * q_vars[i] for i in item) <= 400

prob.solve()

print("STATUS", LpStatus[prob.status])

for v in prob.variables():
    print v.name, "-", v.varValue












