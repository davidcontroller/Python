#Dati tratti da Modelli di Programmazione Lineare, Luigi de Giovanni, Laura Brentegani, pag 16-17 esempio 9
#Soluzione poco elegante ma funziona

from pulp import *

year = [1,2,3,4,5]
plan = ["A", "B", "C", "D"]


prob = LpProblem("FINANCE", LpMaximize)

a_vars = LpVariable.dicts("AVAIL", [(i,j) for i in plan for j in year], 0)

a_vars["A", 5] = 0
a_vars["B", 4] = 0
a_vars["B", 5] = 0
a_vars["C", 1] = 0
a_vars["C", 5] = 0
a_vars["C", 3] = 0
a_vars["C", 4] = 0
a_vars["D", 1] = 0
a_vars["D", 2] = 0
a_vars["D", 3] = 0
a_vars["D", 4] = 0


prob += lpSum([0.4 * a_vars[( "A",j)]
               +0.7 * a_vars[( "B",j)]
               + a_vars[( "C",j)]
               + 0.3 * a_vars[( "D",j)]
                for j in year]) +10000


prob += lpSum(a_vars[(i,1)] for i in plan[:2]) <= 10000

prob += lpSum(a_vars[(i,2)] for i in plan[:3]) <= 10000 - lpSum(a_vars[(i,1)] for i in plan[:2])

prob += lpSum(a_vars[(i,3)] for i in plan[:2]) <= 10000 + 0.4 * a_vars["A", 1] - a_vars["B", 1] - lpSum(a_vars[(i,2)] for i in plan[:3])

prob += a_vars["A", 4] + lpSum(a_vars[(i,3)] for i in plan[:2]) + a_vars["B", 2] + a_vars["C", 2] <= 10000 +0.7 * a_vars["B", 1] + 0.4 * lpSum(a_vars[("A",j)] for j in year[:2])

prob += a_vars["D", 5] + a_vars["C", 2] + a_vars["B", 3] + a_vars["A", 4] <= 10000 + 0.4 * lpSum(a_vars[("A",j)] for j in year[:3]) + 0.7 * lpSum(a_vars[("B",j)] for j in year[:2])

prob.solve()

print("STATUS", LpStatus[prob.status])


for v in prob.variables():
    print v.name, "-", v.varValue

print "Il costo totale è", value(prob.objective)
                             

