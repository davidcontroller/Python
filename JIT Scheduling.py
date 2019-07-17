#Problema di schedulazione MINMAX
#Dati tratti da Modelli di Programmazione Lineare, Luigi de Giovanni, Laura Brentegani, pag 20 esempio 11

from pulp import *

job = [1,2,3,4,5]

prob = LpProblem("scheduling", LpMinimize)

lag_vars = LpVariable.dicts("LAGS",job,cat='Continuous')

min_vars = LpVariable.dicts("MINUTES",job, 0, cat='Integer')

prob += 750 * lpSum(lag_vars[i] for i in job)

#VINCOLI LINEARIZZATI

prob += lag_vars[1] - min_vars[1] >= -27
prob += lag_vars[1] + min_vars[1] >= +27
prob += lag_vars[2] - min_vars[2] >= -31
prob += lag_vars[2] + min_vars[2] >= +31
prob += lag_vars[3] - min_vars[3] >= -38
prob += lag_vars[3] + min_vars[3] >= +38
prob += lag_vars[4] - min_vars[4] >= -45
prob += lag_vars[4] + min_vars[4] >= +45
prob += lag_vars[5] - min_vars[5] >= -47
prob += lag_vars[5] + min_vars[5] >= +47

#ALTRI VINCOLI

prob += min_vars[2] >= min_vars[1] + 5
prob += min_vars[3] >= min_vars[2] + 7
prob += min_vars[4] >= min_vars[3] + 4
prob += min_vars[5] >= min_vars[4] + 7


prob.solve()

print("STATUS", LpStatus[prob.status])

for v in prob.variables():
    print v.name, "-", v.varValue
