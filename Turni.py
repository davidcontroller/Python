from pulp import *
x1 = pulp.LpVariable("x1", lowBound=0, cat="Integer")
x2 = pulp.LpVariable("x2", lowBound=0, cat="Integer")
x3 = pulp.LpVariable("x3", lowBound=0, cat="Integer")
x4 = pulp.LpVariable("x4", lowBound=0, cat="Integer")
x5 = pulp.LpVariable("x5", lowBound=0, cat="Integer")
x6 = pulp.LpVariable("x6", lowBound=0, cat="Integer")

problem = pulp.LpProblem("Min Personale", pulp.LpMinimize)

problem += x1 + x2 + x3 + x4 + x5 + x6 
problem += x1 + x6 >=10
problem += x1 + x2 >=20
problem += x2 + x3 >=45
problem += x3 + x4 >=40
problem += x4 + x5 >=50
problem += x5 + x6 >=12


problem.solve()
for variable in problem.variables():
    print variable.name, "=", variable.varValue

print value(problem.objective)


