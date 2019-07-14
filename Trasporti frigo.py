#Dati tratti da Modelli di Programmazione Lineare, Luigi de Giovanni, Laura Brentegani, pag 9 esempio4


from pulp import *

warehouse = [1,2,3,4]
facility = ["FAC 1", "FAC 2", "FAC 3"]

demand = {1: 10,
          2: 60,
          3: 30,
          4: 40,
          }


maxcap = {"FAC 1": 50,
         "FAC 2": 70,
         "FAC 3": 20}

trasp = {"FAC 1" : {1: 6, 2: 8, 3:3, 4:4},
         "FAC 2" : {1: 2, 2: 3, 3:1, 4:3},
         "FAC 3" : {1: 2, 2: 4, 3:6, 4:5}}

prob = LpProblem("trasp", LpMinimize)

serv_vars = LpVariable.dicts("Service", [(i,j) for i in warehouse for j in facility], 0)

prob += lpSum(trasp[j][i] * serv_vars[(i,j)] for j in facility for i in warehouse)


for j in facility:
    prob += lpSum(serv_vars[(i,j)] for i in warehouse) <= maxcap[j]

for i in warehouse:
    prob += lpSum(serv_vars[(i,j)] for j in facility) >= demand[i]


prob.solve()

print("STATUS", LpStatus[prob.status])

TOL = .000001
for j in facility:
        for i in warehouse:
            if serv_vars[(i,j)].varValue > TOL:
                print"Lo stabilimento", j, "deve spedire al magazzino", i
    

for v in prob.variables():
    print v.name, "-", v.varValue

print "Il costo annuo dei trasporti è", value(prob.objective)
                             

