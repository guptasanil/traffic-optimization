import gurobipy as gp
from gurobipy import *
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Iterable
RESULTPATH_MILP_FCFS = "./results/MILP_FCFS.csv"
RESULTPATH_WAITTIME = "./results/MILP_waittime.csv"
def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def GenerateTestCase(HVratio,avgArrivalInterval,laneCount,N,nonConflictingTrajectories):
    #avgArrivalInterval is per lane
    arrivalTime = [] #2dlist of vehicles' arrival time
    HV = []          #2dlist of vehicles' type (1->HV 0->CAV)
    fuelConsumptions = []
    for l in range(laneCount):
        arrivalTime.append([])
        HV.append([])
        fuelConsumptions.append([])
        t = 5 
        for n in range(N):
            t += (avgArrivalInterval) * -math.log(1- random.random()) 
            arrivalTime[l].append(t)
            HV[l].append(1 if random.random() < HVratio else 0)
            fuel_consumption = random.uniform(6.4, 7.4)
            fuelConsumptions[l].append(fuel_consumption)

    return arrivalTime,HV, fuelConsumptions,nonConflictingTrajectories

def MILP(arrivalTime,hv,nonConflictingTrajectories,G1=1,G2=3,TSTART=0,HVSchedulable=False,obj="throughput"):

    # Model
    m = gp.Model("schedule")
    m.setParam('OutputFlag', 0) #mute output 
    # These params have to be set well to not allow near solutions (e.g. 0.0000001 BIN to bypass big-M constraints) 
    m.setParam("IntegralityFocus",1) 
    m.setParam("IntFeasTol",1e-7) 
    M = 1e7   
    M1 = 1e6
    M2 = 1e5
    M3 = 1e4
    # given:
                                    # HV[l][i]:     is v[l][i] HV?
                                    # a[l][i] : arrival time of v[l][i]
    Nlane = len(arrivalTime)        #number of lanes
    t = []                          # t[l][i]:    entering time of v_l,i
    N = []                          # N[l]:       number of vehicles in lane l                 
    H = []                          # H[l][i][l'][i']: is v[l'][i'] at the head of lane when v[l][i] passes
    o = []                          # o[l][i][l'][i']: If v[l][i] pass before v[l'][i'] 
    r = []                          # r[l][i]: If there exist HV at the head of any lane when v[l][i] is passing 
    z = []                          # z[l][i]: is v[l][i] the last passing vehicle
    f = []

    a = np.asarray(arrivalTime,dtype = 'object')
    HV = np.asarray(hv,dtype = 'object')

    for arrival in arrivalTime:
        N.append(len(arrival))

    for L,arrivalL in enumerate(arrivalTime):
        t.append([])
        r.append([])
        z.append([])
        H.append([])
        o.append([])
        for i in range(N[L]):
            t[L].append(m.addVar(vtype=GRB.CONTINUOUS, name="t"+str(L)+ '_' + str(i) ))
            r[L].append(m.addVar(vtype=GRB.BINARY, name="r"+str(L)+ '_' + str(i) ))
            z[L].append(m.addVar(vtype=GRB.BINARY, name="z"+str(L)+ '_' + str(i) ))
            f[L].append(m.addVar(vtype=GRB.CONTINUOUS, name="t"+str(L)+ '_' + str(i) ))
            H[L].append([])
            o[L].append([])
            for L2 in range(Nlane):
                H[L][i].append([])
                o[L][i].append([])
                for j in range(N[L2]):
                    H[L][i][L2].append(m.addVar(vtype=GRB.BINARY, name="H"+str(L)+ '_' + str(i)  + '_'  +  str(L2) +'_'+str(j) ))
                    o[L][i][L2].append(m.addVar(vtype=GRB.BINARY, name="o"+str(L)+ '_' + str(i)  + '_'  +  str(L2) +'_'+str(j) ))
                H[L][i][L2].append(m.addVar(vtype=GRB.BINARY, name="H"+str(L)+ '_' + str(i)  + '_'  +  str(L2) +'_'+str(N[L2])))
    
    t = np.asarray(t, dtype='object')
    f = np.asarray(t, dtype='object')
    H = np.asarray(H, dtype='object')
    o = np.asarray(o, dtype='object')
    r = np.asarray(r, dtype='object')
    z = np.asarray(z, dtype='object')

    
    if obj == "waitTime":
        m.setObjective(gp.quicksum(t[l][i] for l in range(Nlane) for i in range(N[l])), GRB.MINIMIZE)
    else:
        mint = m.addVar(vtype=GRB.CONTINUOUS,name = "mint")
        lastlist = []
        for  l in range(Nlane):
            if len(t[l]) > 0:
                lastlist.append(t[l][-1])
        m.addConstr(mint == max_(lastlist))
        m.setObjective(mint, GRB.MINIMIZE)

    #Main Constraints: 
    #Altering constraints to equivalent ones or adding constraints that does not affect correctness can could greatly speed up  
    def AddMainConstraints():
        for l in range(Nlane):
            for i in range(N[l]):
                m.addConstr( o[l][i][l][i] == 0 ,"c0_1")
                for j in range(i + 1,N[l]):
                    m.addConstr( o[l][i][l][j] == 1 ,"c0_2")

        for l in range(Nlane):
            for i in range(N[l]):
                m.addConstr( t[l][i] >= a[l][i] ,"c1_1")
                m.addConstr( t[l][i] >= TSTART ,"c1_2") # for MILPBased
        
        
        #pass time
        for l in range(Nlane):
            for i in range(N[l]):
                m.addConstr( gp.quicksum( HV[l2][j] * H[l][i][l2][j]  for l2 in range(Nlane) for j in range(N[l2])) <= M3*r[l][i],"c2_2" )
                for l2 in range(Nlane):
                    for j in range(N[l2]):
                        if (l,l2) in nonConflictingTrajectories or (l2,l) in nonConflictingTrajectories:
                            m.addConstr(  t[l2][j]  - t[l][i] + M*(1-o[l][i][l2][j]) + M2*z[l][i] >= 0 ,"c2_3")
                        else:
                            m.addConstr(  t[l2][j]  - t[l][i] + M*(1-o[l][i][l2][j]) + M2*z[l][i] >= G1 ,"c2_4")
                            m.addConstr(  t[l2][j]  - t[l][i] + M*(1-o[l][i][l2][j]) + M2*z[l][i] >= G2 + M3*(r[l][i]-1),"c2_5")

        if HVSchedulable == False:
            for l in range(Nlane):
                for i in range(N[l]):
                    for l2 in [l2 for l2 in range(Nlane) if l2 != l and not((l,l2) in nonConflictingTrajectories or (l2,l) in nonConflictingTrajectories)]:
                        for j in range(N[l2]):
                            m.addConstr( M2*H[l][i][l2][j]*HV[l2][j] + (a[l][i] - a[l2][j]) <= M2 ,"c3")
        
        for l in range(Nlane):
            for i in range(N[l]):
                for l2 in range(Nlane):
                    for j in range(N[l2]):
                        if (l,i) == (l2,j):
                            continue
                        m.addConstr( o[l][i][l2][j] + o[l2][j][l][i] == 1 ,"c4_1")

                        for l3 in range(Nlane):
                            for k in range(N[l3]):
                                if (l,i) == (l3,k) or (l2,j) == (l3,k):
                                    continue 
                                m.addConstr( o[l][i][l2][j] + o[l2][j][l3][k] - 1 <= o[l][i][l3][k]  ,"c4_2")


        m.addConstr(gp.quicksum(z[l][i] for l in range(Nlane) for i in range(N[l])) == 1,"c_q")
        for l in range(Nlane):
            for i in range(N[l]):
                m.addConstr(gp.quicksum(H[l][i][l2][j] for l2 in range(Nlane) for j in range(N[l2])) - 1 + M3*(z[l][i]-1) <= 0,"c2_1_2")


        for l in range(Nlane):
                for i in range(N[l]):
                    m.addConstr( H[l][i][l][i] == 1 ,"c6_1")
                    for l2 in [l2 for l2 in range(Nlane) if l2 != l]:
                        m.addConstr( gp.quicksum( H[l][i][l2][j]  for j in range(N[l2] + 1) ) == 1 ,"c6_2")
                        m.addConstr( gp.quicksum( o[l2][j][l][i]  for j in range(N[l2]) )   == gp.quicksum( j*H[l][i][l2][j]  for j in range(N[l2] + 1) ) ,"c6_3")
        return 

    def printSolution():
        IntBinVars = [H,o,r,z]
        for Vars in IntBinVars:
            for var in flatten(Vars):
                if not var.X.is_integer(): # Gurobi gets an non integer solution happens about 5~10% depends on case size, tuning M values or other Gurobi params helps
                    # return basic FCFS results when fails
                    return FCFS(arrivalTime,hv,nonConflictingTrajectories,G1,G2)

        if m.status == GRB.OPTIMAL:
            tvars = []
            for lanet in t:
                tvars.append([round(tvar.X,2) for tvar in lanet])
            return tvars,m.objVal
        else:
            print('Model infeasible')
            exit()  

    m.update()
    AddMainConstraints()
    m.update()
    m.optimize()

    tvars,cost = printSolution() 
    runtime = m.Runtime
    return tvars,cost,runtime

def FCFS(arrivalTime,HV,nonConflictingTrajectories,G1=1,G2=3):
    Nlane = len(arrivalTime)
    t = [] # time of entering citical zone
    passTime = [] # time required to exit critical zone,G1 or G2
    N = []
    for i,lane in enumerate(arrivalTime):
        N.append(len(lane))
        t.append([])
        passTime.append([])
        for j in range(len(lane)):
            t[i].append(0)
            passTime[i].append(0)
    
    a = np.asarray(arrivalTime,dtype = 'object')
    HV = np.asarray(HV,dtype = 'object')
    t = np.asarray(t,dtype='object')
    passTime = np.asarray(passTime, dtype='object')
    N = np.asarray(N, dtype='object')

    head = np.zeros([Nlane],dtype = 'int')

    prev = None
    prevT = -100

    G2Until = -100
    passing = []
    while not np.array_equal(head,N):
        Candidate = []
        bestT = 1e5
        best = (-1,-1)
        for lane in range(Nlane):
            if head[lane] == N[lane]:
                continue
            Candidate.append((lane,head[lane]))  
        
        for c in Candidate:
            minT = a[c[0]][c[1]]
            for p in passing:
                if not ((p[0],c[0]) in nonConflictingTrajectories or  (c[0],p[0]) in nonConflictingTrajectories) and t[p[0]][p[1]] + passTime[p[0]][p[1]] > minT:
                    minT = t[p[0]][p[1]] + passTime[p[0]][p[1]]

            if best == (-1,-1) or a[c[0]][c[1]] < a[best[0]][best[1]]:
                best = c
                bestT = minT
            
        passTime[best[0]][best[1]] = G1 
        if G2Until >= bestT:
            passTime[best[0]][best[1]] = G2
        for l,i in enumerate(head):
            if i == N[l]:
                continue
            if HV[l][i] == 1:
                passTime[best[0]][best[1]] = G2

        if HV[best[0]][best[1]] == 1:
            G2Until = bestT + passTime[best[0]][best[1]]

        head[best[0]] += 1
        t[best[0]][best[1]] = bestT
        prev = best
        prevT = bestT
        passing.append(best)

    # print(passTime)
    return t,t.max(), passTime 


def checkWaitTime(arrivalTime,passTime,HV):
    CAVCount = 0
    HVCount = 0
    avgHVWaitTime = 0
    avgCAVWaitTime = 0
    for aT,pT,HV in zip(arrivalTime,passTime,HV):
        for a,p,h in zip(aT,pT,HV):
            if h:
                avgHVWaitTime += p - a 
                HVCount += 1 
            else:
                avgCAVWaitTime += p - a
                CAVCount += 1

    avgCAVWaitTime = -1 if CAVCount == 0 else avgCAVWaitTime / CAVCount
    avgHVWaitTime = -1 if HVCount == 0 else  avgHVWaitTime / HVCount
    return avgHVWaitTime,avgCAVWaitTime

if __name__ == "__main__":
    meanInterval = 2
    testCount = 10
    LANECOUNT = 4 
    VEACHLANE = 3
    MILPcosts = []
    MILG2costs = []

    FCFScosts = []
    avgRuntimes = []
    Ratios = []
    HVWaitTimes = []
    CAVWaitTimes = []
    HVWaitTimes2 = []
    CAVWaitTimes2 = []

    for HVratio in np.arange(0.0,1.1,0.1):
        print("ratio:", HVratio)
        FCFScost = 0
        MILPcost = 0
        MILG2cost = 0
        avgRuntime = 0
        avgHVWaitTime = 0
        avgCAVWaitTime = 0
        avgHVWaitTime2 = 0
        avgCAVWaitTime2 = 0
        successCount = testCount
        i = 0
        while i < testCount:
            arrivalTime,HV,nonConflictingTrajectories = GenerateTestCase(HVratio,meanInterval,LANECOUNT,VEACHLANE,[(0,1),(2,3)])
 
            t,cost,runtime = MILP(arrivalTime,HV,nonConflictingTrajectories,HVSchedulable = False)
            HVWaitTime,CAVWaitTime = checkWaitTime(arrivalTime,t,HV)
            avgHVWaitTime += HVWaitTime
            avgCAVWaitTime += CAVWaitTime
            avgRuntime += runtime
            
            
            t,cost2,runtime = MILP(arrivalTime,HV,nonConflictingTrajectories,HVSchedulable = True)
            MILPcost += cost 
            MILG2cost += cost2 

            t,cost = FCFS(arrivalTime,HV,nonConflictingTrajectories)
            HVWaitTime,CAVWaitTime = checkWaitTime(arrivalTime,t,HV)
            avgHVWaitTime2 += HVWaitTime
            avgCAVWaitTime2 += CAVWaitTime
            FCFScost += cost
            np.set_printoptions(precision=2)
            i += 1

        avgHVWaitTime /= testCount
        avgCAVWaitTime /= testCount
        avgHVWaitTime2 /= testCount
        avgCAVWaitTime2 /= testCount
        avgRuntime /= testCount
        MILPcost /= testCount
        MILG2cost /= testCount
        FCFScost /= testCount
        HVWaitTimes.append(round(avgHVWaitTime,3))
        CAVWaitTimes.append(round(avgCAVWaitTime,3))
        HVWaitTimes2.append(round(avgHVWaitTime2,3))
        CAVWaitTimes2.append(round(avgCAVWaitTime2,3))
        avgRuntimes.append(round(avgRuntime,3))
        MILPcosts.append(round(MILPcost,3))
        MILG2costs.append(round(MILG2cost,3))
        FCFScosts.append(round(FCFScost,3))
        Ratios.append(round(MILPcost/FCFScost,3)) 
    df = pd.DataFrame({
        'MILP': MILPcosts,
        'MILP(HV schedulable)': MILG2costs,
        'FCFS': FCFScosts,
        'runtime' : avgRuntimes,
        },index = np.arange(0,1.1,0.1))
    df.index.name = "HV ratio"
    df.to_csv(RESULTPATH_MILP_FCFS)

    df = pd.DataFrame({
        'MILP-HV': HVWaitTimes,
        'MILP-CAV': CAVWaitTimes,
        'FCFS-HV': HVWaitTimes2,
        'FCFS-CAV': CAVWaitTimes2,
        },index = np.arange(0,1.1,0.1))
    df.index.name = "HV ratio"
    df.to_csv(RESULTPATH_WAITTIME)