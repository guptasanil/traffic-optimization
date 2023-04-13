from MILP import *

import time
import csv
import pandas as pd
RESULTPATH = "./OptimizedDPFuel.csv"



def EnvironmentalCost(fuel_consumption, time, distance):
    distance = 0.1
    fuel_used = (fuel_consumption / 100) * (distance / 1000) * time
    return fuel_used


def DP(arrivalTime, HV, fuelConsumptions, nonConflictingTrajectories, G1=1, G2=3):
    zeros = np.zeros(len(arrivalTime), dtype='int')
    table = {}  # initializes a table to store the solved pass time of each vehicle
    last_vehicle = -1

    def MinPass(Pass):  # recursively computes the minimum pass time for a given set of vehicles
        if tuple(Pass) in table:
            return table[tuple(Pass)]

        if np.array_equal(Pass, zeros):
            table[tuple(Pass)] = 0
            return 0

        mint = math.inf
        last_vehicle = -1
        for i, a in enumerate(arrivalTime):
            if Pass[i] == 0:
                continue

            Pass_ = np.array(Pass, copy=True)
            Pass_[i] -= 1
            t = max(a[Pass[i] - 1], MinPass(Pass_) +
                    PassTime(Pass_), Legal(Pass_, i))
            if t < mint:
                mint = t
                last_vehicle = i
        table[tuple(Pass)] = mint
        return mint

    # get the pass time(time gap) based on the current state of lanes
    def PassTime(Pass):
        for i, n in enumerate(Pass):
            if n >= len(HV[i]):
                continue
            if HV[i][n]:
                return G2
        return G1

    # if the vehicle from lane i can pass next based on the state of lanes (No passing before a HV which arrived earlier)
    def Legal(Pass, i):
        for j, n in enumerate(Pass):
            if j == i:
                continue
            if n >= len(HV[j]):
                continue
            if HV[j][n] and arrivalTime[j][n] < arrivalTime[i][Pass[i]]:
                return math.inf
        return 0

    Pass = []
    for a in arrivalTime:
        Pass.append(len(a))

    Pass = np.asarray(Pass)
    startTime = time.time()
    cost = MinPass(Pass)
    runtime = time.time() - startTime

    fuel_consumption = 0
    for i, a in enumerate(arrivalTime):
        for j, t in enumerate(a):
            pass_time = t - arrivalTime[i][j - 1] if j > 0 else t
            # print(cost)
            fuel_used = EnvironmentalCost(fuelConsumptions[i][j], cost, distance=0.1)
            fuel_consumption += fuel_used

    # print(max(arrivalTime[last_vehicle][Pass[last_vehicle]-1], cost+PassTime(Pass)))

    # print(fuel_consumption)
    return cost, runtime, fuel_consumption

def OptimizedDP(arrivalTime, HV, fuelConsumptions, nonConflictingTrajectories, G1=1, G2=3):
    zeros = np.zeros(len(arrivalTime), dtype='int')
    table = {}  # initializes a table to store the solved pass time of each vehicle
    last_vehicle = -1

    def MinPass(Pass):
        if tuple(Pass) in table:
            return table[tuple(Pass)]

        if np.array_equal(Pass, zeros):
            table[tuple(Pass)] = (0, 0)
            return 0, 0

        min_time = math.inf
        min_fuel = math.inf
        last_vehicle = -1
        for i, a in enumerate(arrivalTime):
            if Pass[i] == 0:
                continue

            Pass_ = np.array(Pass, copy=True)
            Pass_[i] -= 1
            prev_pass_time, prev_fuel = MinPass(Pass_)
            t = max(a[Pass[i] - 1], prev_pass_time + PassTime(Pass_), Legal(Pass_, i))
            pass_time = t - arrivalTime[i][Pass[i] - 1]
            fuel_used = EnvironmentalCost(fuelConsumptions[i][Pass[i] - 1], pass_time, distance)
            total_fuel = prev_fuel + fuel_used

            if total_fuel < min_fuel:
                min_time = t
                min_fuel = total_fuel
                last_vehicle = i

        table[tuple(Pass)] = (min_time, min_fuel)
        return min_time, min_fuel


    # get the pass time(time gap) based on the current state of lanes
    def PassTime(Pass):
        for i, n in enumerate(Pass):
            if n >= len(HV[i]):
                continue
            if HV[i][n]:
                return G2
        return G1

    # if the vehicle from lane i can pass next based on the state of lanes (No passing before a HV which arrived earlier)
    def Legal(Pass, i):
        for j, n in enumerate(Pass):
            if j == i:
                continue
            if n >= len(HV[j]):
                continue
            if HV[j][n] and arrivalTime[j][n] < arrivalTime[i][Pass[i]]:
                return math.inf
        return 0

    Pass = []
    for a in arrivalTime:
        Pass.append(len(a))

    Pass = np.asarray(Pass)
    startTime = time.time()
    cost, total_fuel = MinPass(Pass)
    runtime = time.time() - startTime

    fuel_consumption = 0
    for i, a in enumerate(arrivalTime):
        for j, t in enumerate(a):
            pass_time = t - arrivalTime[i][j - 1] if j > 0 else t
            fuel_used = EnvironmentalCost(fuelConsumptions[i][j], cost, distance=0.1)
            fuel_consumption += fuel_used

    # print(fuel_consumption)
    return cost, runtime, total_fuel


if __name__ == '__main__':
    MEANINTERVAL = 2
    TESTCOUNT = 1
    LANECOUNT = 4
    VEACHLANE = 8
    DPcosts = []
    Opcosts = []
    FCFScosts = []
    avgRuntimes = []
    OpRuntimes = []
    Ratios = []
    totalFuelConsumptions = []
    OpFuelConsumptions = []
    FCFSTotalFuelConsumptions = []


    for HVratio in np.arange(0, 1.1, 0.1):
        FCFScost = 0
        DPcost = 0
        Opcost = 0
        avgRuntime = 0
        OpRuntime = 0
        totalFuelConsumption = 0
        totalFcfsFuel = 0
        OpFuelConsumption = 0

        distance = 0.1

        successCount = TESTCOUNT
        i = 0
        while i < TESTCOUNT:
            arrivalTime, HV, fuelConsumptions, nonConflictingTrajectories = GenerateTestCase(HVratio, MEANINTERVAL, LANECOUNT, VEACHLANE, [])

            cost, runtime, fuel_consumption = DP(arrivalTime, HV, fuelConsumptions, nonConflictingTrajectories, distance)
            DPcost += cost
            costOp, runtimeOp, fuel_consumptionOp = OptimizedDP(arrivalTime, HV, fuelConsumptions, nonConflictingTrajectories, distance)
            Opcost += costOp
            t, cost, passTime = FCFS(arrivalTime, HV, nonConflictingTrajectories)
            FCFScost += cost
            avgRuntime += runtime
            OpRuntime += runtimeOp
            totalFuelConsumption += fuel_consumption
            OpFuelConsumption += fuel_consumptionOp
            np.set_printoptions(precision=3)
            i += 1
        avgRuntime /= TESTCOUNT
        OpRuntime /= TESTCOUNT
        DPcost /= TESTCOUNT
        Opcost /= TESTCOUNT
        FCFScost /= TESTCOUNT
        totalFcfsFuel /= TESTCOUNT
        OpFuelConsumption /= TESTCOUNT

        avgRuntimes.append(round(avgRuntime, 3))
        OpRuntimes.append(round(OpRuntime, 3))
        DPcosts.append(round(DPcost, 3))
        Opcosts.append(round(Opcost, 3))
        FCFScosts.append(round(FCFScost, 3))
        Ratios.append(round(DPcost/FCFScost, 3))
        totalFuelConsumptions.append(round(totalFuelConsumption, 3))
        OpFuelConsumptions.append(round(OpFuelConsumption, 3))
        FCFSTotalFuelConsumptions.append(round(totalFcfsFuel, 3))
        # print(totalFuelConsumptions)

    df = pd.DataFrame({
        'DP Total Time Taken': DPcosts,
        'FCFS Total Time Taken': FCFScosts,
        'ODP Total Time Taken': Opcosts,
        'Runtime of DP': avgRuntimes,
        'Runtime of ODP': OpRuntimes,
        'DP Fuel Consumed': totalFuelConsumptions,
        'ODP Fuel Consumed': OpFuelConsumptions

    }, index=np.arange(0, 1.1, 0.1))
    df.index.name = "HV Ratio"
    df.to_csv(RESULTPATH)
