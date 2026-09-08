#------------------------------------------------------------------------------
# function:     electrolyzer.py                                               #
# Description:  Function to define electrolyzer model equations               #
#               Material balances, component balances                         #
#                                                                             #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all model equations                                  #
#                                                                             #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
except ImportError:
    import getParams

def electrolyzer(m):

    m.el = pyo.Block()

    # Load parameters for electrolyzer from getParams
    electrolyzerParams = getParams.params['Electrolyzer']

    m.el.residenceTime    = pyo.Param(initialize = electrolyzerParams['Residence Time'])   # s
    m.el.costReference    = pyo.Param(initialize = electrolyzerParams['Cost Reference'])    # $/m2
    m.el.areaReference    = pyo.Param(initialize = electrolyzerParams['Area Reference'])    # m2
    m.el.capexFactor      = pyo.Param(initialize = electrolyzerParams['Capex Factor'])      # dimensionless
    m.el.SEC              = pyo.Param(initialize = electrolyzerParams['Specific Energy Consumption']) # kWh/kg-N

    # -------------------- Primary state variables --------------------
    m.el.sludgeMassFlowIn  = pyo.Var(initialize = 10.0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.el.sludgeMassFlowOut = pyo.Var(initialize = 0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.el.nitrogenFlowOut   = pyo.Var(initialize = 0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg-N/s
    m.el.phosphorusFlowOut = pyo.Var(initialize = 0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg-P/s
    m.el.sludgeTSSin       = pyo.Var(initialize = 0.01, within = pyo.NonNegativeReals, bounds = (0, None))  # fraction
    m.el.sludgeTSSout      = pyo.Var(initialize = 0.05, within = pyo.NonNegativeReals, bounds = (0, 1))  # fraction
    m.el.area              = pyo.Var(initialize = 5, within=pyo.NonNegativeReals)  # m2
    m.el.nitrogenConcIn    = pyo.Var(initialize = 300.0, within = pyo.NonNegativeReals)  # g-N/m3 (= ppm)
    m.el.nitrogenConcOut   = pyo.Var(initialize = 600.0, within = pyo.NonNegativeReals)  # g-N/m3 (= ppm)
    m.el.sludgepHIn        = pyo.Var(initialize = 7, within = pyo.NonNegativeReals, bounds = (0, 14))  # pH of sludge in
    m.el.sludgepHOut       = pyo.Var(initialize = 7, within = pyo.NonNegativeReals, bounds = (0, 14))  # pH of sludge out

    # Composition-resolved solids input 
    m.el.sludgeSolidsMassFlowIn = pyo.Var(initialize = 1.0, within = pyo.NonNegativeReals)  # kg/s
    m.el.solidsNMassFrac = pyo.Param(initialize=0.05, mutable=True)  # kg-N/kg-sludge-solids
    # Fraction of solid-bound N liberated to the liquid phase during electrolysis
    m.el.solidsNToLiquidFrac = pyo.Param(initialize=0.219, mutable=True)

    # Dry solids throughput (kg/s) — used for sizing
    m.el.sludgeOutDryBasis = pyo.Expression(expr=m.el.sludgeMassFlowOut * m.el.sludgeTSSout)  # kg/s

    # --- Area sizing from experimental capacity basis (daily units) ---
    m.el.sludgeOutDryBasis_kgPerDay = pyo.Expression(expr=m.el.sludgeOutDryBasis * 86400.0)

    m.el.elCapacityKgDsPerM2PerBatch = pyo.Param(initialize=14.286, mutable=True)
    m.el.batchDurationHr = pyo.Param(initialize=0.5, mutable=True)
    m.el.cleaningDowntimeFrac = pyo.Param(initialize=0, mutable=True)
    m.el.operationHoursPerDay = pyo.Param(initialize=24.0, mutable=True)

    m.el.batchesPerDay = pyo.Expression(
        expr=m.el.operationHoursPerDay * (1.0 - m.el.cleaningDowntimeFrac) / m.el.batchDurationHr
    )

    m.el.dailyCapacityPerM2 = pyo.Expression(
        expr=m.el.elCapacityKgDsPerM2PerBatch * m.el.batchesPerDay
    )

    m.el.elAreaRequired = pyo.Expression(
        expr=m.el.sludgeOutDryBasis_kgPerDay / (m.el.dailyCapacityPerM2 + 1e-9)
    )

    m.el.areaFromCapacity = pyo.Constraint(expr=m.el.area == m.el.elAreaRequired)

    # --- Component-based CAPEX ---
    m.el.stackUnitCost = pyo.Param(initialize=6000.0, mutable=True)  # $/m2
    m.el.stackCapex = pyo.Expression(expr=m.el.stackUnitCost * m.el.area)

    m.el.currentDensity = pyo.Param(initialize=30.0, mutable=True)  # A/m2
    m.el.powerSourceUnitCost = pyo.Param(initialize=20.0, mutable=True)  # $/A
    m.el.totalCurrent = pyo.Expression(expr=m.el.currentDensity * m.el.area)
    m.el.powerSourceCapex = pyo.Expression(expr=m.el.powerSourceUnitCost * m.el.totalCurrent)

    m.el.pumpUnitCost = pyo.Param(initialize=22000.0, mutable=True)  # $/pump
    m.el.areaPerPump = pyo.Param(initialize=3.0, mutable=True)  # m2 per pump
    m.el.numPumps = pyo.Expression(expr=m.el.area / m.el.areaPerPump)
    m.el.pumpCapex = pyo.Expression(expr=m.el.pumpUnitCost * m.el.numPumps)

    m.el.tankPairCost = pyo.Param(initialize=8990.0, mutable=True)  # $/pair
    m.el.areaPerTankPair = pyo.Param(initialize=4.714, mutable=True)  # m2 per pair
    m.el.numTankPairs = pyo.Expression(expr=m.el.area / m.el.areaPerTankPair)
    m.el.tankCapex = pyo.Expression(expr=m.el.tankPairCost * m.el.numTankPairs)

    m.el.capex = pyo.Expression(
        expr= (2 * 1.64 *m.el.stackCapex + m.el.pumpCapex + m.el.tankCapex + m.el.powerSourceCapex) # 2 stacks per electrolyzer, 1.64x stack cost for balance-of-plant
    )

    # Operating Cost -- m.el.power in true kW (SEC[kWh/kg-N] * nitrogenFlowOut[kg-N/s] * 3600 = kW).
    m.el.power = m.el.SEC * m.el.nitrogenFlowOut * 3600.0  # kW
    m.el.opex = m.el.power * m.elecPrice * (m.daysOperation / 3600.0)  # $

    # Material Balance (mass, complete retention -- no solids captured/lost here)
    def materialBalance(blk):
        return blk.sludgeMassFlowOut == blk.sludgeMassFlowIn
    m.el.materialBalance = pyo.Constraint(rule=materialBalance)

    # Nitrogen Balance 
    def nitrogenBalance(blk):
        return blk.nitrogenFlowOut == blk.solidsNMassFrac * blk.sludgeSolidsMassFlowIn
    m.el.nitrogenBalance = pyo.Constraint(rule=nitrogenBalance)

    # Phosphorus Balance 
    def phosphorusBalance(blk):
        return blk.phosphorusFlowOut == 0.0179*blk.sludgeMassFlowIn
    m.el.phosphorusBalance = pyo.Constraint(rule=phosphorusBalance)

    # Sludge TSS Balance
    def sludgeTSSBalance(blk):
        return blk.sludgeTSSout == blk.sludgeTSSin
    m.el.sludgeTSSBalance = pyo.Constraint(rule=sludgeTSSBalance)

    # Liquid-phase nitrogen balance, expressed as explicit N mass conservation:
    # dissolved N in + liberated solids-N == dissolved N out
    m.el.liquidVolFlowIn  = pyo.Expression(expr=(m.el.sludgeMassFlowIn  * (1.0 - m.el.sludgeTSSin))  / m.model().liquidDensity)
    m.el.liquidVolFlowOut = pyo.Expression(expr=(m.el.sludgeMassFlowOut * (1.0 - m.el.sludgeTSSout)) / m.model().liquidDensity)
    m.el.nitrogenMassFlowInLiquid_kgN_s  = pyo.Expression(expr=m.el.nitrogenConcIn  * m.el.liquidVolFlowIn  / 1000.0)
    m.el.nitrogenMassFlowOutLiquid_kgN_s = pyo.Expression(expr=m.el.nitrogenConcOut * m.el.liquidVolFlowOut / 1000.0)

    # -------------------------------------------------------------------
    # Ammonia volatilization 
    m.el.nitrogenLossFraction = pyo.Param(initialize=0.05, mutable=True)  # combined volatilization + oxidation

    m.el.nitrogenMassFlowAvailForLoss_kgN_s = pyo.Expression(
        expr=m.el.nitrogenMassFlowInLiquid_kgN_s + m.el.solidsNToLiquidFrac * m.el.nitrogenFlowOut
    )
    m.el.nitrogenFlowVolatilized_kgN_s = pyo.Expression(
        expr=m.el.nitrogenLossFraction * m.el.nitrogenMassFlowAvailForLoss_kgN_s
    )

    # Liquid-phase nitrogen balance, expressed as explicit N mass conservation:
    # dissolved N in + liberated solids-N == dissolved N out + lost N.
    def liquidNitrogenBalance(blk):
        return (blk.nitrogenMassFlowInLiquid_kgN_s + blk.solidsNToLiquidFrac * blk.nitrogenFlowOut
                == blk.nitrogenMassFlowOutLiquid_kgN_s + blk.nitrogenFlowVolatilized_kgN_s)
    m.el.liquidNitrogenBalance = pyo.Constraint(rule=liquidNitrogenBalance)

    def pHBalance(blk):
        return blk.sludgepHOut == blk.sludgepHIn
    m.el.pHBalance = pyo.Constraint(rule=pHBalance)