#------------------------------------------------------------------------------
# function:     centrifuge.py                                                 #
# Description:  Function to define centrifuge model equations                 #
#               Material balances, component balances                         #
#                                                                             #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all model equations                                  #
#                                                                             #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
from math import pi
try:
    from . import getParams
except ImportError:
    import getParams

def centrifuge(m):
    m.cf = pyo.Block()

    # Load parameters for centrifuge from getParams
    centrifugeParams = getParams.params['Centrifuge']

    m.cf.costReference              = pyo.Param(initialize = centrifugeParams['Cost Reference'])    # $/m3/s
    m.cf.volumeReference            = pyo.Param(initialize = centrifugeParams['Volume Reference'])  # m3/s
    m.cf.capexFactor                = pyo.Param(initialize = centrifugeParams['Capex Factor'])      # dimensionless
    m.cf.beta                       = pyo.Param(initialize = centrifugeParams['Beta'])
    m.cf.solidDiameter              = pyo.Param(initialize = centrifugeParams['Particle Diameter'])  # m
    m.cf.particleDensity            = pyo.Param(initialize = centrifugeParams.get('Particle Density', 2650))  # kg/m3

    # -------------------- Primary state variables (mass basis) --------------------
    m.cf.sludgeMassFlowIn        = pyo.Var(initialize = 10.0,  within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.cf.sludgeTSSin             = pyo.Var(initialize = 0.01,  within = pyo.NonNegativeReals, bounds = (0, None))  # fraction
    m.cf.sludgeMassFlowOutSolid  = pyo.Var(initialize = 0,     within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.cf.sludgeMassFlowOutLiquid = pyo.Var(initialize = 0,     within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.cf.sludgeTSSoutSolid       = pyo.Var(initialize = 0,     within = pyo.NonNegativeReals, bounds = (0, None))  # fraction
    m.cf.sludgeTSSoutLiquid      = pyo.Var(initialize = 0,     within = pyo.NonNegativeReals, bounds = (0, None))  # fraction

    # -------------------- Vessel geometry / operating variables --------------------
    m.cf.tankVolume          = pyo.Var(initialize = 50, within = pyo.NonNegativeReals, bounds = (0.0, 50.0))  # m3
    m.cf.agitRotation        = pyo.Var(initialize = 100, within = pyo.NonNegativeReals, bounds = (0, 250))   # rps
    m.cf.tankDiameter        = pyo.Var(initialize = 2, within = pyo.NonNegativeReals, bounds = (0, 10))   # m
    m.cf.cakeTSS             = pyo.Var(initialize = 0.25, within = pyo.NonNegativeReals, bounds = (0, 1))  # fraction

    m.cf.sludgeVolFlowIn = pyo.Expression(expr = m.cf.sludgeMassFlowIn / m.model().sludgeDensity)  # m3/s



    # Big-M constraints
    m.cf.M_geo = pyo.Param(initialize=1e4, mutable=True)

    def tankDiameterRule_pos(blk):
        return blk.tankVolume - 1.178*blk.tankDiameter**3 <= blk.M_geo * (1 - blk.model().y_cf)
    m.cf.tankDiameterConstr_pos = pyo.Constraint(rule=tankDiameterRule_pos)

    def tankDiameterRule_neg(blk):
        return blk.tankVolume - 1.178*blk.tankDiameter**3 >= -blk.M_geo * (1 - blk.model().y_cf)
    m.cf.tankDiameterConstr_neg = pyo.Constraint(rule=tankDiameterRule_neg)

    # Centrifuge residence time (s) with epsilon to avoid division-by-zero when bypassed
    m.cf.residenceTime = pyo.Expression(expr = m.cf.tankVolume / (m.cf.sludgeVolFlowIn + 1e-8))  # s

    # Centrifuge Angular Velocity (rad/s) from rotor rotation (rps)
    m.cf.omega = pyo.Expression(expr = 2*pi * m.cf.agitRotation)  # rad/s

    # Centrifugal acceleration factor (dimensionless): G = omega^2 * r / g
    m.cf.G = pyo.Expression(expr = (m.cf.omega**2) * (m.cf.tankDiameter/2) / m.accGravity)

    # Stokes terminal settling velocity under gravity (m/s) for a sphere
    m.cf.vt = pyo.Expression(expr = (m.cf.solidDiameter**2) * (m.cf.particleDensity - m.model().sludgeDensity) * m.accGravity / (18 * m.model().sludgeViscosity))  # m/s

    # centrifugal-enhanced terminal velocity (m/s)
    m.cf.vc = pyo.Expression(expr = m.cf.G * m.cf.vt)  # m/s

    # capture characteristic height and rate: h = beta * d, captureRate = vc / h (1/s)
    m.cf.h = pyo.Expression(expr = m.cf.beta * m.cf.solidDiameter)
    m.cf.captureRate = pyo.Expression(expr = m.cf.vc / m.cf.h)  # 1/s

    # Use effectiveCaptureRate in capture expression
    m.cf.solidMassCaptured = pyo.Expression(expr = 1 - pyo.exp(- m.cf.captureRate * m.cf.residenceTime))

    # -------------------- Mass balances  --------------------
    m.cf.solidsIn = pyo.Expression(expr = m.cf.sludgeMassFlowIn * m.cf.sludgeTSSin)

    # Mass of solids captured in solid stream (kg/s) using the single-size capture fraction
    m.cf.solidsS = pyo.Expression(expr = m.cf.solidMassCaptured * m.cf.solidsIn)

    # Mass of solids lost to liquid stream (kg/s)
    m.cf.solidsL = pyo.Expression(expr = (1 - m.cf.solidMassCaptured) * m.cf.solidsIn)

    # -------------------- Composition-resolved solids --------------------
    m.cf.sludgeSolidsMassFlowIn = pyo.Var(initialize = 1.0, within = pyo.NonNegativeReals)  # kg/s
    m.cf.caoSolidsMassFlowIn    = pyo.Var(initialize = 0.0, within = pyo.NonNegativeReals)  # kg/s

    m.cf.sludgeSolidsCaptured_kg_s = pyo.Expression(expr = m.cf.solidMassCaptured * m.cf.sludgeSolidsMassFlowIn)
    m.cf.caoSolidsCaptured_kg_s    = pyo.Expression(expr = m.cf.solidMassCaptured * m.cf.caoSolidsMassFlowIn)
    m.cf.sludgeSolidsLost_kg_s     = pyo.Expression(expr = (1 - m.cf.solidMassCaptured) * m.cf.sludgeSolidsMassFlowIn)
    m.cf.caoSolidsLost_kg_s        = pyo.Expression(expr = (1 - m.cf.solidMassCaptured) * m.cf.caoSolidsMassFlowIn)
    m.cf.caoWeightFractionInCake = pyo.Expression(expr = m.cf.caoSolidsCaptured_kg_s / (m.cf.solidsS + 1e-9))

    # Overall mass balance 
    def materialBalance(blk):
        return blk.sludgeMassFlowIn == blk.sludgeMassFlowOutSolid + blk.sludgeMassFlowOutLiquid  # unit is kg/s
    m.cf.materialBalance = pyo.Constraint(rule=materialBalance)

    # Cake stream TSS closes the solid mass balance directly
    def solidStreamFlow(blk):
        return blk.sludgeMassFlowOutSolid * blk.sludgeTSSoutSolid == m.cf.solidsS  # unit is kg/s
    m.cf.solidStreamFlow = pyo.Constraint(rule=solidStreamFlow)

    # Liquid effluent TSS closes the solid mass balance directly
    def liquidStreamTSS(blk):
        return blk.sludgeMassFlowOutLiquid * blk.sludgeTSSoutLiquid == m.cf.solidsL  # unit is kg/s
    m.cf.liquidStreamTSS = pyo.Constraint(rule=liquidStreamTSS)

    # TSS concentration balance
    def tssBalanceSolid(blk):
        return blk.sludgeTSSoutSolid == blk.cakeTSS
    m.cf.tssBalanceSolid = pyo.Constraint(rule=tssBalanceSolid)

    # Cake compressibility model parameters
    m.cf.s_compressibility = pyo.Param(initialize=centrifugeParams.get('Compressibility Exponent', 0.5))
    m.cf.P0_ref = pyo.Param(initialize=centrifugeParams.get('Reference Pressure', 1e6))  # Pa
    m.cf.TSS_ref = pyo.Param(initialize=centrifugeParams.get('TSS Reference', 0.2))      # fraction (at P0_ref)

    # Consolidation pressure across the bowl -- physical quantity, uses fixed density param.
    m.cf.P_c = pyo.Expression(expr = 0.5 * m.model().sludgeDensity * m.cf.omega**2 * ((m.cf.tankDiameter/2)**2 - (m.cf.tankDiameter/4)**2))

    def cakeTSS_compressibility(blk):
        return blk.cakeTSS == blk.TSS_ref * (blk.P_c / blk.P0_ref)**blk.s_compressibility
    m.cf.cakeTSS_compr_constr = pyo.Constraint(rule=cakeTSS_compressibility)

    # Physical ceiling on mechanically-achievable cake dryness
    m.cf.maxCakeTSS = pyo.Param(initialize=0.20, mutable=True)  
    m.cf.cakeTSSCeiling = pyo.Constraint(expr = m.cf.cakeTSS <= m.cf.maxCakeTSS)

    # Cost calculations
    m.cf.capex = m.cf.costReference*(m.cf.sludgeVolFlowIn/m.cf.volumeReference)**m.cf.capexFactor

    # Operating Cost
    m.cf.powerNumber = pyo.Param(initialize = centrifugeParams['Power Number']/4)    # dimensionless
    m.cf.power_W = pyo.Expression(expr = m.cf.powerNumber * m.model().sludgeDensity * m.cf.agitRotation**3 * m.cf.tankDiameter**5)
    m.cf.power_kW = pyo.Expression(expr = m.cf.power_W / 1000.0)
    # operating hours over the modeled lifetime (daysOperation is stored in seconds)
    m.cf.operating_hours = pyo.Expression(expr = m.model().daysOperation / 3600.0)
    # OPEX over lifetime: energy (kW) * hours * price ($/kWh)
    m.cf.opex = pyo.Expression(expr = m.cf.power_kW * m.cf.operating_hours * m.elecPrice)