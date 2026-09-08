#------------------------------------------------------------------------------
# function:     prepTank.py                                                   #
# Description:  Function to define preparation tank model equations           #
#               Material balances, component balances                         # 
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all preparation tank model equations                 #
#                                                                             #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
except ImportError:
    import getParams

def prepTank(m):

    m.pt = pyo.Block()

    # Load parameters for prep tank from getParams
    prepTankParams = getParams.params['Preparation Tank']

    m.pt.residenceTime            = pyo.Param(initialize = prepTankParams['Residence Time']) # hr
    m.pt.costReference            = pyo.Param(initialize = prepTankParams['Cost Reference']) # $/m3
    m.pt.volumeReference          = pyo.Param(initialize = prepTankParams['Volume Reference']) # m3
    m.pt.capexFactor              = pyo.Param(initialize = prepTankParams['Capex Factor'])   # dimensionless
    m.pt.baseCost                 = pyo.Param(initialize = prepTankParams['Base Cost'])       # $/kg
    m.pt.baseDensity              = pyo.Param(initialize = prepTankParams['Base Density'])    # kg/m3
    m.pt.baseSolubility           = pyo.Param(initialize = prepTankParams['Base Solubility'])  # kg/m3
    m.pt.limeTankCostReference    = pyo.Param(initialize = prepTankParams['Lime Tank Cost'])    # $/m3
    m.pt.limeTankVolumeReference  = pyo.Param(initialize = prepTankParams['Lime Tank Volume'])  # m3
    m.pt.limeTankCapexFactor      = pyo.Param(initialize = prepTankParams['Lime Tank Capex Factor'])      # dimensionless
    m.pt.sludgeTSSOutTarget       = pyo.Param(initialize = prepTankParams['Target TSS'], mutable=True)  # fraction

    # --- Parameters for two-part CaO calculation ---
    # A) Dewatered sludge stream: experimental ratio per kg DRY SOLIDS
    m.pt.caoPerKgDS        = pyo.Param(initialize=0.224, mutable=True)    # kg CaO / kg dry solids
    m.pt.sludgeOnlyDensity = pyo.Param(initialize=1200.0, mutable=True)   # kg/m3 (dewatered sludge, physical const)
    m.pt.sludgeOnlyTSS     = pyo.Param(initialize=0.20, mutable=True)    # fraction (dewatered sludge TS)
    m.pt.centrateDensity   = pyo.Param(initialize=1000.0, mutable=True)   # kg/m3 (physical const)

    # B) Centrate stream: pH chemistry
    m.pt.alkalinityBuffer  = pyo.Param(initialize=20.0, mutable=True)   # mol OH- / m3 liquid
    m.pt.targetOHConc      = pyo.Param(initialize=100.0, mutable=True)  # mol OH- / m3 at pH 13
    m.pt.mwCaO             = pyo.Param(initialize=0.05608)               # kg/mol

    m.pt.minCapex = pyo.Param(initialize=10000.0, mutable=True)
    m.pt.minLimeTankCapex = pyo.Param(initialize=5000.0, mutable=True)

    # Sub-stream flows, linked from flowsheet level
    m.pt.sludgeOnlyFlow_m3s = pyo.Var(initialize=0.001, within=pyo.NonNegativeReals)  # m3/s, dewatered sludge only
    m.pt.centrateFlow_m3s   = pyo.Var(initialize=0.001, within=pyo.NonNegativeReals)  # m3/s, centrate only

    # -------------------- Primary state variables --------------------
    m.pt.sludgeMassFlowIn  = pyo.Var(initialize = 10.0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.pt.waterMassFlowIn   = pyo.Var(initialize = 10.0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s (decision var)
    m.pt.sludgeTSSin       = pyo.Var(initialize = 0.01, within = pyo.NonNegativeReals, bounds = (0, None))  # fraction
    m.pt.sludgeMassFlowOut = pyo.Var(initialize = 0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.pt.sludgepHOut       = pyo.Var(initialize = 7, within = pyo.NonNegativeReals, bounds = (0, 14))
    m.pt.sludgeTSSout      = pyo.Var(initialize = 0.05, within = pyo.NonNegativeReals, bounds = (0, 1))  # fraction
    m.pt.tankVolume        = pyo.Var(initialize = 50, within = pyo.NonNegativeReals, bounds = (0, None))  # m3
    m.pt.limeTankVolume    = pyo.Var(initialize = 20, within = pyo.NonNegativeReals, bounds = (0, None))  # m3
    m.pt.nitrogenConcIn    = pyo.Var(initialize = 750.0, within = pyo.NonNegativeReals)  # g-N/m3 (= ppm)
    m.pt.nitrogenConcOut   = pyo.Var(initialize = 300.0, within = pyo.NonNegativeReals)  # g-N/m3 (= ppm)

    # --- Two-part CaO calculation ---
    m.pt.sludgeDrySolidsMassFlow_kgPerS = pyo.Expression(
        expr=m.pt.sludgeOnlyFlow_m3s * m.pt.sludgeOnlyDensity * m.pt.sludgeOnlyTSS
    )
    m.pt.caoForSludge_kgPerS = pyo.Expression(
        expr=m.pt.caoPerKgDS * m.pt.sludgeDrySolidsMassFlow_kgPerS
    )

    m.pt.centrateOhDemandNH4 = pyo.Expression(
        expr=m.pt.nitrogenConcIn / 14.007 * m.pt.centrateFlow_m3s    # nitrogenConcIn in g-N/m3
    )
    m.pt.centrateOhDemandResidual = pyo.Expression(
        expr=m.pt.targetOHConc * m.pt.centrateFlow_m3s
    )
    m.pt.centrateOhDemandAlkalinity = pyo.Expression(
        expr=m.pt.alkalinityBuffer * m.pt.centrateFlow_m3s
    )
    m.pt.centrateTotalOhDemand = pyo.Expression(
        expr=m.pt.centrateOhDemandNH4 + m.pt.centrateOhDemandResidual + m.pt.centrateOhDemandAlkalinity
    )
    m.pt.caoForCentrate_kgPerS = pyo.Expression(
        expr=(m.pt.centrateTotalOhDemand / 2.0) * m.pt.mwCaO   # 1 mol CaO -> 2 mol OH-
    )

    m.pt.totalCaO_kgPerS = pyo.Expression(
        expr=m.pt.caoForSludge_kgPerS + m.pt.caoForCentrate_kgPerS
    )

    # Capital Cost of prep tank and lime tank
    m.pt.capexTank = m.pt.minCapex + m.pt.costReference*(m.pt.tankVolume/m.pt.volumeReference)**m.pt.capexFactor
    m.pt.capexLimeTank = m.pt.minLimeTankCapex + m.pt.limeTankCostReference*(m.pt.limeTankVolume/m.pt.limeTankVolumeReference)**m.pt.limeTankCapexFactor

    m.pt.capex = 1.64*(m.pt.capexTank + m.pt.capexLimeTank)

    # Operating Cost (CaO chemical cost only; already mass-based)
    m.pt.opex = m.pt.baseCost*m.pt.totalCaO_kgPerS*m.daysOperation

    # -------------------- Sizing-only volumetric conversions --------------------
    m.pt.sludgeVolFlowIn = pyo.Expression(expr = m.pt.sludgeMassFlowIn / m.model().sludgeDensity)  # m3/s
    m.pt.waterVolFlowIn  = pyo.Expression(expr = m.pt.waterMassFlowIn / 1000.0)                    # m3/s (fixed water density)
    m.pt.baseVolFlowIn   = pyo.Expression(expr = m.pt.totalCaO_kgPerS / m.pt.baseDensity)           # m3/s

    def prepTankVolumeRule(blk):
        return blk.tankVolume == blk.residenceTime*(blk.sludgeVolFlowIn + blk.waterVolFlowIn + blk.baseVolFlowIn)  # unit is m3
    m.pt.prepTankVolume = pyo.Constraint(rule=prepTankVolumeRule)

    def limeTankVolumeRule(blk):
        return blk.limeTankVolume == blk.residenceTime*blk.baseVolFlowIn  # unit is m3
    m.pt.limeTankVolumeConstr = pyo.Constraint(rule=limeTankVolumeRule)

    # -------------------- True mass balance --------------------
    def materialBalance(blk):
        return blk.sludgeMassFlowOut == blk.sludgeMassFlowIn + blk.waterMassFlowIn + m.pt.totalCaO_kgPerS  # unit is kg/s
    m.pt.materialBalance = pyo.Constraint(rule=materialBalance)

    # Dry-solids mass flow in (kg/s) -- direct, no density round-trip
    m.pt.massInlet = pyo.Expression(expr=m.pt.sludgeMassFlowIn * m.pt.sludgeTSSin)

    # CaO solids: undissolved fraction contributes to outlet solids
    m.pt.liquidVolFlowOut = pyo.Expression(
        expr=(m.pt.sludgeMassFlowOut * (1.0 - m.pt.sludgeTSSout)) / m.model().liquidDensity
    )
    m.pt.dissolvedCaO_kgPerS = pyo.Expression(
        expr=m.pt.baseSolubility * m.pt.liquidVolFlowOut
    )
    m.pt.baseSolidsFlow = pyo.Expression(
        expr=m.pt.totalCaO_kgPerS - m.pt.dissolvedCaO_kgPerS
    )

    # Total solids in outlet = sludge solids + undissolved CaO solids
    m.pt.totalOutletSolids = pyo.Expression(expr=m.pt.massInlet + m.pt.baseSolidsFlow)

    # Composition-resolved solids outputs
    m.pt.sludgeSolidsMassFlowOut = pyo.Expression(expr=m.pt.massInlet)      # kg/s, organic/sludge solids only
    m.pt.caoSolidsMassFlowOut    = pyo.Expression(expr=m.pt.baseSolidsFlow) # kg/s, undissolved CaO solids only

    # Outlet TSS from mass balance 
    def outletTSSBalance(blk):
        return blk.sludgeTSSout * blk.sludgeMassFlowOut == m.pt.totalOutletSolids
    m.pt.outletTSSBalance = pyo.Constraint(rule=outletTSSBalance)

    # Nitrogen balance, expressed as explicit N mass conservation 
    m.pt.liquidVolFlowIn = pyo.Expression(
        expr=(m.pt.sludgeMassFlowIn * (1.0 - m.pt.sludgeTSSin)) / m.model().liquidDensity
    )
    m.pt.nitrogenMassFlowIn_kgN_s  = pyo.Expression(expr=m.pt.nitrogenConcIn  * m.pt.liquidVolFlowIn  / 1000.0)
    m.pt.nitrogenMassFlowOut_kgN_s = pyo.Expression(expr=m.pt.nitrogenConcOut * m.pt.liquidVolFlowOut / 1000.0)

    def nitrogenBalance(blk):
        return blk.nitrogenMassFlowIn_kgN_s == blk.nitrogenMassFlowOut_kgN_s
    m.pt.nitrogenBalance = pyo.Constraint(rule=nitrogenBalance)

    # Component Balance
    def componentBalance(blk):
        return blk.sludgepHOut == 13  
    m.pt.componentBalance = pyo.Constraint(rule=componentBalance)

    # Optional outlet TSS target — activate/deactivate from the flowsheet as needed
    def targetTSSRule(blk):
        return blk.sludgeTSSout == blk.sludgeTSSOutTarget
    m.pt.targetTSSConstr = pyo.Constraint(rule=targetTSSRule)
    m.pt.targetTSSConstr.deactivate()  # OFF by default