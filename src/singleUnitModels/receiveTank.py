#------------------------------------------------------------------------------
# function:     receiveTank.py                                                #
# Description:  Function to define receiving tank model equations             #
#               Material balances, component balances                        #
#                                                                             #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all receiving tank model equations                   #
#                                                                             #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
except ImportError:
    import getParams

def receiveTank(m):

    m.rt = pyo.Block()

    # Load parameters for receiving tank from getParams
    receiveTankParams = getParams.params['Receiving Tank']

    m.rt.residenceTime   = pyo.Param(initialize = receiveTankParams['Residence Time']) # hr
    m.rt.costReference   = pyo.Param(initialize = receiveTankParams['Cost Reference']) # $/m3
    m.rt.volumeReference = pyo.Param(initialize = receiveTankParams['Volume Reference']) # m3
    m.rt.capexFactor     = pyo.Param(initialize = receiveTankParams['Capex Factor'])   # dimensionless
    m.rt.acidCost        = pyo.Param(initialize = receiveTankParams['Acid Cost'])       # $/kg
    m.rt.acidDensity     = pyo.Param(initialize = receiveTankParams['Acid Density'])    # kg/m3
    m.rt.solidsNFraction  = pyo.Param(initialize = 0.05, mutable = True)                 # kg-N/kg-dry solids
    m.rt.sludgepHin = pyo.Var(initialize = 13.0, within = pyo.NonNegativeReals, bounds = (0, 14))
    m.rt.targetpH   = pyo.Param(initialize = 7.0, mutable = True)
    m.rt.freeOhMolPerL = pyo.Expression(expr=10**(m.rt.sludgepHin - 14))  # 
    m.rt.molwtNkgPerMol   = pyo.Param(initialize = 0.014, mutable = True)                # kg/mol
    m.rt.molwtH2SO4kgPerMol = pyo.Param(initialize = 0.098, mutable = True)              # kg/mol
    m.rt.acidSolutionWtFraction = pyo.Param(initialize = 0.93, mutable = True)           # wt% H2SO4 solution basis
    m.rt.residualOHMolPerM3 = pyo.Param(initialize=100.0, mutable=True)                  # mol OH- / m3 liquid

    # -------------------- Primary state variables (mass basis) --------------------
    m.rt.sludgeMassFlowIn  = pyo.Var(initialize = 10.0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.rt.acidMassFlowIn    = pyo.Var(initialize = 0.1,  within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.rt.sludgeTSSin       = pyo.Var(initialize = 0.01, within = pyo.NonNegativeReals, bounds = (0, None))  # fraction
    m.rt.sludgeMassFlowOut = pyo.Var(initialize = 0, within = pyo.NonNegativeReals, bounds = (0, None))  # kg/s
    m.rt.sludgepHout   = pyo.Var(initialize = 7, within = pyo.NonNegativeReals, bounds = (0, 14))
    m.rt.sludgeTSSout  = pyo.Var(initialize = 0.05, within = pyo.NonNegativeReals, bounds = (0, 1))  # fraction
    m.rt.tankVolume    = pyo.Var(initialize = 50, within = pyo.NonNegativeReals, bounds = (0, None))  # m3
    m.rt.nitrogenConcIn  = pyo.Var(initialize = 600.0, within = pyo.NonNegativeReals)  # g-N/m3 (= ppm)
    m.rt.nitrogenConcOut = pyo.Var(initialize = 560.0, within = pyo.NonNegativeReals)  # g-N/m3 (= ppm)

    # Composition-resolved solids input, set by the connecting flowsheet 
    m.rt.sludgeSolidsMassFlowIn = pyo.Var(initialize = 1.0, within = pyo.NonNegativeReals)  # kg/s

    m.rt.minCapex = pyo.Param(initialize=10000.0, mutable=True)

    # Capital Cost
    m.rt.capex = m.rt.minCapex + 1.64*m.rt.costReference*(m.rt.tankVolume/m.rt.volumeReference)**m.rt.capexFactor

    # -------------------- Stoichiometric acid bookkeeping --------------------
    # Total solids mass (sludge + any CaO carried through) 
    m.rt.solidsMassKgPerS = pyo.Expression(expr = m.rt.sludgeMassFlowIn * m.rt.sludgeTSSin)
    m.rt.liquidMassKgPerS = pyo.Expression(expr = m.rt.sludgeMassFlowIn * (1.0 - m.rt.sludgeTSSin))
    # Liquid volumetric flow is genuinely required here (mol/L-based acid-base chemistry)
    m.rt.liquidVolumeFlow_m3s = pyo.Expression(expr = m.rt.liquidMassKgPerS / m.model().liquidDensity)
    m.rt.liquidVolFlowLPerS   = pyo.Expression(expr = m.rt.liquidVolumeFlow_m3s * 1000.0)

    # CORRECTED: N-content basis uses sludgeSolidsMassFlowIn (organic solids
    # only), not solidsMassKgPerS (which includes CaO, carrying zero N).
    m.rt.solidsNMassFlowKgPerS = pyo.Expression(expr = m.rt.solidsNFraction * m.rt.sludgeSolidsMassFlowIn)
    m.rt.inletLiquidNMassFlowKgPerS = pyo.Expression(expr = m.rt.nitrogenConcIn * m.rt.liquidVolFlowLPerS / 1e6)
    m.rt.freeOhMolPerS = pyo.Expression(expr = m.rt.freeOhMolPerL * m.rt.liquidVolFlowLPerS) 
    m.rt.liquidNhMolPerS = pyo.Expression(expr = (m.rt.inletLiquidNMassFlowKgPerS + m.rt.solidsNMassFlowKgPerS) / m.rt.molwtNkgPerMol)
    m.rt.targetOhMolPerS = pyo.Expression(expr = 10**(m.rt.targetpH - 14) * m.rt.liquidVolFlowLPerS) 
    m.rt.ohNeutralizationDemand = pyo.Expression(
        expr=m.rt.residualOHMolPerM3 * m.rt.liquidVolumeFlow_m3s
    )
    # Sludge buffering capacity 
    m.rt.alkalinityBufferMolPerM3 = pyo.Param(initialize=20.0, mutable=True)  # mol H+ demand / m3 liquid
    m.rt.bufferingAcidDemandMolPerS = pyo.Expression(
        expr=m.rt.alkalinityBufferMolPerM3 * m.rt.liquidVolumeFlow_m3s
    )
    m.rt.totalAcidEqPerS = pyo.Expression(
        expr=m.rt.ohNeutralizationDemand + m.rt.liquidNhMolPerS + m.rt.bufferingAcidDemandMolPerS
    )
    m.rt.h2so4RequiredMolPerS = pyo.Expression(expr = m.rt.totalAcidEqPerS / 2.0)
    m.rt.h2so4RequiredKgPerS = pyo.Expression(expr = m.rt.h2so4RequiredMolPerS * m.rt.molwtH2SO4kgPerMol)
    m.rt.acidSolutionMassFlowKgPerS = pyo.Expression(expr = m.rt.h2so4RequiredKgPerS / m.rt.acidSolutionWtFraction)

    # Operating Cost
    m.rt.opex = m.rt.acidCost * m.rt.h2so4RequiredKgPerS * m.daysOperation

    # -------------------- Sizing-only volumetric conversion --------------------
    m.rt.sludgeVolFlowIn = pyo.Expression(expr = m.rt.sludgeMassFlowIn / m.model().sludgeDensity)  # m3/s

    def receiveTankVolumeRule(blk):
        return blk.tankVolume == blk.residenceTime*blk.sludgeVolFlowIn  # unit is m3
    m.rt.receiveTankVolume = pyo.Constraint(rule=receiveTankVolumeRule)

    # Acid mass flow is now set directly and stoichiometrically 
    def acidFlowRule(blk):
        return blk.acidMassFlowIn == blk.acidSolutionMassFlowKgPerS
    m.rt.acidFlowConstr = pyo.Constraint(rule=acidFlowRule)

    # mass balance 
    def materialBalance(blk):
        return blk.sludgeMassFlowOut == blk.sludgeMassFlowIn + blk.acidMassFlowIn
    m.rt.materialBalance = pyo.Constraint(rule=materialBalance)

    def componentBalance(blk):
        return blk.sludgepHout == blk.targetpH
    m.rt.componentBalance = pyo.Constraint(rule=componentBalance)

    # Outlet TSS concentration 
    def outletTSS_rule(blk):
            solids_in = blk.sludgeMassFlowIn * blk.sludgeTSSin
            solids_out = blk.sludgeMassFlowOut * blk.sludgeTSSout
            return solids_out == solids_in
    m.rt.outletTSS = pyo.Constraint(rule=outletTSS_rule)

    # Liquid-phase nitrogen balance, expressed as explicit N mass conservation
    m.rt.liquidVolFlowOut = pyo.Expression(expr=(m.rt.sludgeMassFlowOut * (1.0 - m.rt.sludgeTSSout)) / m.model().liquidDensity)
    m.rt.nitrogenMassFlowIn_kgN_s  = pyo.Expression(expr=m.rt.nitrogenConcIn  * m.rt.liquidVolumeFlow_m3s / 1000.0)
    m.rt.nitrogenMassFlowOut_kgN_s = pyo.Expression(expr=m.rt.nitrogenConcOut * m.rt.liquidVolFlowOut     / 1000.0)

    def nitrogenBalance(blk):
        return blk.nitrogenMassFlowIn_kgN_s == blk.nitrogenMassFlowOut_kgN_s
    m.rt.nitrogenBalance = pyo.Constraint(rule=nitrogenBalance)