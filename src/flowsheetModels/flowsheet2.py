#------------------------------------------------------------------------------
# function:    flowsheet2.py                                                  #
# Description: Flowsheet 2 based on Electrolysis FS2 - Modified               #
#              Process flow: Feed -> Prep Tank -> Electrolyzer -> NF          #
#              NF concentrate -> Acid Tank (E-GROW stream)                    #
#              NF permeate -> HFMC (GPM surrogate) -> Dryer -> Amm. sulfate   #
#------------------------------------------------------------------------------

import os
import sys

# Ensure repo root is importable when this file is run directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pyomo.environ as pyo

import src.singleUnitModels.getParams as getParams
import src.singleUnitModels.getStandardParams as getStandardParams
import src.singleUnitModels.prepTank as prepTank
import src.singleUnitModels.electrolyzer as electrolyzer
import src.singleUnitModels.goMembraneDewatering as goMembraneDewatering
import src.singleUnitModels.nanofiltration as nf
import src.singleUnitModels.receiveTank as receiveTank
import src.singleUnitModels.gpm as gpm
import src.singleUnitModels.dryer as dryer
import src.singleUnitModels.storageTank as storageTank


# ------------------------------
# Model and feed specifications
# ------------------------------
m = pyo.ConcreteModel()

try:
    getParams.getParams(m)
except Exception:
    pass
getStandardParams.getStandardParams(m)

# Two inlet feed streams (dewatered sludge + centrate)
m.dewateredSludgeFlow_m3s = pyo.Param(initialize=76.5/86400.0,  mutable=True)  # m3/s
m.dewateredSludgeDensity  = pyo.Param(initialize=1200.0,         mutable=True)  # kg/m3
m.dewateredSludgeTSS      = pyo.Param(initialize=0.20,           mutable=True)  # mass fraction (20% TS)
m.centrateFlow_m3s        = pyo.Param(initialize=214/86400.0,  mutable=True)  # m3/s
m.centrateDensity         = pyo.Param(initialize=1000.0,         mutable=True)  # kg/m3
m.centrateTSS             = pyo.Param(initialize=0.0,            mutable=True)  # mass fraction (0% TS)

# Combined feed properties computed from the two streams
m.feedMassFlow_kg_s = pyo.Expression(
    expr=m.dewateredSludgeFlow_m3s * m.dewateredSludgeDensity
       + m.centrateFlow_m3s * m.centrateDensity
)
m.feedDrySolids_kg_s = pyo.Expression(
    expr=m.dewateredSludgeFlow_m3s * m.dewateredSludgeDensity * m.dewateredSludgeTSS
)
m.feedFlow_m3s = pyo.Expression(
    expr=m.dewateredSludgeFlow_m3s + m.centrateFlow_m3s
)
m.feedTSS = pyo.Expression(
    expr=m.feedDrySolids_kg_s / m.feedMassFlow_kg_s
)

m.feedNitrogenConc = pyo.Param(initialize=750.0, mutable=True)         # 750 g-N/m3 (from centrate)


# ------------------------------
# Unit blocks
# ------------------------------
prepTank.prepTank(m)
electrolyzer.electrolyzer(m)
goMembraneDewatering.goMembraneDewatering(m)
receiveTank.receiveTank(m)      # used as acid tank for E-GROW stream
nf.nf(m)                        # bivalent (Ca/Mg) removal NF stage, applied to GO permeate liquid
gpm.gpm(m)                      # used as GPM
dryer.dryer(m)
storageTank.storageTank(m)

m.pt.sludgeTSSOutTarget.set_value(0.08)  # change to 8%
m.pt.targetTSSConstr.deactivate()

# NF retentate TSS is free — combinedProductTSSTarget (added below) drives the
# optimizer to dewater enough to compensate for the GPM product liquid dilution.
m.go.del_component(m.go.targetSolidsConstraint)

# ------------------------------
# Flowsheet connections
# ------------------------------

# Feed -> Prep Tank
m.feed_to_pt_flow = pyo.Constraint(expr=m.pt.sludgeMassFlowIn == m.feedMassFlow_kg_s)
m.feed_to_pt_tss  = pyo.Constraint(expr=m.pt.sludgeTSSin == m.feedTSS)
m.feed_to_pt_n    = pyo.Constraint(expr=m.pt.nitrogenConcIn == m.feedNitrogenConc)
# Link sub-stream flows to prep tank for split CaO calculation
m.linkSludgeToPt   = pyo.Constraint(expr=m.pt.sludgeOnlyFlow_m3s == m.dewateredSludgeFlow_m3s)
m.linkCentrateToPt = pyo.Constraint(expr=m.pt.centrateFlow_m3s   == m.centrateFlow_m3s)

# Prep Tank -> Electrolyzer
m.pt_to_el_flow = pyo.Constraint(expr=m.el.sludgeMassFlowIn == m.pt.sludgeMassFlowOut)
m.pt_to_el_tss  = pyo.Constraint(expr=m.el.sludgeTSSin == m.pt.sludgeTSSout)
m.pt_to_el_n    = pyo.Constraint(expr=m.el.nitrogenConcIn == m.pt.nitrogenConcOut)
# Composition link 
m.pt_to_el_sludgeSolids = pyo.Constraint(expr=m.el.sludgeSolidsMassFlowIn == m.pt.sludgeSolidsMassFlowOut)
# pH link 
m.pt_to_el_ph = pyo.Constraint(expr=m.el.sludgepHIn == m.pt.sludgepHOut)

# Electrolyzer -> NanoFiltration
m.el_to_go_flow = pyo.Constraint(expr=m.go.sludgeMassFlowIn == m.el.sludgeMassFlowOut)
m.el_to_go_tss  = pyo.Constraint(expr=m.go.sludgeTSSin == m.el.sludgeTSSout)
m.el_to_go_ph   = pyo.Constraint(expr=m.go.sludgepHIn == m.el.sludgepHOut)

# go concentrate (retentate) -> Acid Tank (E-GROW line)
m.go_to_rt_flow = pyo.Constraint(expr=m.rt.sludgeMassFlowIn == m.go.sludgeMassFlowOut)
m.go_to_rt_tss  = pyo.Constraint(expr=m.rt.sludgeTSSin == m.go.sludgeTSSout)
# carry dissolved N concentration through concentrate line
m.go_to_rt_n    = pyo.Constraint(expr=m.rt.nitrogenConcIn == m.el.nitrogenConcOut)
# Composition link 
m.go_to_rt_sludgeSolids = pyo.Constraint(expr=m.rt.sludgeSolidsMassFlowIn == m.pt.sludgeSolidsMassFlowOut)

# go permeate -> nf (bivalent removal) -> GPM
m.go_to_nf_flow = pyo.Constraint(expr=m.nf.massFlowIn == m.go.permeateMassFlow)
m.go_to_nf_pH   = pyo.Constraint(expr=m.nf.pHIn   == m.go.liquidpHOut)

m.nf_to_gpm_flow    = pyo.Constraint(expr=m.gpm.massFlowIn == m.nf.permeateMassFlow)
m.nf_to_gpm_inletpH = pyo.Constraint(expr=m.gpm.inletpH == m.nf.pHOut)

# Require at least 90% nitrogen capture across GPM (based on fresh feed inlet).
m.gpm_capture_target = pyo.Constraint(
    expr=m.gpm.concOut <= 0.10 * m.gpm.concInFresh
)

# Dryer disconnected after GPM, since it is added to EGROW product stream
m.dr.sludgeSolidsFlowIn_mass.fix(0.0)
m.dr.caoSolidsFlowIn_mass.fix(0.0)
m.dr.waterIn_mass.fix(0.0)
m.dr.nitrogenMassFlowIn_kgN_s.fix(0.0)
m.dr.pHIn.fix(7.0)
m.dr.finalSolids_frac.fix(0.25)
m.dr.airFlowIn.fix(0.0)

# E-GROW product -> Storage Tank (size for 2 days of product storage)
m.rt_to_st_flow = pyo.Constraint(expr=m.st.productMassFlowIn == m.rt.sludgeMassFlowOut)


# E-GROW retentate stream at acid tank outlet (before GPM mixing)
m.egrow_flow_kg_day = pyo.Expression(expr=m.rt.sludgeMassFlowOut * 86400.0)

# Estimated final E-GROW product N (wet basis).
m.productSolidsNMassFrac    = pyo.Param(initialize=0.05, mutable=True)   # kg-N/kg-solids
m.solidsNToLiquidFrac_total = pyo.Param(initialize=0.219, mutable=True)
m.solidsNToLiquidFrac_TAN   = pyo.Param(initialize=0.0455, mutable=True)
m.productLiquidNConc_g_m3 = pyo.Expression(expr=m.feedNitrogenConc)  # g-N/m3

m.egrowTotalMassFlow_kg_s = pyo.Expression(
    expr=m.rt.sludgeMassFlowOut
)
m.egrowSolidsMassFlow_kg_s = pyo.Expression(
    expr=m.rt.sludgeMassFlowOut * m.rt.sludgeTSSout
)
m.egrowLiquidMass_kg_s = pyo.Expression(
    expr=m.rt.sludgeMassFlowOut * (1.0 - m.rt.sludgeTSSout)
)

# Original feed solids (kg/s) 
m.originalSludgeSolidsKgPerS = pyo.Expression(
    expr=m.feedDrySolids_kg_s
)

# Solids-N tracking based on feed solids.
# Original feed solids-N (kg/s).
m.solidsNInFeed_kg_s = pyo.Expression(
    expr=m.productSolidsNMassFrac * m.originalSludgeSolidsKgPerS
)

# el.nitrogenBalance 
m.solidsNToLiquid_kg_s = pyo.Expression(
    expr=m.solidsNToLiquidFrac_total * m.solidsNInFeed_kg_s
)
# AMMONIACAL-ONLY mobilized N -- used for nf/GPM TAN feed.
m.solidsNToLiquid_TAN_kg_s = pyo.Expression(
    expr=m.solidsNToLiquidFrac_TAN * m.solidsNInFeed_kg_s
)
# Remaining solids-N in product (based on TOTAL mobilization, i.e. any N form
# that left the solid phase).
m.solidsNRemaining_kg_s = pyo.Expression(
    expr=m.solidsNInFeed_kg_s - m.solidsNToLiquid_kg_s
)

# Inlet dissolved N mass flow (kg/s) from feed liquid.
m.inletLiquidNMassFlow_kg_s = pyo.Expression(
    expr=m.productLiquidNConc_g_m3 * (m.feedFlow_m3s * (1.0 - m.feedTSS)) / 1000.0
)
# Product liquid N concentration at RT outlet (kg-N per kg-liquid).
m.liquidNConc_kgNPerKgLiquid = pyo.Expression(
    expr=(m.inletLiquidNMassFlow_kg_s + m.solidsNToLiquid_kg_s)
         / (m.el.sludgeMassFlowOut * (1.0 - m.el.sludgeTSSout) + 1e-9)
)
# Product liquid mass in final E-GROW stream (kg/s).
m.productLiquidMass_kg_s = pyo.Expression(
    expr=m.go.sludgeMassFlowOut * (1.0 - m.go.sludgeTSSout)
)
# Product liquid N = concentration × retained product liquid.
m.productLiquidNMassFlow_kg_s = pyo.Expression(
    expr=m.liquidNConc_kgNPerKgLiquid * m.productLiquidMass_kg_s
)
m.productSolidsNMassFlow_kg_s = pyo.Expression(
    expr=m.solidsNRemaining_kg_s
)
m.productNMassFlow_kg_s = pyo.Expression(
    expr=m.productSolidsNMassFlow_kg_s + m.productLiquidNMassFlow_kg_s
)
m.productN_wtPercent_wet = pyo.Expression(
    expr=100.0 * m.productNMassFlow_kg_s / (m.egrowTotalMassFlow_kg_s + 1e-9)
)

# --- Combined product: acid tank outlet + GPM ammonium sulfate solution ---

# Total combined product mass flow (kg/s)
m.combinedProductFlow_kgPerS = pyo.Expression(
    expr=m.rt.sludgeMassFlowOut + m.gpm.productMassFlow
)

# Combined product solids (only from retentate — GPM product is liquid, no solids)
m.combinedProductSolids_kgPerS = pyo.Expression(
    expr=m.rt.sludgeMassFlowOut * m.rt.sludgeTSSout
)

# Combined product liquid mass (kg/s)
m.combinedProductLiquid_kgPerS = pyo.Expression(
    expr=m.combinedProductFlow_kgPerS - m.combinedProductSolids_kgPerS
)

# Combined product TSS (mass fraction)
m.combinedProductTSS = pyo.Expression(
    expr=m.combinedProductSolids_kgPerS / (m.combinedProductFlow_kgPerS + 1e-9)
)

# N recovered by GPM and added back to combined product
m.gpmRecoveredN_kgPerS = pyo.Expression(
    expr=m.gpm.N_removed
)

# Combined product N = solids N + retentate liquid N + GPM recovered N
m.combinedProductN_kgPerS = pyo.Expression(
    expr=m.productSolidsNMassFlow_kg_s
       + m.productLiquidNMassFlow_kg_s
       + m.gpmRecoveredN_kgPerS
)

m.combinedProductN_wtPercent = pyo.Expression(
    expr=100.0 * m.combinedProductN_kgPerS / (m.combinedProductFlow_kgPerS + 1e-9)
)

# ------------------------------------------------------------------
# Phosphorus and potassium accounting 
# ------------------------------------------------------------------
# Solids-phase P/K
m.solidsPFraction = pyo.Param(initialize=0.025, mutable=True)    # kg-P/kg-dry-solids
m.solidsKFraction = pyo.Param(initialize=0.0035, mutable=True)   # kg-K/kg-dry-solids
 
# Centrate (liquid-phase) P/K
m.centratePConc_mgL = pyo.Param(initialize=13.0, mutable=True)   # mg-P/L
m.centrateKConc_mgL = pyo.Param(initialize=275.0, mutable=True)  # mg-K/L
 
m.feedSolidsP_kg_s = pyo.Expression(expr=m.solidsPFraction * m.feedDrySolids_kg_s)
m.feedSolidsK_kg_s = pyo.Expression(expr=m.solidsKFraction * m.feedDrySolids_kg_s)
 
m.feedLiquid_kg_s = pyo.Expression(expr=m.feedMassFlow_kg_s - m.feedDrySolids_kg_s)
m.feedLiquidP_kg_s = pyo.Expression(expr=m.centratePConc_mgL * (m.feedLiquid_kg_s / 1000.0) / 1000.0)
m.feedLiquidK_kg_s = pyo.Expression(expr=m.centrateKConc_mgL * (m.feedLiquid_kg_s / 1000.0) / 1000.0)
 
m.feedTotalP_kg_s = pyo.Expression(expr=m.feedSolidsP_kg_s + m.feedLiquidP_kg_s)
m.feedTotalK_kg_s = pyo.Expression(expr=m.feedSolidsK_kg_s + m.feedLiquidK_kg_s)
 
m.productP_kg_s = pyo.Expression(expr=m.feedTotalP_kg_s)
 
m.elLiquidMass_kg_s = pyo.Expression(expr=m.el.sludgeMassFlowOut * (1.0 - m.el.sludgeTSSout))
m.goRetentateLiquidMass_kg_s = pyo.Expression(expr=m.go.sludgeMassFlowOut * (1.0 - m.go.sludgeTSSout))
m.goLiquidRetentionFrac = pyo.Expression(
    expr=m.goRetentateLiquidMass_kg_s / (m.elLiquidMass_kg_s + 1e-9)
)
m.productK_kg_s = pyo.Expression(
    expr=m.feedSolidsK_kg_s + m.feedLiquidK_kg_s * m.goLiquidRetentionFrac
)
 
# Reported on the SAME basis as N here.
m.productP_wtPercent_wet = pyo.Expression(expr=100.0 * m.productP_kg_s / (m.combinedProductFlow_kgPerS + 1e-9))
m.productK_wtPercent_wet = pyo.Expression(expr=100.0 * m.productK_kg_s / (m.combinedProductFlow_kgPerS + 1e-9))


# Target combined product TSS
m.targetProductTss = pyo.Param(initialize=0.25, mutable=True)
m.combinedProductTSSTarget = pyo.Constraint(
    expr=m.combinedProductTSS == m.targetProductTss
)

m.pt.waterMassFlowIn.fix(0.0)


m.elLiquidVolFlow_m3_s = pyo.Expression(expr=m.elLiquidMass_kg_s / m.liquidDensity)
m.correctedDissolvedNConc_kgPerM3 = pyo.Expression(
    expr=(m.inletLiquidNMassFlow_kg_s + m.solidsNToLiquid_TAN_kg_s)
         / (m.elLiquidVolFlow_m3_s + 1e-9)
)

# Feed TAN concentration into nf (kg-N/m3) -- go permeate is solids-free liquid
# at the same dissolved N concentration as the bulk liquid at electrolyzer outlet.
m.go_to_nf_tan = pyo.Constraint(
    expr=m.nf.concInTAN == m.correctedDissolvedNConc_kgPerM3
)

# --- Divalent (Ca/Mg) load into the nf ---
m.dissolvedCaPerKgCaO = pyo.Param(initialize=0.0468, mutable=True)
m.dissolvedMgPerKgDS  = pyo.Param(initialize=0.00350, mutable=True)

m.dissolvedCaMassFlow_kg_s = pyo.Expression(expr=m.dissolvedCaPerKgCaO * m.pt.totalCaO_kgPerS)
m.dissolvedMgMassFlow_kg_s = pyo.Expression(expr=m.dissolvedMgPerKgDS * m.pt.sludgeDrySolidsMassFlow_kgPerS)
m.dissolvedDivalentMassFlow_kg_s = pyo.Expression(expr=m.dissolvedCaMassFlow_kg_s + m.dissolvedMgMassFlow_kg_s)

m.go_to_nf_ca = pyo.Constraint(
    expr=m.nf.concInCa == m.dissolvedCaMassFlow_kg_s / (m.elLiquidVolFlow_m3_s + 1e-9)
)
m.go_to_nf_mg = pyo.Constraint(
    expr=m.nf.concInMg == m.dissolvedMgMassFlow_kg_s / (m.elLiquidVolFlow_m3_s + 1e-9)
)

# nf permeate TAN (post-rejection) feeds the GPM.
m.nf_to_gpm_n = pyo.Constraint(
    expr=m.gpm.concInFresh == m.nf.concPermTAN
)


# --- GPM pH mixing credit ---
# Total H+ equivalents added to GPM as H2SO4 (mol H+/s)
m.gpmTotalAcidEqAdded = pyo.Expression(
    expr=m.gpm.acidFlowIn * m.gpm.acidDensity * m.gpm.w_acid_in / 0.098 * 2.0
)

# H+ equivalents consumed by NH3 capture (1 mol H+ per mol NH3)
m.gpmAcidConsumedByNH3 = pyo.Expression(
    expr=m.gpm.N_removed / 0.014
)

# Excess H+ equivalents remaining in GPM product after NH3 reaction (mol H+/s)
m.gpmExcessAcidEq = pyo.Expression(
    expr=m.gpmTotalAcidEqAdded - m.gpmAcidConsumedByNH3
)

# Adjusted RT acid demand after subtracting GPM credit
m.rtAdjustedAcidEq = pyo.Expression(
    expr=m.rt.totalAcidEqPerS - m.gpmExcessAcidEq
)

# Replace the RT acid flow constraint with the credit-adjusted version.
m.rt.acidFlowConstr.deactivate()
m.rtAdjustedAcidFlow = pyo.Constraint(
    expr=m.rt.acidMassFlowIn == m.rtAdjustedAcidEq
         * m.rt.molwtH2SO4kgPerMol / (2.0 * m.rt.acidSolutionWtFraction)
)

# Cost objective in $
m.totalCapex = pyo.Expression(expr=1.35*(m.pt.capex + m.el.capex + m.go.capex + m.nf.capex + m.rt.capex + m.gpm.capex + m.st.capex))
m.totalOpex = pyo.Expression(expr=m.pt.opex + m.el.opex + m.go.opex + m.nf.opex + m.rt.opex + m.gpm.opex + m.st.opex)
m.totalCost = pyo.Objective(expr= (m.totalCapex + m.totalOpex)/10**6, sense=pyo.minimize)

tol = 1e-3
solver = pyo.SolverFactory('ipopt', options={'tol': tol, 'max_iter': 5000})
solver = pyo.SolverFactory('baron',options = {'EpsA': 1*tol, 'AbsConFeasTol': 1*tol, 'TDo':0, 'MDo':0,'OBTTDo':1}, executable ='C:/baron/baron.exe')
sol = solver.solve(m, tee=True)


# quick printout
print('Feed flow (m3/day):', pyo.value(m.feedFlow_m3s) * 86400.0)
print('Feed solids (wt frac):', pyo.value(m.feedTSS))
print('Feed dissolved N (g-N/m3):', pyo.value(m.feedNitrogenConc))
print('Combined product mass flow (kg/day):', pyo.value(m.combinedProductFlow_kgPerS) * 86400)
print('Combined product TSS (fraction):', pyo.value(m.combinedProductTSS))
print('Combined product N (wt% wet):', pyo.value(m.combinedProductN_wtPercent))
print('Combined product N (kg/day):', pyo.value(m.combinedProductN_kgPerS) * 86400)
print('Total Cost ($):', pyo.value(m.totalCost*10**6))

egrow_wet_tonnes = m.combinedProductFlow_kgPerS * m.daysOperation * 1e-3
n_product_tonnes = m.combinedProductN_kgPerS * m.daysOperation * 1e-3

print('Combined product (wet tonnes total):', pyo.value(egrow_wet_tonnes))
print('Combined product N (t-N total):', pyo.value(n_product_tonnes))
print('Combined product N from solids (kg/s):', pyo.value(m.productSolidsNMassFlow_kg_s))
print('Combined product N from liquid (kg/s):', pyo.value(m.productLiquidNMassFlow_kg_s))
print('Combined product N from GPM (kg/s):', pyo.value(m.gpmRecoveredN_kgPerS))
print('Combined product total N (kg/s):', pyo.value(m.combinedProductN_kgPerS))
print('Combined product N (wt% wet):', pyo.value(m.combinedProductN_wtPercent))

total_capex = 1.35*(m.pt.capex + m.el.capex + m.go.capex + m.nf.capex + m.rt.capex + m.gpm.capex + m.st.capex)
total_opex = m.pt.opex + 2*m.el.opex + m.go.opex + m.nf.opex + m.rt.opex + m.gpm.opex + m.st.opex
total_cost = total_capex + total_opex
print('Total cost ($ total):', pyo.value(total_cost))
print('Total cost ($/wet tonne combined product):', pyo.value(total_cost / (egrow_wet_tonnes + 1e-9)))
print("-------------------------------")
print("Prep Tank Sludge In Mass Flow (kg/day):", pyo.value(m.pt.sludgeMassFlowIn)*86400)
print("Prep Tank Water In Mass Flow (kg/day):", pyo.value(m.pt.waterMassFlowIn)*86400)
print("Prep Tank CaO Mass Flow (kg/day):", pyo.value(m.pt.totalCaO_kgPerS)*86400)
print("Prep Tank Sludge Out Mass Flow (kg/day):", pyo.value(m.pt.sludgeMassFlowOut)*86400)
print("Prep Tank Sludge TSS In (fraction):", pyo.value(m.pt.sludgeTSSin))
print("Prep Tank Sludge TSS Out (fraction):", pyo.value(m.pt.sludgeTSSout))
print("Prep Tank Volume (m3):", pyo.value(m.pt.tankVolume))
print("Prep Tank Lime Tank Volume (m3):", pyo.value(m.pt.limeTankVolume))
print("-------------------------------")
print("Electrolyzer Sludge In Mass Flow (kg/day):", pyo.value(m.el.sludgeMassFlowIn)*86400)
print("Electrolyzer Sludge Out Mass Flow (kg/day):", pyo.value(m.el.sludgeMassFlowOut)*86400)
print("Electrolyzer feed solids content (wt%):", pyo.value(m.el.sludgeTSSin)*100)
print("Electrolyzer Nitrogen Out Flow (kg-N/day):", pyo.value(m.el.nitrogenFlowOut)*86400)
print("Electrolyzer Sludge TSS Out (fraction):", pyo.value(m.el.sludgeTSSout))
print("Electrolyzer pH in/out:", pyo.value(m.el.sludgepHIn), pyo.value(m.el.sludgepHOut))
print("Electrolyzer Area (m2):", pyo.value(m.el.area))
print("--------------------------------")
print("GO Membrane Dewatering Sludge In Mass Flow (kg/day):", pyo.value(m.go.sludgeMassFlowIn)*86400)
print("GO Membrane Dewatering Sludge Out Mass Flow (kg/day):", pyo.value(m.go.sludgeMassFlowOut)*86400)
print("GO Membrane Dewatering Permeate Mass Flow (kg/day):", pyo.value(m.go.permeateMassFlow)*86400)
print("GO Membrane Dewatering Sludge TSS In (fraction):", pyo.value(m.go.sludgeTSSin))
print("GO Membrane Dewatering pH in/out:", pyo.value(m.go.sludgepHIn), pyo.value(m.go.liquidpHOut))
print("GO Membrane Dewatering Sludge TSS Out (fraction):", pyo.value(m.go.sludgeTSSout))
print("GO Membrane Dewatering Delta-pi (bar, was dead/always-zero before rework):", pyo.value(m.go.deltaPi))
print("GO Membrane Dewatering Pressure Drop (bar):", pyo.value(m.go.deltaP))
print("GO Membrane Dewatering Area (m2):", pyo.value(m.go.area))
print("--------------------------------")
print("NanoFiltration (bivalent removal) Mass Flow In (kg/day):", pyo.value(m.nf.massFlowIn)*86400)
print("NanoFiltration Permeate Mass Flow (kg/day):", pyo.value(m.nf.permeateMassFlow)*86400)
print("NanoFiltration Retentate Mass Flow (kg/day):", pyo.value(m.nf.retentateMassFlow)*86400)
print("NanoFiltration Water Recovery (-):", pyo.value(m.nf.recovery))
print("NanoFiltration TAN In (kg-N/m3):", pyo.value(m.nf.concInTAN))
print("NanoFiltration TAN Permeate (kg-N/m3):", pyo.value(m.nf.concPermTAN))
print("NanoFiltration Ca In (kg/m3):", pyo.value(m.nf.concInCa))
print("NanoFiltration Ca Permeate (kg/m3):", pyo.value(m.nf.concPermCa))
print("NanoFiltration Mg In (kg/m3):", pyo.value(m.nf.concInMg))
print("NanoFiltration Mg Permeate (kg/m3):", pyo.value(m.nf.concPermMg))
print("NanoFiltration pH In:", pyo.value(m.nf.pHIn))
print("NanoFiltration pH Out:", pyo.value(m.nf.pHOut))
print("NanoFiltration Pressure Drop (bar):", pyo.value(m.nf.deltaP))
print("NanoFiltration Area (m2):", pyo.value(m.nf.area))
print("NanoFiltration CAPEX ($):", pyo.value(1.35*m.nf.capex))
print("NanoFiltration OPEX ($):", pyo.value(m.nf.opex))
print("--------------------------------")
print("GPM Fresh Feed Conc (ppm): ", pyo.value(m.gpm.concInFresh)*1000)
print("GPM Conc In (ppm): ", pyo.value(m.gpm.concIn)*1000)
print("GPM Conc Out (ppm): ", pyo.value(m.gpm.concOut)*1000)
print("GPM Mass Flow In (kg/day): ", pyo.value(m.gpm.massFlowIn * 86400))
print("GPM Mass Flow Out (kg/day): ", pyo.value(m.gpm.massFlowOut * 86400))
print("GPM area (m2): ", pyo.value(m.gpm.area))
print("GPM Feed pH out: ", pyo.value(m.gpm.pH_out))
print("Alpha (Average): ", pyo.value(m.gpm.alpha_avg))
print("Salt Mass Flow (kg/day): ", pyo.value(m.gpm.saltMassFlow * 86400))
print("Water vapor crossed to draw side (kg/day): ", pyo.value(m.gpm.jWaterVaporMassFlow * 86400))
print("Extra dilution water for AS solubility ceiling (kg/day): ", pyo.value(m.gpm.extraDilutionWaterNeeded * 86400))
print("Product Mass Flow (kg/day): ", pyo.value(m.gpm.productMassFlow * 86400))
print("Product N wt%: ", pyo.value(m.gpm.N_wt_percent))
print("GPM Acid Flow In (m3/day): ", pyo.value(m.gpm.acidFlowIn * 86400))
print("GPM Acid Concentration (wt%): ", pyo.value(m.gpm.w_acid_in * 100))
print("GPM Capex ($): ", pyo.value(m.gpm.capex))
print("GPM Opex ($): ", pyo.value(m.gpm.opex))
print("GPM N removal (%): ", pyo.value(100.0 * m.gpm.N_removed / (m.gpm.volFlowIn * m.gpm.concInFresh + 1e-9)))
n_rem_residual = abs(
    pyo.value(m.gpm.N_removed)
    - pyo.value(m.gpm.volFlowIn * (m.gpm.concInFresh - m.gpm.concOut))
)
print('N_rem_constr residual:', n_rem_residual)
print("-------------------------------")
print("Receive Tank Sludge In Mass Flow (kg/day):", pyo.value(m.rt.sludgeMassFlowIn)*86400)
print("Receive Tank Sludge Out Mass Flow (kg/day):", pyo.value(m.rt.sludgeMassFlowOut)*86400)
print("Receive Tank Acid In Mass Flow (kg/day):", pyo.value(m.rt.acidMassFlowIn)*86400)
print("Receive Tank Sludge TSS Out (fraction):", pyo.value(m.rt.sludgeTSSout))
print("Receive Tank Volume (m3):", pyo.value(m.rt.tankVolume))


print("-------------------------------")
print("Combined product mass flow (kg/day):", pyo.value(m.combinedProductFlow_kgPerS)*86400)
print("Combined product TSS (fraction):", pyo.value(m.combinedProductTSS))
print("Combined product N (wt% wet):", pyo.value(m.combinedProductN_wtPercent))
print("Combined product N (kg/day):", pyo.value(m.combinedProductN_kgPerS)*86400)

print('----------')
feed_liquid_n_tonnes = m.feedFlow_m3s * m.daysOperation * m.feedNitrogenConc * 1e-6
feed_solids_n_tonnes_model = m.solidsNInFeed_kg_s * m.daysOperation * 1e-3
feed_total_n_tonnes_model = feed_liquid_n_tonnes + feed_solids_n_tonnes_model

combined_liquid_n_tonnes_model = m.productLiquidNMassFlow_kg_s * m.daysOperation * 1e-3
combined_solids_n_tonnes_model = m.solidsNRemaining_kg_s * m.daysOperation * 1e-3
gpm_recovered_n_tonnes_model = m.gpmRecoveredN_kgPerS * m.daysOperation * 1e-3
gpm_effluent_n_tonnes_model = m.gpm.volFlowOut * m.gpm.concOut * m.daysOperation * 1e-3

combined_n_out_tonnes_model = (
    combined_liquid_n_tonnes_model + combined_solids_n_tonnes_model + gpm_recovered_n_tonnes_model
)
all_outlets_n_tonnes_model = combined_n_out_tonnes_model + gpm_effluent_n_tonnes_model

n_recovery_products_pct_model = 100.0 * combined_n_out_tonnes_model / (feed_total_n_tonnes_model + 1e-9)
n_recovery_all_outlets_pct_model = 100.0 * all_outlets_n_tonnes_model / (feed_total_n_tonnes_model + 1e-9)

total_capex_per_wet_tonne = total_capex / (egrow_wet_tonnes + 1e-9)
total_opex_per_wet_tonne = total_opex / (egrow_wet_tonnes + 1e-9)
total_cost_per_wet_tonne = total_cost / (egrow_wet_tonnes + 1e-9)
total_capex_per_tn = total_capex / (n_product_tonnes + 1e-9)
total_opex_per_tn = total_opex / (n_product_tonnes + 1e-9)
total_cost_per_tn = total_cost / (n_product_tonnes + 1e-9)

print('Combined product N (wt% wet):', pyo.value(m.combinedProductN_wtPercent))
print('GPM N captured (wt% in GPM product):', pyo.value(m.gpm.N_wt_percent))
print('Feed dissolved N (t-N):', pyo.value(feed_liquid_n_tonnes))
print('Feed solids-bound N (t-N, model basis):', pyo.value(feed_solids_n_tonnes_model))
print('Feed total N (t-N, model basis):', pyo.value(feed_total_n_tonnes_model))
print('Combined product N from liquid (t-N, model basis):', pyo.value(combined_liquid_n_tonnes_model))
print('Combined product N from solids (t-N, model basis):', pyo.value(combined_solids_n_tonnes_model))
print('Combined product N from GPM (t-N, model basis):', pyo.value(gpm_recovered_n_tonnes_model))
print('Combined product N total (t-N, model basis):', pyo.value(combined_n_out_tonnes_model))
print('GPM liquid effluent N outlet (t-N, model basis):', pyo.value(gpm_effluent_n_tonnes_model))
print('N out in all outlets (t-N, model basis):', pyo.value(all_outlets_n_tonnes_model))
print('N recovery to combined product (%):', pyo.value(n_recovery_products_pct_model))
print('N recovery to all outlets (%):', pyo.value(n_recovery_all_outlets_pct_model))

# Individual unit operation capex and opex (all on combined product basis)
print('Prep Tank CAPEX ($/wet tonne):', pyo.value(1.35*m.pt.capex/(egrow_wet_tonnes + 1e-9)))
print('Electrolyzer CAPEX ($/wet tonne):', pyo.value(1.35*m.el.capex/(egrow_wet_tonnes + 1e-9)))
print('Receive Tank CAPEX ($/wet tonne):', pyo.value(1.35*m.rt.capex/(egrow_wet_tonnes + 1e-9)))
print('GO Membrane Dewatering CAPEX ($/wet tonne):', pyo.value(1.35*m.go.capex/(egrow_wet_tonnes + 1e-9)))
print('Nanofiltration (bivalent removal) CAPEX ($/wet tonne):', pyo.value(1.35*m.nf.capex/(egrow_wet_tonnes + 1e-9)))
print('GPM CAPEX ($/wet tonne):', pyo.value(1.35*m.gpm.capex/(egrow_wet_tonnes + 1e-9)))
print('Storage Tank CAPEX ($/wet tonne):', pyo.value(1.35*m.st.capex/(egrow_wet_tonnes + 1e-9)))
print('Prep Tank OPEX ($/wet tonne):', pyo.value(m.pt.opex/(egrow_wet_tonnes + 1e-9)))
print('Electrolyzer OPEX ($/wet tonne):', pyo.value(2*m.el.opex/(egrow_wet_tonnes + 1e-9)))
print('Receive Tank OPEX ($/wet tonne):', pyo.value(m.rt.opex/(egrow_wet_tonnes + 1e-9)))
print('GO Membrane Dewatering OPEX ($/wet tonne):', pyo.value(m.go.opex/(egrow_wet_tonnes + 1e-9)))
print('Nanofiltration (bivalent removal) OPEX ($/wet tonne):', pyo.value(m.nf.opex/(egrow_wet_tonnes + 1e-9)))
print('GPM OPEX ($/wet tonne):', pyo.value(m.gpm.opex/(egrow_wet_tonnes + 1e-9)))
print('Storage Tank OPEX ($/wet tonne):', pyo.value(m.st.opex/(egrow_wet_tonnes + 1e-9)))
print('---------------------------------')

print('Total CAPEX ($ total):', pyo.value(total_capex))
print('Total OPEX ($ total):', pyo.value(total_opex))
print('Total cost ($ total):', pyo.value(total_cost))
print('Total CAPEX ($/wet tonne):', pyo.value(total_capex_per_wet_tonne))
print('Total OPEX ($/wet tonne):', pyo.value(total_opex_per_wet_tonne))
print('Total cost ($/wet tonne):', pyo.value(total_cost_per_wet_tonne))
print('Total CAPEX ($/t-N):', pyo.value(total_capex_per_tn))
print('Total OPEX ($/t-N):', pyo.value(total_opex_per_tn))
print('Total cost ($/t-N):', pyo.value(total_cost_per_tn))

print('================ PHOSPHORUS / POTASSIUM (rudimentary, top-level) ========')
print('  Feed P total (kg/day):', pyo.value(m.feedTotalP_kg_s * 86400.0))
print('  Feed K total (kg/day):', pyo.value(m.feedTotalK_kg_s * 86400.0))
print('  Product P (kg/day):', pyo.value(m.productP_kg_s * 86400.0))
print('  Product K (kg/day):', pyo.value(m.productK_kg_s * 86400.0))
print('  Product P (wt% wet):', pyo.value(m.productP_wtPercent_wet))
print('  Product K (wt% wet):', pyo.value(m.productK_wtPercent_wet))
print('  Product P2O5 (wt% wet, = P wt% x 2.29):', pyo.value(m.productP_wtPercent_wet * 2.29))
print('  Product K2O (wt% wet, = K wt% x 1.20):', pyo.value(m.productK_wtPercent_wet * 1.20))
print('  GO Membrane Dewatering liquid retention fraction (K tracer basis):', pyo.value(m.goLiquidRetentionFrac))
print('===================================================')



print('================ INPUTS / ELECTRICITY / CHEMICALS (FS2) =================')

def _safe_value_end(expr):
    try:
        return pyo.value(expr)
    except Exception:
        return 'n/a'

# Mass inputs (kg/day)
fs2_feed_sludge_kg_day = m.feedMassFlow_kg_s * 86400.0
fs2_water_added_kg_day = m.pt.waterMassFlowIn * 86400.0
fs2_cao_slurry_kg_day = m.pt.totalCaO_kgPerS * 86400.0
fs2_rt_acid_solution_kg_day = m.rt.acidMassFlowIn * 86400.0
fs2_rt_h2so4_kg_day = m.rt.h2so4RequiredKgPerS * 86400.0
fs2_rt_acid_equiv_mol_day = m.rt.totalAcidEqPerS * 86400.0
fs2_gpm_acid_solution_kg_day = m.gpm.acidFlowIn * m.gpm.acidDensity * 86400.0
fs2_gpm_acid_h2so4_equiv_mol_day = m.gpm.acidFlowIn * m.gpm.acidDensity * m.gpm.w_acid_in * 1000.0 / 98.079 * 86400.0
fs2_gpm_h2so4_kg_day = fs2_gpm_acid_solution_kg_day * m.gpm.w_acid_in
fs2_total_acid_solution_kg_day = fs2_rt_acid_solution_kg_day + fs2_gpm_acid_solution_kg_day
fs2_combined_product_kg_day = m.combinedProductFlow_kgPerS * 86400.0
fs2_nitrogen_in_product_kg_day = m.combinedProductN_kgPerS * 86400.0

print('Mass inputs (kg/day):')
print('  Feed sludge:', _safe_value_end(fs2_feed_sludge_kg_day))
print('  Added water (prep tank):', _safe_value_end(fs2_water_added_kg_day))
print('  CaO/base stream (prep tank):', _safe_value_end(fs2_cao_slurry_kg_day))
print('  Receive-tank acid stream:', _safe_value_end(fs2_rt_acid_solution_kg_day))
print('  Receive-tank H2SO4 equivalent:', _safe_value_end(fs2_rt_h2so4_kg_day))
print('  Receive-tank acid equivalents (mol/day):', _safe_value_end(fs2_rt_acid_equiv_mol_day))
print('  GPM acid solution stream:', _safe_value_end(fs2_gpm_acid_solution_kg_day))
print('  GPM acid H2SO4 equivalent (kg/day):', _safe_value_end(fs2_gpm_h2so4_kg_day))
print('  Total acid solution requirement:', _safe_value_end(fs2_total_acid_solution_kg_day))
print('  Combined product produced (kg/day):', _safe_value_end(fs2_combined_product_kg_day))
print('  Nitrogen in combined product (kg/day):', _safe_value_end(fs2_nitrogen_in_product_kg_day))

# Electricity requirement (unit model basis)
fs2_el_power_kW = m.el.power
fs2_nf_pump_kW = m.nf.pumpPower
fs2_gpm_pump_kW = m.gpm.pumpPower

fs2_el_energy_kWh_day = fs2_el_power_kW * 24.0
fs2_nf_energy_kWh_day = fs2_nf_pump_kW * 24.0
fs2_gpm_energy_kWh_day = fs2_gpm_pump_kW * 24.0

print('Electricity requirement by unit (power kW, energy kWh/day):')
print('  Electrolyzer:', 'kW =', _safe_value_end(fs2_el_power_kW), ', kWh/day =', _safe_value_end(fs2_el_energy_kWh_day))
print('  NanoFiltration pump:', 'kW =', _safe_value_end(fs2_nf_pump_kW), ', kWh/day =', _safe_value_end(fs2_nf_energy_kWh_day))
print('  GPM recirculation pump:', 'kW =', _safe_value_end(fs2_gpm_pump_kW), ', kWh/day =', _safe_value_end(fs2_gpm_energy_kWh_day))

# Chemical usage (kg/day)
print('Chemical usage (kg/day):')
print('  CaO/base stream:', _safe_value_end(fs2_cao_slurry_kg_day))
print('  Receive-tank acid stream:', _safe_value_end(fs2_rt_acid_solution_kg_day))
print('  GPM acid solution stream:', _safe_value_end(fs2_gpm_acid_solution_kg_day))
print('  Total acid solution requirement:', _safe_value_end(fs2_total_acid_solution_kg_day))
print('  GPM pure H2SO4 equivalent:', _safe_value_end(fs2_gpm_h2so4_kg_day))
print('========================================================================')
