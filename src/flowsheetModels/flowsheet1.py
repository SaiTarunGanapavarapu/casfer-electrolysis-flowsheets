#------------------------------------------------------------------------------
# function:    flowsheet1.py                                                  #
# Description: Optimization model for wastewater treatment process            #
#              Process flow: Feed -> Prep Tank -> Electrolyzer ->             #
#              Receive Tank -> [optional Centrifuge] -> Dryer -> Storage      #
#                                                                             #
#                                                                             #
# Input:       - m : Pyomo concrete model                                     #
#                                                                             #
# Output:      - m with all parameters, variables, constraints, and objective #
#                                                                             #
#------------------------------------------------------------------------------

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pyomo.environ as pyo
import src.singleUnitModels.getParams as getParams
import src.singleUnitModels.getStandardParams as getStandardParams
import src.singleUnitModels.prepTank as prepTank
import src.singleUnitModels.electrolyzer as electrolyzer
import src.singleUnitModels.receiveTank as receiveTank
import src.singleUnitModels.centrifuge as centrifuge
import src.singleUnitModels.dryer as dryer
import src.singleUnitModels.storageTank as storageTank


# Create concrete model
m = pyo.ConcreteModel()

# Load parameters and standard constants
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

# Combined feed mass flow and dry-solids mass flow 
m.feedMassFlow_kg_s = pyo.Expression(
    expr=m.dewateredSludgeFlow_m3s * m.dewateredSludgeDensity
       + m.centrateFlow_m3s * m.centrateDensity
)
m.feedDrySolids_kg_s = pyo.Expression(
    expr=m.dewateredSludgeFlow_m3s * m.dewateredSludgeDensity * m.dewateredSludgeTSS
)
m.feedLiquid_kg_s = pyo.Expression(
    expr=m.feedMassFlow_kg_s - m.feedDrySolids_kg_s
)
m.feedFlow_m3s = pyo.Expression(
    expr=m.dewateredSludgeFlow_m3s + m.centrateFlow_m3s
)
m.feedTSS = pyo.Expression(
    expr=m.feedDrySolids_kg_s / m.feedMassFlow_kg_s
)

m.inletNitrogenConc = pyo.Param(initialize=750.0, mutable=True)        # g-N/m3 (750 ppm, from centrate)
m.targetProductTSS = pyo.Param(initialize=0.25, mutable=True)          # 25 wt fraction

# Add a binary variable to control the centrifuge and define Big-M (mass basis)
m.y_cf = pyo.Var(within=pyo.Binary, initialize=1)
M_mass = float(pyo.value(m.totalFlowIn) * 1200.0 * 100.0)  # kg/s

# --- Flow splitting variables ---
m.flow_to_cf_mass = pyo.Var(within=pyo.NonNegativeReals, initialize=M_mass)      # kg/s
m.flow_bypass_cf_mass = pyo.Var(within=pyo.NonNegativeReals, initialize=0)       # kg/s

# Initialize unit Blocks
prepTank.prepTank(m)
electrolyzer.electrolyzer(m)
receiveTank.receiveTank(m)
centrifuge.centrifuge(m)
dryer.dryer(m)
storageTank.storageTank(m)

m.pt.targetTSSConstr.deactivate()
m.pt.waterMassFlowIn.fix(0.0)  # water addition is fixed at zero

# Tighten centrifuge geometry indicator big-M to reduce slack when y_cf≈1
try:
    m.cf.M_geo.set_value(10.0)
except Exception:
    pass

# --- connections that are always active  ---
def feedToPrepTankMassFlow_rule(mod):
    return mod.pt.sludgeMassFlowIn == mod.feedMassFlow_kg_s
m.feedToPrepTankMassFlow = pyo.Constraint(rule=feedToPrepTankMassFlow_rule)

def feedToPrepTankTSS_rule(mod):
    return mod.pt.sludgeTSSin == mod.feedTSS
m.feedToPrepTankTSS = pyo.Constraint(rule=feedToPrepTankTSS_rule)

# Link sub-stream flows to prep tank for split CaO calculation 
m.linkSludgeToPt   = pyo.Constraint(expr=m.pt.sludgeOnlyFlow_m3s == m.dewateredSludgeFlow_m3s)
m.linkCentrateToPt = pyo.Constraint(expr=m.pt.centrateFlow_m3s   == m.centrateFlow_m3s)

def prepTankToElectrolyzerMassFlow_rule(mod):
    return mod.el.sludgeMassFlowIn == mod.pt.sludgeMassFlowOut
m.prepTankToElectrolyzerMassFlow = pyo.Constraint(rule=prepTankToElectrolyzerMassFlow_rule)

def prepTankToElectrolyzerTSS_rule(mod):
    return mod.el.sludgeTSSin == mod.pt.sludgeTSSout
m.prepTankToElectrolyzerTSS = pyo.Constraint(rule=prepTankToElectrolyzerTSS_rule)

# Composition link 
m.prepTankToElectrolyzerSludgeSolids = pyo.Constraint(
    expr=m.el.sludgeSolidsMassFlowIn == m.pt.sludgeSolidsMassFlowOut
)

def electrolyzerToReceiveTankMassFlow_rule(mod):
    return mod.rt.sludgeMassFlowIn == mod.el.sludgeMassFlowOut
m.electrolyzerToReceiveTankMassFlow = pyo.Constraint(rule=electrolyzerToReceiveTankMassFlow_rule)

def electrolyzerToReceiveTankTSS_rule(mod):
    return mod.rt.sludgeTSSin == mod.el.sludgeTSSout
m.electrolyzerToReceiveTankTSS = pyo.Constraint(rule=electrolyzerToReceiveTankTSS_rule)

# Composition link
m.electrolyzerToReceiveTankSludgeSolids = pyo.Constraint(
    expr=m.rt.sludgeSolidsMassFlowIn == m.pt.sludgeSolidsMassFlowOut
)

# --- Liquid nitrogen concentration connections ---
def feedNitrogenConc_rule(mod):
    return mod.pt.nitrogenConcIn == mod.inletNitrogenConc
m.feedNitrogenConc = pyo.Constraint(rule=feedNitrogenConc_rule)

def prepTankToElectrolyzerNConc_rule(mod):
    return mod.el.nitrogenConcIn == mod.pt.nitrogenConcOut
m.prepTankToElectrolyzerNConc = pyo.Constraint(rule=prepTankToElectrolyzerNConc_rule)

def electrolyzerToReceiveTankNConc_rule(mod):
    return mod.rt.nitrogenConcIn == mod.el.nitrogenConcOut
m.electrolyzerToReceiveTankNConc = pyo.Constraint(rule=electrolyzerToReceiveTankNConc_rule)


# --- Connections for optional centrifuge ---

# 1. Total mass flow out of the receive tank is the sum of the two split streams.
def receiveTankTotalMassFlowOut_rule(mod):
    return mod.rt.sludgeMassFlowOut == mod.flow_to_cf_mass + mod.flow_bypass_cf_mass
m.receiveTankTotalMassFlowOut = pyo.Constraint(rule=receiveTankTotalMassFlowOut_rule)

# 2. The mass flow To the centrifuge is controlled by the binary variable y_cf.
def flowToCentrifuge_rule(mod):
    return mod.flow_to_cf_mass <= M_mass * mod.y_cf
m.flowToCentrifuge_rule = pyo.Constraint(rule=flowToCentrifuge_rule)

# 3. The mass flow that BYPASSES the centrifuge is also controlled by y_cf.
def flowBypassCentrifuge_rule(mod):
    return mod.flow_bypass_cf_mass <= M_mass * (1 - mod.y_cf)
m.flowBypassCentrifuge_rule = pyo.Constraint(rule=flowBypassCentrifuge_rule)

# 4. Link the split mass stream to the centrifuge inlet.
def link_flow_to_cf_rule(mod):
    return mod.cf.sludgeMassFlowIn == mod.flow_to_cf_mass
m.link_flow_to_cf = pyo.Constraint(rule=link_flow_to_cf_rule)

def receiveTankToCentrifugeTSS_rule(mod):
    return mod.cf.sludgeTSSin == mod.rt.sludgeTSSout
m.receiveTankToCentrifugeTSS = pyo.Constraint(rule=receiveTankToCentrifugeTSS_rule)

# 5. Composition-resolved solids split 
m.sludgeSolidsAtRTOutlet_kg_s = pyo.Expression(expr=m.pt.sludgeSolidsMassFlowOut)
m.caoSolidsAtRTOutlet_kg_s    = pyo.Expression(expr=m.pt.caoSolidsMassFlowOut)
m.totalSolidsAtSplit_kg_s = pyo.Expression(expr=m.sludgeSolidsAtRTOutlet_kg_s + m.caoSolidsAtRTOutlet_kg_s)
m.sludgeSolidsFracAtSplit = pyo.Expression(expr=m.sludgeSolidsAtRTOutlet_kg_s / (m.totalSolidsAtSplit_kg_s + 1e-9))
m.caoSolidsFracAtSplit    = pyo.Expression(expr=1.0 - m.sludgeSolidsFracAtSplit)

m.cfPathTotalSolids_kg_s     = pyo.Expression(expr=m.flow_to_cf_mass * m.rt.sludgeTSSout)
m.bypassPathTotalSolids_kg_s = pyo.Expression(expr=m.flow_bypass_cf_mass * m.rt.sludgeTSSout)

m.cfCompositionSludge = pyo.Constraint(expr=m.cf.sludgeSolidsMassFlowIn == m.cfPathTotalSolids_kg_s * m.sludgeSolidsFracAtSplit)
m.cfCompositionCao    = pyo.Constraint(expr=m.cf.caoSolidsMassFlowIn    == m.cfPathTotalSolids_kg_s * m.caoSolidsFracAtSplit)

m.bypassSludgeSolids_kg_s = pyo.Expression(expr=m.bypassPathTotalSolids_kg_s * m.sludgeSolidsFracAtSplit)
m.bypassCaoSolids_kg_s    = pyo.Expression(expr=m.bypassPathTotalSolids_kg_s * m.caoSolidsFracAtSplit)

# 6. Dryer inlet, composition-resolved: solids/water mass from whichever path is active 
m.dr_sludge_solids_link = pyo.Constraint(
    expr=m.dr.sludgeSolidsFlowIn_mass == m.cf.sludgeSolidsCaptured_kg_s + m.bypassSludgeSolids_kg_s
)
m.dr_cao_solids_link = pyo.Constraint(
    expr=m.dr.caoSolidsFlowIn_mass == m.cf.caoSolidsCaptured_kg_s + m.bypassCaoSolids_kg_s
)
m.dr_water_link = pyo.Constraint(
    expr=m.dr.waterIn_mass == (
        m.cf.sludgeMassFlowOutSolid - m.cf.solidsS
        + m.flow_bypass_cf_mass * (1.0 - m.rt.sludgeTSSout)
    )
)

# --- Option A: Conditional minimums when ON + output gating ---
m.Vmin_cf = pyo.Param(initialize=0.1, mutable=True)   # m3 (minimum tank volume when ON)
m.Nmin_cf = pyo.Param(initialize=20.0, mutable=True)  # rps (minimum rotation when ON)
m.Vmax_cf = pyo.Param(initialize=2.0, mutable=True)   # m3 (reasonable upper bound when ON)
m.Dmax_cf = pyo.Param(initialize=1.0, mutable=True)   # m  (reasonable upper bound when ON)

m.cf_min_volume_on = pyo.Constraint(expr = m.cf.tankVolume >= m.Vmin_cf * m.y_cf)
m.cf_min_rpm_on    = pyo.Constraint(expr = m.cf.agitRotation >= m.Nmin_cf * m.y_cf)

# Gate centrifuge outlet mass flows so they are zero when OFF
m.cf_out_solid_gate  = pyo.Constraint(expr = m.cf.sludgeMassFlowOutSolid  <= M_mass * m.y_cf)
m.cf_out_liquid_gate = pyo.Constraint(expr = m.cf.sludgeMassFlowOutLiquid <= M_mass * m.y_cf)

eps_off = 1e-6
m.cf_max_volume_gate = pyo.Constraint(expr = m.cf.tankVolume   <= m.Vmax_cf * m.y_cf + eps_off * (1 - m.y_cf))
m.cf_max_diameter_gate = pyo.Constraint(expr = m.cf.tankDiameter <= m.Dmax_cf * m.y_cf + eps_off * (1 - m.y_cf))
m.cf_max_rps_gate      = pyo.Constraint(expr = m.cf.agitRotation <= 250.0   * m.y_cf + eps_off * (1 - m.y_cf))

# Dryer to Storage Tank 
m.dr_to_st_flow = pyo.Constraint(expr=m.st.productMassFlowIn == m.dr.productFlow_mass)

m.tonnesProductTotal = m.dr.productFlow_mass*10**-3*m.daysOperation  # in wet tonnes

# Estimated final product N (wet basis) at E-GROW product stream.
m.solidsNToLiquidFrac = pyo.Param(initialize=0.219, mutable=True)

# Solids-N into the electrolyzer/centrifuge path
m.solidsNInToCentrifuge_kg_s = pyo.Expression(expr=m.el.nitrogenFlowOut)

m.solidsNConvertedToLiquid_kg_s = pyo.Expression(
    expr=m.solidsNToLiquidFrac * m.solidsNInToCentrifuge_kg_s
)
m.solidsNRemainingToProduct_kg_s = pyo.Expression(
    expr=m.solidsNInToCentrifuge_kg_s - m.solidsNConvertedToLiquid_kg_s
)

m.organicSolidsCaptureRatio = pyo.Expression(
    expr=m.dr.sludgeSolidsMassFlowOut / (m.sludgeSolidsAtRTOutlet_kg_s + 1e-9)
)
m.totalSolidsCaptureRatio = pyo.Expression(
    expr=m.dr.solidsFlow_mass / (m.totalSolidsAtSplit_kg_s + 1e-9)
)

# Inlet dissolved N mass flow basis (kg-N/s)
m.inletLiquidNMassFlow_kg_s = pyo.Expression(
    expr=m.inletNitrogenConc * m.feedFlow_m3s / 1000.0
)

# Dissolved N concentration in the liquid at RT outlet (kg-N per kg-liquid).
m.liquidNConc_kgNPerKgLiquid = pyo.Expression(
    expr=m.rt.nitrogenConcOut / (m.liquidDensity * 1000.0)
)
m.productSolidsNMassFlow_kg_s = pyo.Expression(expr=m.solidsNRemainingToProduct_kg_s * m.organicSolidsCaptureRatio)
m.productLiquidMass_kg_s = pyo.Expression(
    expr=m.dr.productFlow_mass * (1.0 - m.dr.finalSolids_frac)
)

# Dryer pH link: the receive tank reprotonates the stream to its target pH
# (~7) before the [optional centrifuge]/dryer stage
m.dryerPHLink = pyo.Constraint(expr=m.dr.pHIn == m.rt.sludgepHout)

# Dissolved N mass flow entering the dryer with the cake/bypass liquid
m.nitrogenMassFlowIntoDryer_kg_s = pyo.Expression(
    expr=m.liquidNConc_kgNPerKgLiquid * m.dr.waterIn_mass
)
m.dryerNitrogenInLink = pyo.Constraint(
    expr=m.dr.nitrogenMassFlowIn_kgN_s == m.nitrogenMassFlowIntoDryer_kg_s
)

m.productNVolatilized_kg_s = pyo.Expression(expr=m.dr.nitrogenMassFlowVapor_kgN_s)
m.productLiquidNMassFlow_kg_s = pyo.Expression(expr=m.dr.nitrogenMassFlowOut_kgN_s)
m.productNMassFlow_kg_s = pyo.Expression(
    expr=m.productSolidsNMassFlow_kg_s + m.productLiquidNMassFlow_kg_s
)
m.productN_wtPercent_wet = pyo.Expression(
    expr=100.0 * m.productNMassFlow_kg_s / (m.dr.productFlow_mass + 1e-9)
)

# Dissolved N concentration in centrifuge liquid effluent (g-N/m3)
m.cfLiquidEffluentNConc_g_m3 = pyo.Expression(
    expr=(m.liquidNConc_kgNPerKgLiquid * m.liquidDensity * 1000.0)
)

# ------------------------------------------------------------------
# Phosphorus and potassium accounting
# ------------------------------------------------------------------
m.solidsPFraction = pyo.Param(initialize=0.025, mutable=True)    # kg-P/kg-dry-solids
m.solidsKFraction = pyo.Param(initialize=0.0035, mutable=True)   # kg-K/kg-dry-solids

m.centratePConc_mgL = pyo.Param(initialize=13.0, mutable=True)   # mg-P/L
m.centrateKConc_mgL = pyo.Param(initialize=275.0, mutable=True)  # mg-K/L

m.feedSolidsP_kg_s = pyo.Expression(expr=m.solidsPFraction * m.feedDrySolids_kg_s)
m.feedSolidsK_kg_s = pyo.Expression(expr=m.solidsKFraction * m.feedDrySolids_kg_s)

# Explicit mass-flow -> volumetric-flow conversion (kg/s -> L/s) via the
# stated liquid density, made explicit instead of a bare "/1000.0".
m.feedLiquidVolFlow_L_s = pyo.Expression(
    expr=(m.feedLiquid_kg_s / m.liquidDensity) * 1000.0
)
m.feedLiquidP_kg_s = pyo.Expression(expr=m.centratePConc_mgL * m.feedLiquidVolFlow_L_s / 1.0e6)
m.feedLiquidK_kg_s = pyo.Expression(expr=m.centrateKConc_mgL * m.feedLiquidVolFlow_L_s / 1.0e6)

m.feedTotalP_kg_s = pyo.Expression(expr=m.feedSolidsP_kg_s + m.feedLiquidP_kg_s)
m.feedTotalK_kg_s = pyo.Expression(expr=m.feedSolidsK_kg_s + m.feedLiquidK_kg_s)

# FS1 conditions the feed to pH~13 with CaO, so soluble P co-precipitates with
# excess Ca2+ and reports quantitatively with the solids. K is a passive
# tracer following whatever water reaches the dryer inlet.
m.addedWater_kg_s = pyo.Expression(expr=m.pt.waterMassFlowIn)
m.totalLiquidPool_kg_s = pyo.Expression(expr=m.feedLiquid_kg_s + m.addedWater_kg_s)
m.liquidRetentionFrac = pyo.Expression(
    expr=m.dr.waterIn_mass / (m.totalLiquidPool_kg_s + 1e-9)
)

m.productP_kg_s = pyo.Expression(
    expr=m.feedSolidsP_kg_s * m.organicSolidsCaptureRatio + m.feedLiquidP_kg_s * m.totalSolidsCaptureRatio
)
m.productK_kg_s = pyo.Expression(expr=m.feedTotalK_kg_s * m.liquidRetentionFrac)

m.productP_wtPercent_wet = pyo.Expression(expr=100.0 * m.productP_kg_s / (m.dr.productFlow_mass + 1e-9))
m.productK_wtPercent_wet = pyo.Expression(expr=100.0 * m.productK_kg_s / (m.dr.productFlow_mass + 1e-9))

# Explicit product solids target at flowsheet level 
m.product_tss_target = pyo.Constraint(expr=m.dr.finalSolids_frac == m.targetProductTSS)

# Calculate costs per tonne of product for each unit and total
m.pt.capexPerTonne = pyo.Expression(expr=m.pt.capex/(m.tonnesProductTotal + 1e-6))
m.pt.opexPerTonne = pyo.Expression(expr=m.pt.opex/(m.tonnesProductTotal + 1e-6))
m.el.capexPerTonne = pyo.Expression(expr=m.el.capex/(m.tonnesProductTotal + 1e-6))
m.el.opexPerTonne = pyo.Expression(expr=m.el.opex/(m.tonnesProductTotal + 1e-6))
m.rt.capexPerTonne = pyo.Expression(expr=m.rt.capex/(m.tonnesProductTotal + 1e-6))
m.rt.opexPerTonne = pyo.Expression(expr=m.rt.opex/(m.tonnesProductTotal + 1e-6))
m.cf.capexPerTonne = pyo.Expression(expr=m.y_cf * m.cf.capex / (m.tonnesProductTotal + 1e-6))
m.cf.opexPerTonne = pyo.Expression(expr=m.y_cf * m.cf.opex / (m.tonnesProductTotal + 1e-6))
m.dr.capexPerTonne = pyo.Expression(expr=m.dr.capex/(m.tonnesProductTotal + 1e-6))
m.dr.opexPerTonne = pyo.Expression(expr=m.dr.opex/(m.tonnesProductTotal + 1e-6))
m.st.capexPerTonne = pyo.Expression(expr=m.st.capex/(m.tonnesProductTotal + 1e-6))
m.st.opexPerTonne = pyo.Expression(expr=m.st.opex/(m.tonnesProductTotal + 1e-6))

# capex contains multiplication by 1.32 to account for additional costs like installation, engineering, and contingency.
m.capex = pyo.Expression(expr= (m.pt.capexPerTonne + m.el.capexPerTonne + m.rt.capexPerTonne + m.cf.capexPerTonne + m.dr.capexPerTonne + m.st.capexPerTonne)*1.32)
m.opex  = pyo.Expression(expr=m.pt.opexPerTonne + m.el.opexPerTonne + m.rt.opexPerTonne + m.cf.opexPerTonne + m.dr.opexPerTonne + m.st.opexPerTonne)

m.totalCost = pyo.Objective(rule = m.capex + m.opex, sense=pyo.minimize)

tol = 1e-2
solver = pyo.SolverFactory('ipopt')
solver = pyo.SolverFactory('baron',options = {'EpsA': 100*tol, 'AbsConFeasTol': 1*tol, 'TDo':0, 'MDo':0,'OBTTDo':1}, executable ='C:/baron/baron.exe')
sol = solver.solve(m, tee=True)


print("================ FEED / TARGETS ==================")
print("Feed mass flow (kg/day):", pyo.value(m.feedMassFlow_kg_s) * 86400)
print("Feed flow (m3/day, reporting only):", pyo.value(m.feedFlow_m3s) * 86400)
print("Feed TSS (fraction):", pyo.value(m.feedTSS))
print("Feed dissolved N (g-N/m3):", pyo.value(m.inletNitrogenConc))
print("Target product TSS (fraction):", pyo.value(m.targetProductTSS))
print("==================================================")

print("Prep Tank Sludge In Mass Flow (kg/day):", pyo.value(m.pt.sludgeMassFlowIn)*86400)
print("Prep Tank Water In Mass Flow (kg/day):", pyo.value(m.pt.waterMassFlowIn)*86400)
print("Prep Tank CaO Mass Flow (kg/day):", pyo.value(m.pt.totalCaO_kgPerS)*86400)
print("Prep Tank Sludge Out Mass Flow (kg/day):", pyo.value(m.pt.sludgeMassFlowOut)*86400)
print("Prep Tank Sludge TSS In (fraction):", pyo.value(m.pt.sludgeTSSin))
print("Prep Tank Sludge TSS Out (fraction):", pyo.value(m.pt.sludgeTSSout))
print("Prep Tank Volume (m3):", pyo.value(m.pt.tankVolume))
print("Prep Tank Lime Tank Volume (m3):", pyo.value(m.pt.limeTankVolume))
print("Prep Tank Base Cost ($/t):", pyo.value(m.pt.opex/(m.tonnesProductTotal + 1e-6)))
print("-------------------------------")
print("Electrolyzer Sludge In Mass Flow (kg/day):", pyo.value(m.el.sludgeMassFlowIn)*86400)
print("Electrolyzer Sludge Out Mass Flow (kg/day):", pyo.value(m.el.sludgeMassFlowOut)*86400)
print("Electrolyzer Sludge Dry Basis (kg/day):", pyo.value(m.el.sludgeOutDryBasis)*86400)
print("Electrolyzer Nitrogen Out Flow (kg-N/day):", pyo.value(m.el.nitrogenFlowOut)*86400)
print("Electrolyzer N loss fraction (volatilization + oxidation, placeholder):", pyo.value(m.el.nitrogenLossFraction))
print("Electrolyzer N volatilized (kg-N/s):", pyo.value(m.el.nitrogenFlowVolatilized_kgN_s))
print("Electrolyzer N volatilized (kg-N/day):", pyo.value(m.el.nitrogenFlowVolatilized_kgN_s)*86400)
print('Phosphorus and potassium (rudimentary, top-level; see chat log for sourcing):')
print('  Feed P total (kg/day):', pyo.value(m.feedTotalP_kg_s * 86400.0))
print('  Feed K total (kg/day):', pyo.value(m.feedTotalK_kg_s * 86400.0))
print('  Product P (kg/day):', pyo.value(m.productP_kg_s * 86400.0))
print('  Product K (kg/day):', pyo.value(m.productK_kg_s * 86400.0))
print('  Product P (wt%% wet):', pyo.value(m.productP_wtPercent_wet))
print('  Product K (wt%% wet):', pyo.value(m.productK_wtPercent_wet))
print('  Product P2O5 (wt%% wet, = P wt%% x 2.29):', pyo.value(m.productP_wtPercent_wet * 2.29))
print('  Product K2O (wt%% wet, = K wt%% x 1.20):', pyo.value(m.productK_wtPercent_wet * 1.20))
print("Electrolyzer Sludge TSS Out (fraction):", pyo.value(m.el.sludgeTSSout))
print("Electrolyzer Area (m2):", pyo.value(m.el.area))
print("-------------------------------")
print("Receive Tank Sludge In Mass Flow (kg/day):", pyo.value(m.rt.sludgeMassFlowIn)*86400)
print("Receive Tank Sludge Out Mass Flow (kg/day):", pyo.value(m.rt.sludgeMassFlowOut)*86400)
print("Receive Tank Acid In Mass Flow (kg/day):", pyo.value(m.rt.acidMassFlowIn)*86400)
print("Receive Tank Sludge TSS Out (fraction):", pyo.value(m.rt.sludgeTSSout))
print("Receive Tank Volume (m3):", pyo.value(m.rt.tankVolume))
print("Receive Tank acid Cost ($/t):", pyo.value(m.rt.opex/(m.tonnesProductTotal + 1e-6)))
print("-------------------------------")
print("Centrifuge Active (y_cf):", pyo.value(m.y_cf))
print("Centrifuge Sludge In Mass Flow (kg/day):", pyo.value(m.cf.sludgeMassFlowIn)*86400)
print("Centrifuge Capture Rate :", pyo.value(m.cf.solidMassCaptured))
print("Centrifuge Sludge TSS In (fraction):", pyo.value(m.cf.sludgeTSSin))
print("Centrifuge Sludge Out Solid Mass Flow (kg/day):", pyo.value(m.cf.sludgeMassFlowOutSolid)*86400)
print("Centrifuge Sludge Out Liquid Mass Flow (kg/day):", pyo.value(m.cf.sludgeMassFlowOutLiquid)*86400)
print("Centrifuge Solid TSS Out (fraction):", pyo.value(m.cf.sludgeTSSoutSolid))
print("Centrifuge Liquid TSS Out (fraction):", pyo.value(m.cf.sludgeTSSoutLiquid))
print("Centrifuge Cake CaO Weight Fraction (fraction):", pyo.value(m.cf.caoWeightFractionInCake))
print("Centrifuge Liquid Effluent N Conc (g-N/m3):", pyo.value(m.cfLiquidEffluentNConc_g_m3))
print("Centrifuge Capture Rate (1/s):", pyo.value(m.cf.captureRate))
print("Centrifuge Residence Time (s):", pyo.value(m.cf.residenceTime))
print("Centrifuge Volume (m3):", pyo.value(m.cf.tankVolume))
print("Centrifuge Diameter (m):", pyo.value(m.cf.tankDiameter))
print("Centrifuge Agitation Rotation (rps):", pyo.value(m.cf.agitRotation))
print("Centrifuge Pressure (bar):", pyo.value(m.cf.P_c)/10**5)
print("Centrifuge Geo Residual V-1.178*D^3 (m3):", pyo.value(m.cf.tankVolume - 1.178*m.cf.tankDiameter**3))
print("Solids-N into centrifuge (kg-N/s):", pyo.value(m.solidsNInToCentrifuge_kg_s))
print("Solids-N converted to liquid NH3 (kg-N/s):", pyo.value(m.solidsNConvertedToLiquid_kg_s))
print("-------------------------------")
print("Dryer Solids In Mass Flow (kg/day):", pyo.value(m.dr.solidsFlow_mass)*86400)
print("Dryer Water In Mass Flow (kg/day):", pyo.value(m.dr.waterIn_mass)*86400)
print("Dryer Inlet Solids Content In (fraction, reporting):", pyo.value(m.dr.sludgeTSSIn_frac))
print("Dryer Air Flow In (m3/s):", pyo.value(m.dr.airFlowIn))
print("Dryer Air Temp In (C):", pyo.value(m.dr.airTempIn))
print("Dryer Air Temp Out (C, solved):", pyo.value(m.dr.airTempOut))
print("Dryer Humidity Ratio at saturation, T_out (kg/kg):", pyo.value(m.dr.humidityRatioSat))
print("Dryer Henry's constant for NH3 at T_out (mol/L/atm):", pyo.value(m.dr.henryNH3_T))
print("Dryer liquid pH in (from receive tank):", pyo.value(m.dr.pHIn))
print("Dryer free-ammonia fraction at pH in:", pyo.value(m.dr.freeAmmoniaFrac))
print("Dryer N stripping fraction (ammonia volatilized):", pyo.value(m.dr.strippingFraction))
print("Dryer Solids Content Out (fraction):", pyo.value(m.dr.finalSolids_frac))
print("Dryer Product CaO Weight Fraction (fraction):", pyo.value(m.dr.caoWeightFractionInProduct))
print("Dryer Water Vapor Out Flow (kg/day):", pyo.value(m.dr.waterVaporFlowOut)*86400)
print("Dryer Total Product Flow (kg/s):", pyo.value(m.dr.productFlow_mass))
print("Dryer Heat Duty (kW):", pyo.value(m.dr.heatDuty))
print("Dryer Blower Power (kW):", pyo.value(m.dr.blowerPower))
print("Inlet liquid N concentration (g-N/m3):", pyo.value(m.inletNitrogenConc))
print("Solids-N remaining to product (kg-N/s):", pyo.value(m.solidsNRemainingToProduct_kg_s))
print("Product Solids N Mass Flow (kg-N/s):", pyo.value(m.productSolidsNMassFlow_kg_s))
print("Product Liquid N Mass Flow (kg-N/s):", pyo.value(m.productLiquidNMassFlow_kg_s))
print("Product Total Mass Flow (kg/day):", pyo.value(m.dr.productFlow_mass)*86400)
print("Final E-GROW Product N Mass Flow (kg-N/s):", pyo.value(m.productNMassFlow_kg_s))
print("Final E-GROW Product N Mass Flow (kg-N/day):", pyo.value(m.productNMassFlow_kg_s*86400))
print("Final E-GROW Product N (wt% wet):", pyo.value(m.productN_wtPercent_wet))

# Nitrogen recovery computation (combines solid and liquid nitrogen; now
# also accounts for BOTH the electrolyzer's and the dryer's ammonia
# volatilization losses, plus what leaves via the centrifuge's liquid
# effluent -- broken out explicitly below so the full balance is legible
# rather than lumped into a single unexplained "recovery" gap.)
inletDissolvedNitrogenMassFlow_kg_s = pyo.value(m.inletLiquidNMassFlow_kg_s)
solidBoundNitrogenFromElectrolyzer_kg_s = pyo.value(m.solidsNInToCentrifuge_kg_s)
liquidTransferredNitrogenFromSolids_kg_s = pyo.value(m.solidsNConvertedToLiquid_kg_s)
totalInletNitrogenMassFlow_kg_s = inletDissolvedNitrogenMassFlow_kg_s + solidBoundNitrogenFromElectrolyzer_kg_s
totalOutletNitrogenMassFlow_kg_s = pyo.value(m.productNMassFlow_kg_s)
nitrogenVolatilizedDryer_kg_s = pyo.value(m.productNVolatilized_kg_s)
nitrogenVolatilizedElectrolyzer_kg_s = pyo.value(m.el.nitrogenFlowVolatilized_kgN_s)
nitrogenVolatilizedTotal_kg_s = nitrogenVolatilizedDryer_kg_s + nitrogenVolatilizedElectrolyzer_kg_s
# Whatever isn't in the product and isn't accounted for by volatilization
# left via the centrifuge's liquid effluent (centrate) -- see the FS0
# discussion of the same phenomenon; this is the dominant loss term when
# the centrifuge is active (y_cf=1), since most of the liquid phase never
# enters the dryer at all.
nitrogenLostToCentrate_kg_s = max(0.0, totalInletNitrogenMassFlow_kg_s - totalOutletNitrogenMassFlow_kg_s - nitrogenVolatilizedTotal_kg_s)
nitrogenRecoveryPercent = 100.0 * totalOutletNitrogenMassFlow_kg_s / (totalInletNitrogenMassFlow_kg_s + 1e-9)
nitrogenVolatilizedPercent = 100.0 * nitrogenVolatilizedTotal_kg_s / (totalInletNitrogenMassFlow_kg_s + 1e-9)
nitrogenLostToCentratePercent = 100.0 * nitrogenLostToCentrate_kg_s / (totalInletNitrogenMassFlow_kg_s + 1e-9)

print("Inlet dissolved N mass flow (kg-N/s):", inletDissolvedNitrogenMassFlow_kg_s)
print("Inlet solids-bound N mass flow (kg-N/s):", solidBoundNitrogenFromElectrolyzer_kg_s)
print("Solids-N transferred to liquid (kg-N/s):", liquidTransferredNitrogenFromSolids_kg_s)
print("Total inlet nitrogen mass flow (kg-N/s):", totalInletNitrogenMassFlow_kg_s)
print("Total outlet nitrogen mass flow - solids (kg-N/s):", pyo.value(m.productSolidsNMassFlow_kg_s))
print("Total outlet nitrogen mass flow - liquid (kg-N/s):", pyo.value(m.productLiquidNMassFlow_kg_s))
print("Nitrogen volatilized in electrolyzer off-gas (kg-N/s):", nitrogenVolatilizedElectrolyzer_kg_s)
print("Nitrogen volatilized in dryer exhaust (kg-N/s):", nitrogenVolatilizedDryer_kg_s)
print("Nitrogen volatilized total, electrolyzer + dryer (kg-N/s):", nitrogenVolatilizedTotal_kg_s)
print("Nitrogen lost to centrifuge liquid effluent, i.e. centrate (kg-N/s):", nitrogenLostToCentrate_kg_s)
print("Nitrogen lost to centrate (% of inlet N):", nitrogenLostToCentratePercent)
print("Total outlet nitrogen mass flow (kg-N/s):", totalOutletNitrogenMassFlow_kg_s)
print("Nitrogen recovery (%):", nitrogenRecoveryPercent)
print("Nitrogen volatilized (% of inlet N, electrolyzer + dryer):", nitrogenVolatilizedPercent)
print("Nitrogen lost to centrate (% of inlet N):", nitrogenLostToCentratePercent)

print("-------------------------------")
print("Prep Tank Capex ($):", pyo.value(m.pt.capex))
print("Prep Tank Opex ($):", pyo.value(m.pt.opex))
print("Electrolyzer Capex ($):", pyo.value(m.el.capex))
print("Electrolyzer Opex ($):", pyo.value(m.el.opex))
print("Receive Tank Capex ($):", pyo.value(m.rt.capex))
print("Receive Tank Opex ($):", pyo.value(m.rt.opex))
print("Centrifuge Capex ($):", pyo.value(m.cf.capex))
print("Centrifuge Opex ($):", pyo.value(m.cf.opex))
print("Dryer Capex ($):", pyo.value(m.dr.capex))
print("Dryer Opex ($):", pyo.value(m.dr.opex))
print("Storage Tank Capex ($):", pyo.value(m.st.capex))
print("Storage Tank Opex ($):", pyo.value(m.st.opex))
print("-------------------------------")
print("Prep Tank Capex ($/t):", pyo.value(1.32 * m.pt.capexPerTonne))
print("Prep Tank Opex ($/t):", pyo.value(m.pt.opexPerTonne))
print("Electrolyzer Capex ($/t):", pyo.value(1.32 * m.el.capexPerTonne))
print("Electrolyzer Opex ($/t):", pyo.value(m.el.opexPerTonne))
print("Receive Tank Capex ($/t):", pyo.value(1.32*m.rt.capexPerTonne))
print("Receive Tank Opex ($/t):", pyo.value(m.rt.opexPerTonne))
print("Centrifuge Capex ($/t):", pyo.value(1.32*m.cf.capexPerTonne))
print("Centrifuge Opex ($/t):", pyo.value(m.cf.opexPerTonne))
print("Dryer Capex ($/t):", pyo.value(1.2*m.dr.capexPerTonne))
print("Dryer Opex ($/t):", pyo.value(m.dr.opexPerTonne))
print("Storage Tank Capex ($/t):", pyo.value(1.32 * m.st.capexPerTonne))
print("Storage Tank Opex ($/t):", pyo.value(m.st.opexPerTonne))
print("-------------------------------")
print("Total Capex ($/t):", pyo.value(m.capex))
print("Total Opex ($/t):", pyo.value(m.opex))
print("Total Cost ($/t):", pyo.value(m.capex + m.opex))
print("-------------------------------")


print("================ INPUTS / ELECTRICITY / CHEMICALS (FS1) =================")

def _safe_value_end(expr):
    try:
        return pyo.value(expr)
    except Exception:
        return 'n/a'

# Mass inputs (kg/day) -- all direct, no volume/density round-trips
fs1_feed_sludge_kg_day = m.feedMassFlow_kg_s * 86400.0
fs1_water_added_kg_day = m.pt.waterMassFlowIn * 86400.0
fs1_cao_slurry_kg_day = m.pt.totalCaO_kgPerS * 86400.0
fs1_acid_solution_kg_day = m.rt.acidMassFlowIn * 86400.0
fs1_h2so4_kg_day = m.rt.h2so4RequiredKgPerS * 86400.0
fs1_acid_equiv_mol_day = m.rt.totalAcidEqPerS * 86400.0
fs1_egrow_product_kg_day = m.dr.productFlow_mass * 86400.0
fs1_nitrogen_in_egrow_kg_day = m.productNMassFlow_kg_s * 86400.0

print('Mass inputs (kg/day):')
print('  Feed sludge:', _safe_value_end(fs1_feed_sludge_kg_day))
print('  Added water (prep tank):', _safe_value_end(fs1_water_added_kg_day))
print('  CaO/base stream (prep tank):', _safe_value_end(fs1_cao_slurry_kg_day))
print('  Acid stream (receive tank):', _safe_value_end(fs1_acid_solution_kg_day))
print('  Receive-tank H2SO4 equivalent:', _safe_value_end(fs1_h2so4_kg_day))
print('  Receive-tank acid equivalents (mol/day):', _safe_value_end(fs1_acid_equiv_mol_day))
print('  E-GROW produced (kg/day):', _safe_value_end(fs1_egrow_product_kg_day))
print('  Nitrogen in E-GROW product (kg/day):', _safe_value_end(fs1_nitrogen_in_egrow_kg_day))


# Electricity requirement (unit model basis)
fs1_el_power_kW = m.el.power
fs1_cf_power_kW = m.cf.power_kW
fs1_dryer_heat_kW = m.dr.heatDuty
fs1_dryer_blower_kW = m.dr.blowerPower

fs1_el_energy_kWh_day = fs1_el_power_kW * 24.0
fs1_cf_energy_kWh_day = fs1_cf_power_kW * 24.0
# Per-m3 normalization uses the centrifuge's own sizing-only volumetric flow.
fs1_cf_energy_kWh_m3 = fs1_cf_energy_kWh_day / (m.cf.sludgeVolFlowIn * 86400.0)
fs1_dryer_heat_energy_kWh_day = fs1_dryer_heat_kW * 24.0
fs1_dryer_blower_energy_kWh_day = fs1_dryer_blower_kW * 24.0

print('Electricity requirement by unit (power kW, energy kWh/day):')
print('  Electrolyzer:', 'kW =', _safe_value_end(fs1_el_power_kW), ', kWh/day =', _safe_value_end(fs1_el_energy_kWh_day))
print('  Centrifuge shaft:', 'kW =', _safe_value_end(fs1_cf_power_kW), ', kWh/day =', _safe_value_end(fs1_cf_energy_kWh_day))
print('  Dryer heat duty:', 'kW =', _safe_value_end(fs1_dryer_heat_kW), ', kWh/day =', _safe_value_end(fs1_dryer_heat_energy_kWh_day))
print('  Dryer blower:', 'kW =', _safe_value_end(fs1_dryer_blower_kW), ', kWh/day =', _safe_value_end(fs1_dryer_blower_energy_kWh_day))
print('  Centrifuge energy per m3 (sizing basis):', 'kWh/m3 =', _safe_value_end(fs1_cf_energy_kWh_m3))

print('Chemical usage (kg/day):')
print('  CaO/base stream:', _safe_value_end(fs1_cao_slurry_kg_day))
print('  Acid stream:', _safe_value_end(fs1_acid_solution_kg_day))
print('========================================================================')
