#------------------------------------------------------------------------------
# function:    flowsheet0.py                                                  #
# Description: Baseline optimization model for wastewater treatment process   #
#              Process flow: Feed -> Centrifuge -> Dryer -> Storage Tank      #
#                                                                             #
#                                                                             #
# Input:       - m : Pyomo concrete model                                     #
#                                                                             #
# Output:      - m with all parameters, variables, constraints, and objective #
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
m.centrateFlow_m3s        = pyo.Param(initialize=214/86400.0,    mutable=True)  # m3/s
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
m.feedTSS = pyo.Expression(
    expr=m.feedDrySolids_kg_s / m.feedMassFlow_kg_s
)

# Reporting-only combined volumetric flow (not used by any downstream unit).
m.feedFlow_m3s = pyo.Expression(
    expr=m.dewateredSludgeFlow_m3s + m.centrateFlow_m3s
)

m.inletNitrogenConc = pyo.Param(initialize=750.0, mutable=True)        # g-N/m3 (750 ppm, from centrate)
m.targetProductTSS = pyo.Param(initialize=0.90, mutable=True)         # 90 wt fraction target

m.y_cf = pyo.Param(initialize=1)

# Initialize unit Blocks
centrifuge.centrifuge(m)
dryer.dryer(m)
storageTank.storageTank(m)

# Minimum practical centrifuge sizing 
m.Vmin_cf = pyo.Param(initialize=0.1, mutable=True)   # m3
m.Nmin_cf = pyo.Param(initialize=20.0, mutable=True)  # rps
m.cf_min_volume = pyo.Constraint(expr=m.cf.tankVolume >= m.Vmin_cf)
m.cf_min_rpm = pyo.Constraint(expr=m.cf.agitRotation >= m.Nmin_cf)

m.Vmax_cf = pyo.Param(initialize=2.0, mutable=True)   # m3
m.cf_max_volume = pyo.Constraint(expr=m.cf.tankVolume <= m.Vmax_cf)

# --- Direct Connections: Feed -> Centrifuge -> Dryer -> Storage ---

def feedToCentrifugeMassFlow_rule(mod):
    return mod.cf.sludgeMassFlowIn == mod.feedMassFlow_kg_s
m.feedToCentrifugeMassFlow = pyo.Constraint(rule=feedToCentrifugeMassFlow_rule)

def feedToCentrifugeTSS_rule(mod):
    return mod.cf.sludgeTSSin == mod.feedTSS
m.feedToCentrifugeTSS = pyo.Constraint(rule=feedToCentrifugeTSS_rule)


m.cfCompositionSludge = pyo.Constraint(expr=m.cf.sludgeSolidsMassFlowIn == m.cf.solidsIn)
m.cfCompositionCao = pyo.Constraint(expr=m.cf.caoSolidsMassFlowIn == 0.0)

# Dryer solids = centrifuge captured solids
m.dr_sludge_solids_link = pyo.Constraint(
    expr=m.dr.sludgeSolidsFlowIn_mass == m.cf.sludgeSolidsCaptured_kg_s
)
m.dr_cao_solids_link = pyo.Constraint(
    expr=m.dr.caoSolidsFlowIn_mass == m.cf.caoSolidsCaptured_kg_s
)
# Dryer water = cake liquid mass flow (total cake mass minus its solids content)
m.dr_water_link = pyo.Constraint(
    expr=m.dr.waterIn_mass == (m.cf.sludgeMassFlowOutSolid - m.cf.solidsS)
)

# Explicit product solids target at flowsheet level
m.product_tss_target = pyo.Constraint(expr=m.dr.finalSolids_frac == m.targetProductTSS)

# FS0 has no pH-adjustment unit 
m.dryerFeedPH = pyo.Param(initialize=7.5, mutable=True)
m.dryerPHLink = pyo.Constraint(expr=m.dr.pHIn == m.dryerFeedPH)

# Dryer to Storage Tank
m.dr_to_st_flow = pyo.Constraint(expr=m.st.productMassFlowIn == m.dr.productFlow_mass)

m.tonnesProductTotal = m.dr.productFlow_mass * 10**-3 * m.daysOperation  # in wet tonnes

# Estimated final product N (wet basis) at E-GROW product stream.
m.productSolidsNMassFrac = pyo.Param(initialize=0.05, mutable=True)  # kg-N/kg-solids

# Solids-N into the centrifuge is just the feed dry-solids mass flow
m.solidsNInToCentrifuge_kg_s = pyo.Expression(
    expr=m.productSolidsNMassFrac * m.feedDrySolids_kg_s
)
# No liberation in FS0
m.solidsNRemainingToProduct_kg_s = pyo.Expression(
    expr=m.solidsNInToCentrifuge_kg_s
)

# Not all organic solids entering the centrifuge are captured into the cake
# -- some are lost to the liquid effluent per solidMassCaptured. The N 
# (and native P, further below) associated with that lost fraction leaves 
# with the centrate, not the product

m.organicSolidsCaptureRatio = pyo.Expression(
    expr=m.dr.sludgeSolidsMassFlowOut / (m.feedDrySolids_kg_s + 1e-9)
)

# Inlet dissolved N mass flow basis (kg-N/s)
m.inletLiquidNMassFlow_kg_s = pyo.Expression(
    expr=m.inletNitrogenConc * m.feedFlow_m3s / 1000.0
)

# Dissolved N concentration in the liquid fed into the centrifuge (kg-N per kg-liquid). 
m.liquidNConc_kgNPerKgLiquid = pyo.Expression(
    expr=m.inletLiquidNMassFlow_kg_s / (m.feedLiquid_kg_s + 1e-9)
)

m.productSolidsNMassFlow_kg_s = pyo.Expression(expr=m.solidsNRemainingToProduct_kg_s * m.organicSolidsCaptureRatio)
# Product liquid mass that actually ends up in the final product
m.productLiquidMass_kg_s = pyo.Expression(
    expr=m.dr.productFlow_mass * (1.0 - m.dr.finalSolids_frac)
)

# Dissolved N mass flow entering the dryer with the cake liquid 
m.nitrogenMassFlowIntoDryer_kg_s = pyo.Expression(
    expr=m.liquidNConc_kgNPerKgLiquid * m.dr.waterIn_mass
)
m.dryerNitrogenInLink = pyo.Constraint(
    expr=m.dr.nitrogenMassFlowIn_kgN_s == m.nitrogenMassFlowIntoDryer_kg_s
)

# Nitrogen lost to the dryer exhaust (ammonia volatilization) and nitrogen
# retained in the liquid that ends up in the product
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
# Solids-phase P/K
m.solidsPFraction = pyo.Param(initialize=0.025, mutable=True)    # kg-P/kg-dry-solids
m.solidsKFraction = pyo.Param(initialize=0.0035, mutable=True)   # kg-K/kg-dry-solids

# Centrate (liquid-phase) P/K:
m.centratePConc_mgL = pyo.Param(initialize=13.0, mutable=True)   # mg-P/L
m.centrateKConc_mgL = pyo.Param(initialize=275.0, mutable=True)  # mg-K/L

m.feedSolidsP_kg_s = pyo.Expression(expr=m.solidsPFraction * m.feedDrySolids_kg_s)
m.feedSolidsK_kg_s = pyo.Expression(expr=m.solidsKFraction * m.feedDrySolids_kg_s)

# Explicit mass-flow -> volumetric-flow conversion (kg/s -> L/s) for the
# feed liquid, using the stated liquid density -- needed because the P/K
# concentrations are reported per liter, not per kg.
m.feedLiquidVolFlow_L_s = pyo.Expression(
    expr=(m.feedLiquid_kg_s / m.liquidDensity) * 1000.0
)
m.feedLiquidP_kg_s = pyo.Expression(expr=m.centratePConc_mgL * m.feedLiquidVolFlow_L_s / 1.0e6)  # mg/L * L/s / 1e6 = kg/s
m.feedLiquidK_kg_s = pyo.Expression(expr=m.centrateKConc_mgL * m.feedLiquidVolFlow_L_s / 1.0e6)

m.feedTotalP_kg_s = pyo.Expression(expr=m.feedSolidsP_kg_s + m.feedLiquidP_kg_s)
m.feedTotalK_kg_s = pyo.Expression(expr=m.feedSolidsK_kg_s + m.feedLiquidK_kg_s)

# FS0 has no CaO/electrolyzer stage, so there is no Ca-driven precipitation
# mechanism to carry centrate-P onto the solids the way FS1-FS5 can. Both
# centrate-P and centrate-K are therefore treated identically here: passive,
# non-reactive tracers that follow whatever water reaches the DRYER
m.liquidRetentionFrac = pyo.Expression(
    expr=m.dr.waterIn_mass / (m.feedLiquid_kg_s + 1e-9)
)

m.productP_kg_s = pyo.Expression(
    expr=m.feedSolidsP_kg_s * m.organicSolidsCaptureRatio + m.feedLiquidP_kg_s * m.liquidRetentionFrac
)
m.productK_kg_s = pyo.Expression(expr=m.feedTotalK_kg_s * m.liquidRetentionFrac)

m.productP_wtPercent_wet = pyo.Expression(expr=100.0 * m.productP_kg_s / (m.dr.productFlow_mass + 1e-9))
m.productK_wtPercent_wet = pyo.Expression(expr=100.0 * m.productK_kg_s / (m.dr.productFlow_mass + 1e-9))

# Calculate costs per tonne of product for each unit and total (PT, EL, RT removed)
m.cf.capexPerTonne = pyo.Expression(expr=m.cf.capex / (m.tonnesProductTotal + 1e-6))
m.cf.opexPerTonne = pyo.Expression(expr=m.cf.opex / (m.tonnesProductTotal + 1e-6))
m.dr.capexPerTonne = pyo.Expression(expr=m.dr.capex / (m.tonnesProductTotal + 1e-6))
m.dr.opexPerTonne = pyo.Expression(expr=m.dr.opex / (2*m.tonnesProductTotal + 1e-6))
m.st.capexPerTonne = pyo.Expression(expr=m.st.capex / (m.tonnesProductTotal + 1e-6))
m.st.opexPerTonne = pyo.Expression(expr=m.st.opex / (m.tonnesProductTotal + 1e-6))

# capex contains multiplication by 1.32 to account for additional costs
m.capex = pyo.Expression(expr=(m.cf.capexPerTonne + m.dr.capexPerTonne + m.st.capexPerTonne) * 1.32)
m.opex  = pyo.Expression(expr=m.cf.opexPerTonne + m.dr.opexPerTonne + m.st.opexPerTonne)

m.totalCost = pyo.Objective(rule=m.capex + m.opex, sense=pyo.minimize)

tol = 1e-2
solver = pyo.SolverFactory('ipopt')
solver = pyo.SolverFactory('baron', options={'EpsA': 100*tol, 'AbsConFeasTol': 1*tol, 'TDo':0, 'MDo':0,'OBTTDo':1}, executable='C:/baron/baron.exe')
sol = solver.solve(m, tee=True)

# -----------------------------------------------------------------------------
# Print Outputs
# -----------------------------------------------------------------------------
print("================ FEED / TARGETS ==================")
print("Feed mass flow (kg/day):", pyo.value(m.feedMassFlow_kg_s) * 86400)
print("Feed flow (m3/day, reporting only):", pyo.value(m.feedFlow_m3s) * 86400)
print("Feed TSS (fraction):", pyo.value(m.feedTSS))
print("Feed dissolved N (g-N/m3):", pyo.value(m.inletNitrogenConc))
print("Target product TSS (fraction):", pyo.value(m.targetProductTSS))
print("==================================================")

print("Centrifuge Sludge In Mass Flow (kg/day):", pyo.value(m.cf.sludgeMassFlowIn)*86400)
print("Centrifuge Sludge In Vol Flow (m3/day, sizing-only):", pyo.value(m.cf.sludgeVolFlowIn)*86400)
print("Centrifuge Capture Rate :", pyo.value(m.cf.solidMassCaptured))
print("Centrifuge Sludge TSS In (fraction):", pyo.value(m.cf.sludgeTSSin))
print("Centrifuge Sludge Out Solid Mass Flow (kg/day):", pyo.value(m.cf.sludgeMassFlowOutSolid)*86400)
print("Centrifuge Sludge Out Liquid Mass Flow (kg/day):", pyo.value(m.cf.sludgeMassFlowOutLiquid)*86400)
print("Centrifuge Solid TSS Out (fraction):", pyo.value(m.cf.sludgeTSSoutSolid))
print("Centrifuge Liquid TSS Out (fraction):", pyo.value(m.cf.sludgeTSSoutLiquid))
print("Centrifuge Liquid Effluent N Conc (g-N/m3):", pyo.value(m.cfLiquidEffluentNConc_g_m3))
print("Centrifuge Capture Rate (1/s):", pyo.value(m.cf.captureRate))
print("Centrifuge Residence Time (s):", pyo.value(m.cf.residenceTime))
print("Centrifuge Volume (m3):", pyo.value(m.cf.tankVolume))
print("Centrifuge Diameter (m):", pyo.value(m.cf.tankDiameter))
print("Centrifuge Agitation Rotation (rps):", pyo.value(m.cf.agitRotation))
print("Centrifuge Pressure (bar):", pyo.value(m.cf.P_c)/10**5)
print("Centrifuge Geo Residual V-1.178*D^3 (m3):", pyo.value(m.cf.tankVolume - 1.178*m.cf.tankDiameter**3))
print("Solids-N into centrifuge (kg-N/s):", pyo.value(m.solidsNInToCentrifuge_kg_s))
print("-------------------------------")
print("Dryer Solids In Mass Flow (kg/day):", pyo.value(m.dr.solidsFlow_mass)*86400)
print("Dryer Water In Mass Flow (kg/day):", pyo.value(m.dr.waterIn_mass)*86400)
print("Dryer Inlet Solids Content In (fraction, reporting):", pyo.value(m.dr.sludgeTSSIn_frac))
print("Dryer Air Flow In (m3/s):", pyo.value(m.dr.airFlowIn))
print("Dryer Air Temp In (C):", pyo.value(m.dr.airTempIn))
print("Dryer Air Temp Out (C, solved):", pyo.value(m.dr.airTempOut))
print("Dryer Humidity Ratio at saturation, T_out (kg/kg):", pyo.value(m.dr.humidityRatioSat))
print("Dryer Humidity Ratio actually achieved (kg/kg):", pyo.value((m.dr.ambientHumidityRatio*m.dr.airMassFlowDry_kgPerS + m.dr.totalVolatileMassOut_kgPerS)/(m.dr.airMassFlowDry_kgPerS+1e-9)))
print("Dryer Henry's constant for NH3 at T_out (mol/L/atm):", pyo.value(m.dr.henryNH3_T))
print("Dryer liquid pH in:", pyo.value(m.dr.pHIn))
print("Dryer free-ammonia fraction at pH in:", pyo.value(m.dr.freeAmmoniaFrac))
print("Dryer N stripping fraction (ammonia volatilized):", pyo.value(m.dr.strippingFraction))
print("Dryer Solids Content Out (fraction):", pyo.value(m.dr.finalSolids_frac))
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

# Nitrogen recovery computation
inletDissolvedNitrogenMassFlow_kg_s = pyo.value(m.inletLiquidNMassFlow_kg_s)
solidBoundNitrogenFromFeed_kg_s = pyo.value(m.solidsNInToCentrifuge_kg_s)
totalInletNitrogenMassFlow_kg_s = inletDissolvedNitrogenMassFlow_kg_s + solidBoundNitrogenFromFeed_kg_s
totalOutletNitrogenMassFlow_kg_s = pyo.value(m.productNMassFlow_kg_s)
nitrogenVolatilized_kg_s = pyo.value(m.productNVolatilized_kg_s)
nitrogenRecoveryPercent = 100.0 * totalOutletNitrogenMassFlow_kg_s / (totalInletNitrogenMassFlow_kg_s + 1e-9)
nitrogenVolatilizedPercent = 100.0 * nitrogenVolatilized_kg_s / (totalInletNitrogenMassFlow_kg_s + 1e-9)

print("Inlet dissolved N mass flow (kg-N/s):", inletDissolvedNitrogenMassFlow_kg_s)
print("Inlet solids-bound N mass flow (kg-N/s):", solidBoundNitrogenFromFeed_kg_s)
print("Total inlet nitrogen mass flow (kg-N/s):", totalInletNitrogenMassFlow_kg_s)
print("Total outlet nitrogen mass flow - solids (kg-N/s):", pyo.value(m.productSolidsNMassFlow_kg_s))
print("Total outlet nitrogen mass flow - liquid (kg-N/s):", pyo.value(m.productLiquidNMassFlow_kg_s))
print("Nitrogen volatilized in dryer exhaust (kg-N/s):", nitrogenVolatilized_kg_s)
print("Total outlet nitrogen mass flow (kg-N/s):", totalOutletNitrogenMassFlow_kg_s)
print("Nitrogen recovery (%):", nitrogenRecoveryPercent)
print("Nitrogen volatilized (% of inlet N):", nitrogenVolatilizedPercent)

print("-------------------------------")
print("Centrifuge Capex ($):", pyo.value(m.cf.capex))
print("Centrifuge Opex ($):", pyo.value(m.cf.opex))
print("Dryer Capex ($):", pyo.value(m.dr.capex))
print("Dryer Opex ($):", pyo.value(m.dr.opex))
print("Storage Tank Capex ($):", pyo.value(m.st.capex))
print("Storage Tank Opex ($):", pyo.value(m.st.opex))
print("-------------------------------")
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


print("================ INPUTS / ELECTRICITY / CHEMICALS (FS0) =================")

def _safe_value_end(expr):
    try:
        return pyo.value(expr)
    except Exception:
        return 'n/a'

# Mass inputs (kg/day) -- direct
fs0_feed_sludge_kg_day = m.feedMassFlow_kg_s * 86400.0
fs0_egrow_product_kg_day = m.dr.productFlow_mass * 86400.0
fs0_nitrogen_in_egrow_kg_day = m.productNMassFlow_kg_s * 86400.0

print('Mass inputs (kg/day):')
print('  Feed sludge:', _safe_value_end(fs0_feed_sludge_kg_day))
print('  E-GROW produced (kg/day):', _safe_value_end(fs0_egrow_product_kg_day))
print('  Nitrogen in E-GROW product (kg/day):', _safe_value_end(fs0_nitrogen_in_egrow_kg_day))

print('Phosphorus and potassium (rudimentary, top-level; see chat log for sourcing):')
print('  Feed P total (kg/day):', _safe_value_end(m.feedTotalP_kg_s * 86400.0))
print('  Feed K total (kg/day):', _safe_value_end(m.feedTotalK_kg_s * 86400.0))
print('  Product P (kg/day):', _safe_value_end(m.productP_kg_s * 86400.0))
print('  Product K (kg/day):', _safe_value_end(m.productK_kg_s * 86400.0))
print('  Product P (wt%% wet):', _safe_value_end(m.productP_wtPercent_wet))
print('  Product K (wt%% wet):', _safe_value_end(m.productK_wtPercent_wet))
print('  Product P2O5 (wt%% wet, = P wt%% x 2.29):', _safe_value_end(m.productP_wtPercent_wet * 2.29))
print('  Product K2O (wt%% wet, = K wt%% x 1.20):', _safe_value_end(m.productK_wtPercent_wet * 1.20))

# Electricity requirement (unit model basis)
fs0_cf_power_kW = m.cf.power_kW
fs0_dryer_heat_kW = m.dr.heatDuty
fs0_dryer_blower_kW = m.dr.blowerPower

fs0_cf_energy_kWh_day = fs0_cf_power_kW * 24.0
# Per-m3 normalization uses the centrifuge's own sizing-only volumetric flow
# expression (cf.sludgeVolFlowIn), not a shared/flowsheet-level density.
fs0_cf_energy_kWh_m3 = fs0_cf_energy_kWh_day / (m.cf.sludgeVolFlowIn * 86400.0)
fs0_dryer_heat_energy_kWh_day = fs0_dryer_heat_kW * 24.0
fs0_dryer_blower_energy_kWh_day = fs0_dryer_blower_kW * 24.0

print('Electricity requirement by unit (power kW, energy kWh/day):')
print('  Centrifuge shaft:', 'kW =', _safe_value_end(fs0_cf_power_kW), ', kWh/day =', _safe_value_end(fs0_cf_energy_kWh_day))
print('  Dryer heat duty:', 'kW =', _safe_value_end(fs0_dryer_heat_kW), ', kWh/day =', _safe_value_end(fs0_dryer_heat_energy_kWh_day))
print('  Dryer blower:', 'kW =', _safe_value_end(fs0_dryer_blower_kW), ', kWh/day =', _safe_value_end(fs0_dryer_blower_energy_kWh_day))
print('  Centrifuge energy per m3 (sizing basis):', 'kWh/m3 =', _safe_value_end(fs0_cf_energy_kWh_m3))

print('========================================================================')
