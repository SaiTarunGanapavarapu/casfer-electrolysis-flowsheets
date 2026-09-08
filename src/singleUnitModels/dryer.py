#------------------------------------------------------------------------------
# function:     dryer.py                                                      #
# Description:  Function to define dryer model equations                      #
#               Material balances, component balances                         #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all model equations                                  #
#------------------------------------------------------------------------------

import math
import pyomo.environ as pyo
try:
    from . import getParams
except ImportError:
    import getParams

_LN10 = math.log(10.0)

def dryer(m):

    # Create a block for the dryer
    m.dr = pyo.Block()

    # Load parameters for dryer from getParams
    dryerParams = getParams.params['Dryer']

    m.dr.costReference = pyo.Param(initialize = dryerParams['Cost Reference'])   # $ (Cost for a dryer of reference duty)
    m.dr.dutyReference = pyo.Param(initialize = dryerParams['Duty Reference'])      # kg-water/s
    m.dr.capexFactor   = pyo.Param(initialize = dryerParams['Capex Factor'])       # dimensionless

    # Dryer energy parameters
    m.dr.blowerEfficiency  = pyo.Param(initialize = dryerParams['Blower Efficiency'], default=0.7)  # fraction
    m.dr.latentHeat        = pyo.Param(initialize = dryerParams['Latent Heat of Vaporization'])  # kWh/kg-water
    m.dr.airDensity        = pyo.Param(initialize = dryerParams['Air Density'], default=1.225)  # kg/m3 (standard air)
    m.dr.airSpecificHeat   = pyo.Param(initialize = dryerParams['Air Specific Heat'])  # kWh/kg-K
    m.dr.heatLossFactor    = pyo.Param(initialize = dryerParams['Heat Loss Factor'], default=0.15)  # % (e.g., 10 for 10% loss)
    m.dr.ambientTemp       = pyo.Param(initialize = dryerParams['Ambient Temperature'] - 273.15, default=25.0)  # deg C
    m.dr.blowerSEC         = pyo.Param(initialize = dryerParams.get('Blower Specific Energy', 5000.0), mutable=True) # J/m3 (Pa)

    # -- Inputs (set by the connecting flowsheet via linking constraints) --
    m.dr.sludgeSolidsFlowIn_mass = pyo.Var(initialize = 0.20, within = pyo.NonNegativeReals)  # kg/s, sludge-derived organic solids IN
    m.dr.caoSolidsFlowIn_mass    = pyo.Var(initialize = 0.05, within = pyo.NonNegativeReals)  # kg/s, undissolved CaO solids IN (0 for flowsheets with no CaO stage)
    m.dr.solidsFlow_mass = pyo.Expression(expr = m.dr.sludgeSolidsFlowIn_mass + m.dr.caoSolidsFlowIn_mass)  # kg/s, total 
    m.dr.waterIn_mass       = pyo.Var(initialize = 0.75, within = pyo.NonNegativeReals)  # kg/s (water mass flow IN)

    m.dr.nitrogenMassFlowIn_kgN_s = pyo.Var(initialize = 0.01, within = pyo.NonNegativeReals)  # kg-N/s, dissolved N entering with waterIn_mass
    m.dr.pHIn = pyo.Var(initialize = 7.0, within = pyo.NonNegativeReals, bounds = (0, 14))     # liquid pH entering the dryer

    # Fixed inlet drying-air temperature 
    m.dr.airTempIn = pyo.Param(initialize=dryerParams.get('Air Inlet Temperature C', 180.0), mutable=True)  # deg C

    # Air exit temperature is now a decision variable
    m.dr.airTempOut = pyo.Var(initialize=80.0, within=pyo.NonNegativeReals, bounds=(35.0, 95.0))  # deg C

    # -- Decision/Operating Variables --
    m.dr.airFlowIn          = pyo.Var(initialize = 1.0, within = pyo.NonNegativeReals, bounds = (0, None))  # m3/s (Air flow rate)
    m.dr.finalSolids_frac   = pyo.Var(initialize = 0.9, within = pyo.NonNegativeReals, bounds = (0, 1)) # fraction

    # -- Calculated Flows --
    m.dr.productFlow_mass   = pyo.Var(initialize = 0.3, within = pyo.NonNegativeReals)  # kg/s (Total final product mass OUT)
    m.dr.waterOut_mass      = pyo.Var(initialize = 0.02, within = pyo.NonNegativeReals) # kg/s (Water mass flow Out with product)
    m.dr.waterVaporFlowOut  = pyo.Var(initialize = 0.70, within = pyo.NonNegativeReals) # kg/s (Water Evaporated)

    # -- Energy & Costs --
    m.dr.heatDuty       = pyo.Var(initialize=1000, within=pyo.NonNegativeReals) # kW (Total heat required)
    m.dr.blowerPower    = pyo.Var(initialize=50, within=pyo.NonNegativeReals)   # kW (Electrical power for blower)
    m.dr.capex          = pyo.Var(initialize=1e6, within=pyo.NonNegativeReals) # $ (Total installed cost)
    m.dr.opex           = pyo.Var(initialize=1e5, within=pyo.NonNegativeReals)  # $ (Total lifetime operating cost)

    # Reporting-only inlet TSS fraction, derived from the two mass-flow inputs.
    m.dr.sludgeTSSIn_frac = pyo.Expression(
        expr = m.dr.solidsFlow_mass / (m.dr.solidsFlow_mass + m.dr.waterIn_mass + 1e-9)
    )

    # Composition-resolved product outputs
    m.dr.sludgeSolidsMassFlowOut = pyo.Expression(expr = m.dr.sludgeSolidsFlowIn_mass)
    m.dr.caoSolidsMassFlowOut    = pyo.Expression(expr = m.dr.caoSolidsFlowIn_mass)
    # Reporting only: CaO's dilution of the final product's nutrient content.
    m.dr.caoWeightFractionInProduct = pyo.Expression(expr = m.dr.caoSolidsMassFlowOut / (m.dr.solidsFlow_mass + 1e-9))

    # Calculate Total Product Mass Flow Out
    def product_out_rule(blk):
        return blk.productFlow_mass * blk.finalSolids_frac == blk.solidsFlow_mass
    m.dr.product_out_constr = pyo.Constraint(rule=product_out_rule)

    # Calculate Water Mass Flow Out (in Product)
    def water_out_rule(blk):
        return blk.waterOut_mass == blk.productFlow_mass * (1.0 - blk.finalSolids_frac)
    m.dr.water_out_constr = pyo.Constraint(rule=water_out_rule)

    # Calculate Water Evaporated ("Duty")
    def water_evap_rule(blk):
        return blk.waterVaporFlowOut == blk.waterIn_mass - blk.waterOut_mass
    m.dr.water_evap_constr = pyo.Constraint(rule=water_evap_rule)

    # -------------------------------------------------------------------
    # Ammonia volatilization submodel
    # -------------------------------------------------------------------
    m.dr.airTempOut_K = pyo.Expression(expr = m.dr.airTempOut + 273.15)

    # Temperature-dependent pKa for the NH3/NH4+ equilibrium, using the van't Hoff relation.
    m.dr.pKaNH3_A = pyo.Param(initialize=0.09018, mutable=True)
    m.dr.pKaNH3_B = pyo.Param(initialize=2729.92, mutable=True)  # K
    m.dr.pKaNH3_T = pyo.Expression(expr = m.dr.pKaNH3_A + m.dr.pKaNH3_B / m.dr.airTempOut_K)
    m.dr.molarMassN_kgPerMol   = pyo.Param(initialize=0.014007, mutable=True)
    m.dr.molarMassNH3_kgPerMol = pyo.Param(initialize=0.017031, mutable=True)
    m.dr.molarMassAir_kgPerMol   = pyo.Param(initialize=0.028970, mutable=True)
    m.dr.molarMassWater_kgPerMol = pyo.Param(initialize=0.018015, mutable=True)

    # Henry's constant (concentration-basis, mol/L/atm) at a reference temp,
    m.dr.henryNH3Ref_molPerLAtm = pyo.Param(initialize=58.5, mutable=True)
    m.dr.henryNH3RefTemp_K = pyo.Param(initialize=298.15, mutable=True)
    # Heat of solution of NH3(g) -> NH3(aq)
    m.dr.enthalpySolutionNH3_JPerMol = pyo.Param(initialize=-34200.0, mutable=True)
    m.dr.gasConstant_JPerMolK = pyo.Param(initialize=8.314, mutable=True)
    m.dr.pressureTotal_atm = pyo.Param(initialize=1.0, mutable=True)
    # Latent heat of NH3 desorption from solution
    m.dr.latentHeatNH3_kWhPerKg = pyo.Param(initialize=0.381, mutable=True)

    # van't Hoff temperature correction: H decreases (more volatile) as T rises.
    m.dr.henryNH3_T = pyo.Expression(
        expr = m.dr.henryNH3Ref_molPerLAtm * pyo.exp(
            (-m.dr.enthalpySolutionNH3_JPerMol / m.dr.gasConstant_JPerMolK)
            * (1.0/m.dr.airTempOut_K - 1.0/m.dr.henryNH3RefTemp_K)
        )
    )

    # Free-ammonia fraction of dissolved TAN, now using the T-dependent pKa
    m.dr.freeAmmoniaFrac = pyo.Expression(
        expr = 1.0 / (1.0 + 10**(m.dr.pKaNH3_T - m.dr.pHIn))
    )
    m.dr.nitrogenMassFlowAvail_kgN_s = pyo.Expression(
        expr = m.dr.freeAmmoniaFrac * m.dr.nitrogenMassFlowIn_kgN_s
    )

    # Total gas-phase molar flow leaving the stage (dry air + water vapor).
    m.dr.airMolarFlow_molPerS = pyo.Expression(expr = (m.dr.airFlowIn * m.dr.airDensity) / m.dr.molarMassAir_kgPerMol)
    m.dr.waterVaporMolarFlow_molPerS = pyo.Expression(expr = m.dr.waterVaporFlowOut / m.dr.molarMassWater_kgPerMol)
    m.dr.gasMolarFlowTotal_molPerS = pyo.Expression(expr = m.dr.airMolarFlow_molPerS + m.dr.waterVaporMolarFlow_molPerS)

    # Liquid volumetric flow entering the stage, via the dedicated liquid-phase density.
    m.dr.liquidVolFlowIn_Lps = pyo.Expression(expr = (m.dr.waterIn_mass / m.model().liquidDensity) * 1000.0)

    m.dr.strippingFactor = pyo.Expression(
        expr = m.dr.gasMolarFlowTotal_molPerS
             / (m.dr.henryNH3_T * m.dr.pressureTotal_atm * m.dr.liquidVolFlowIn_Lps + 1e-9)
    )
    m.dr.strippingFraction = pyo.Expression(expr = m.dr.strippingFactor / (1.0 + m.dr.strippingFactor))

    m.dr.nitrogenMassFlowVapor_kgN_s = pyo.Expression(
        expr = m.dr.strippingFraction * m.dr.nitrogenMassFlowAvail_kgN_s
    )
    m.dr.nitrogenMassFlowOut_kgN_s = pyo.Expression(
        expr = m.dr.nitrogenMassFlowIn_kgN_s - m.dr.nitrogenMassFlowVapor_kgN_s
    )
    m.dr.ammoniaMassFlowVapor_kgPerS = pyo.Expression(
        expr = m.dr.nitrogenMassFlowVapor_kgN_s * (m.dr.molarMassNH3_kgPerMol / m.dr.molarMassN_kgPerMol)
    )

    # -------------------------------------------------------------------
    # Psychrometric saturation constraint (Antoine equation for water, valid ~1-100 C).
    # -------------------------------------------------------------------
    m.dr.antoineA = pyo.Param(initialize=8.07131, mutable=True)
    m.dr.antoineB = pyo.Param(initialize=1730.63, mutable=True)
    m.dr.antoineC = pyo.Param(initialize=233.426, mutable=True)
    # Safety margin below saturation to avoid condensation risk in ducts.
    m.dr.humiditySafetyFactor = pyo.Param(initialize=0.80, mutable=True)  # fraction of Ysat
    # Ambient makeup-air humidity ratio
    m.dr.ambientHumidityRatio = pyo.Param(initialize=0.010, mutable=True)  # kg-water/kg-dry-air

    m.dr.satVaporPressure_mmHg = pyo.Expression(
        expr = pyo.exp(_LN10 * (m.dr.antoineA - m.dr.antoineB/(m.dr.antoineC + m.dr.airTempOut)))
    )
    m.dr.satVaporPressure_atm = pyo.Expression(expr = m.dr.satVaporPressure_mmHg / 760.0)
    m.dr.humidityRatioSat = pyo.Expression(
        expr = 0.622 * m.dr.satVaporPressure_atm / (m.dr.pressureTotal_atm - m.dr.satVaporPressure_atm + 1e-9)
    )

    m.dr.airMassFlowDry_kgPerS = pyo.Expression(expr = m.dr.airFlowIn * m.dr.airDensity)
    # Water plus ammonia mass added to the gas phase, both must be
    # carried by the exiting air without exceeding the saturation limit.
    m.dr.totalVolatileMassOut_kgPerS = pyo.Expression(
        expr = m.dr.waterVaporFlowOut + m.dr.ammoniaMassFlowVapor_kgPerS
    )

    def humiditySaturationRule(blk):
        return (blk.ambientHumidityRatio * blk.airMassFlowDry_kgPerS + blk.totalVolatileMassOut_kgPerS
                <= blk.humiditySafetyFactor * blk.humidityRatioSat * blk.airMassFlowDry_kgPerS)
    m.dr.humiditySaturationConstr = pyo.Constraint(rule=humiditySaturationRule)

    # Capital Cost (Capex) - Scales with Evaporation Duty
    def capex_rule(blk):
        denominator = blk.dutyReference + 1e-9
        return blk.capex == blk.costReference * (blk.waterVaporFlowOut / denominator)**blk.capexFactor
    m.dr.capex_constr = pyo.Constraint(rule=capex_rule)

    # Heat Duty Rule: heat added to raise makeup air from ambient to airTempIn.
    def heat_duty_rule(blk):
        sensible_kW = blk.airSpecificHeat * blk.airDensity * blk.airFlowIn * \
                      (blk.airTempIn - blk.ambientTemp) * 3600.0
        denominator = (1.0 - blk.heatLossFactor) + 1e-9
        return blk.heatDuty == sensible_kW / denominator
    m.dr.heat_duty_constr = pyo.Constraint(rule=heat_duty_rule)

    # Energy balance: sensible heat released by air cooling from airTempIn to
    # airTempOut must supply the latent heat of the evaporated water AND the stripped ammonia.
    def air_energy_balance_rule(blk):
        energy_supplied_by_air_kW = blk.airSpecificHeat * blk.airDensity * blk.airFlowIn * \
                                  (blk.airTempIn - blk.airTempOut) * 3600.0
        latent_kW = (blk.latentHeat * blk.waterVaporFlowOut
                     + blk.latentHeatNH3_kWhPerKg * blk.ammoniaMassFlowVapor_kgPerS) * 3600.0
        return energy_supplied_by_air_kW == latent_kW
    m.dr.air_energy_balance_constr = pyo.Constraint(rule=air_energy_balance_rule)

    # Blower Power (kW)
    def blower_power_rule(blk):
        power_watts = (blk.airFlowIn * blk.blowerSEC) 
        return blk.blowerPower == power_watts / 1000.0
    m.dr.blower_power_constr = pyo.Constraint(rule=blower_power_rule)

    # Operating Cost
    def opex_rule(blk):
        operating_hours = m.model().daysOperation / 3600.0 + 1e-9
        total_power_kW = blk.blowerPower + blk.heatDuty
        return blk.opex == total_power_kW * operating_hours * m.model().elecPrice
    m.dr.opex_constr = pyo.Constraint(rule=opex_rule)