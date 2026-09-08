#-------------------------------------------------------------------------------
# function:     gpm.py                                                         #
# Description:  GPM model with recirculation loop and pressure-drop-based pump #
#               cost. Recirculation increases loop hydraulics and effective    #
#               treatment passes so the optimizer can trade membrane area vs   #
#               recirculation power.                                           #
#                                                                              #
#                                                                              #
# Input:        - m : Pyomo concrete model                                     #
#                                                                              #
# Output:       - m with all NF / selective-membrane model equations           #
#-------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
except ImportError:
    import getParams

def gpm(m):
    m.gpm = pyo.Block()

    gpmParams = getParams.params['Gas Permeable Membrane']

    # Core parameters
    m.gpm.pKa = pyo.Param(initialize=gpmParams['pKa'], within=pyo.Any)
    m.gpm.acidCost = pyo.Param(initialize=gpmParams['Acid Cost'] / 2.0)  # $/kg solution
    m.gpm.inletDensity = pyo.Param(initialize=gpmParams['Inlet Density'])
    m.gpm.inletViscosity = pyo.Param(initialize=gpmParams['Inlet Viscosity'])
    m.gpm.inletDiffusivity = pyo.Param(initialize=gpmParams['Inlet Diffusivity'])
    m.gpm.hydDiameter = pyo.Param(initialize=0.001)
    m.gpm.moduleLength = pyo.Param(initialize=0.5)

    m.gpm.modulesInSeries = pyo.Param(initialize=20.0, mutable=True)
    m.gpm.effLength = pyo.Expression(expr=m.gpm.moduleLength * m.gpm.modulesInSeries)

    m.gpm.acidDensity = pyo.Param(initialize=gpmParams['Acid Density'])
    m.gpm.costReference = pyo.Param(initialize=125*130)
    m.gpm.areaReference = pyo.Param(initialize=130)
    m.gpm.capexFactor = pyo.Param(initialize=gpmParams['Capex Factor'])
    m.gpm.pumpEff = pyo.Param(initialize=gpmParams['Pump Efficiency'])
    m.gpm.costExponent = pyo.Param(initialize=0.7)

    # Recirculation and hydraulic loss assumptions
    m.gpm.recircEffect = pyo.Param(initialize=1.0, mutable=True)
    m.gpm.minorK = pyo.Param(initialize=3.0, mutable=True)
    m.gpm.staticDP = pyo.Param(initialize=10000.0, mutable=True)   # Pa

    # Molecular weights (kg/mol)
    m.gpm.molwtN = pyo.Param(initialize=0.01401)
    m.gpm.molwtCaO = pyo.Param(initialize=0.05608)
    m.gpm.molwtH2SO4 = pyo.Param(initialize=0.09808, mutable=True)
    m.gpm.molwtAmSulfate = pyo.Param(initialize=0.13214, mutable=True)  # (NH4)2SO4
    m.gpm.targetNwtPercent = pyo.Param(initialize=12.17, mutable=True)
    m.gpm.productDensity = pyo.Param(
        initialize=gpmParams.get('Product Density', pyo.value(m.gpm.acidDensity)),
        mutable=True,
    )  # kg/m3, used to estimate product molarity from wt%

    # -------------------- Primary state variables (mass basis) --------------------
    m.gpm.massFlowIn  = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)   # kg/s
    m.gpm.concInFresh = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)   # kg-N/m3 (fresh feed from NF permeate)
    m.gpm.concIn      = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)   # kg-N/m3 (mixed with recirculation)
    m.gpm.inletpH     = pyo.Var(initialize=7.0, within=pyo.NonNegativeReals, bounds=(1.0, 13.0))
    m.gpm.massFlowOut = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)   # kg/s
    m.gpm.concOut     = pyo.Var(initialize=0.5, within=pyo.NonNegativeReals)   # kg-N/m3
    m.gpm.N_removed   = pyo.Var(initialize=0.1, within=pyo.NonNegativeReals)   # kg-N/s

    # Sizing-only volumetric conversions
    m.gpm.volFlowIn  = pyo.Expression(expr=m.gpm.massFlowIn / m.model().liquidDensity)   # m3/s
    m.gpm.volFlowOut = pyo.Expression(expr=m.gpm.massFlowOut / m.model().liquidDensity)  # m3/s

    # Design and loop variables
    m.gpm.area = pyo.Var(initialize=100.0, bounds=(5.0, 20000.0))              # m2 installed membrane area
    m.gpm.recircRatio = pyo.Var(initialize=1.0, bounds=(0.0, 10.0))           # Qrecirc / Qfeed
    m.gpm.velocity = pyo.Var(initialize=0.2, bounds=(0.02, 2.0))              # m/s in membrane channels

    # Mass transfer
    m.gpm.Re = pyo.Var(initialize=1000.0, within=pyo.NonNegativeReals)
    m.gpm.Sc = pyo.Var(initialize=500.0, within=pyo.NonNegativeReals)
    m.gpm.Sh = pyo.Var(initialize=100.0, within=pyo.NonNegativeReals)
    m.gpm.kL = pyo.Var(initialize=1e-5, within=pyo.NonNegativeReals)
    m.gpm.kO = pyo.Var(initialize=1e-5, within=pyo.NonNegativeReals)

    # Acid side
    m.gpm.w_acid_in = pyo.Var(initialize=0.5, bounds=(0.1, 0.5), within=pyo.NonNegativeReals)
    m.gpm.acidFlowIn = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)  # m3/s

    # Optional base addition disabled by default
    m.gpm.baseFlow = pyo.Var(initialize=0.0, within=pyo.NonNegativeReals)
    m.gpm.baseFlow.fix(0.0)

    # pH chemistry state (raffinate/outlet)
    m.gpm.H_out = pyo.Var(initialize=1e-7, bounds=(1e-12, 1e2), within=pyo.NonNegativeReals)
    m.gpm.OH_out = pyo.Var(initialize=1e-7, bounds=(1e-12, 1e5), within=pyo.NonNegativeReals)
    m.gpm.NH4_out = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals)
    m.gpm.NH3_out = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals)
    m.gpm.Z_net_out = pyo.Var(initialize=0.0, bounds=(-1e6, 1e6), within=pyo.NonNegativeReals)
    m.gpm.pH_out = pyo.Var(initialize=10.0, bounds=(8.0, 13.0), within=pyo.NonNegativeReals)

    # Inlet chemistry expressions
    m.gpm.H_in = pyo.Expression(expr=10 ** (3 - m.gpm.inletpH))
    m.gpm.OH_in = pyo.Expression(expr=1e-8 / (m.gpm.H_in + 1e-12))
    m.gpm.TAN_in_molar = pyo.Expression(expr=m.gpm.concIn / m.gpm.molwtN)
    m.gpm.alpha_in = pyo.Expression(expr=1.0 / (1.0 + 10 ** (m.gpm.pKa - m.gpm.inletpH)))
    m.gpm.NH4_in = pyo.Expression(expr=m.gpm.TAN_in_molar * (1.0 - m.gpm.alpha_in))
    m.gpm.Z_net_in = pyo.Expression(expr=m.gpm.OH_in - m.gpm.H_in - m.gpm.NH4_in)

    # Recirculation helpers
    m.gpm.loopFlow = pyo.Expression(expr=m.gpm.volFlowIn * (1.0 + m.gpm.recircRatio))

    # Recirculation inlet dilution -- a volumetric dilution balance
    def inlet_conc_recirculation_rule(blk):
        return (blk.concIn * blk.loopFlow ==
                blk.concInFresh * blk.volFlowIn + blk.concOut * blk.recircRatio * blk.volFlowIn)
    m.gpm.inlet_conc_recirculation = pyo.Constraint(rule=inlet_conc_recirculation_rule)

    # Geometry and hydraulics
    m.gpm.crossArea = pyo.Expression(expr=(m.gpm.hydDiameter * m.gpm.area) / (4.0 * m.gpm.effLength))

    def velocity_rule(blk):
        return blk.velocity == blk.loopFlow / (blk.crossArea + 1e-12)
    m.gpm.velocityConstr = pyo.Constraint(rule=velocity_rule)

    def re_rule(blk):
        return blk.Re == (blk.velocity * blk.inletDensity * blk.hydDiameter) / (blk.inletViscosity + 1e-12)
    m.gpm.ReConstr = pyo.Constraint(rule=re_rule)

    def sc_rule(blk):
        return blk.Sc == blk.inletViscosity / (blk.inletDensity * blk.inletDiffusivity + 1e-18)
    m.gpm.ScConstr = pyo.Constraint(rule=sc_rule)

    def sh_rule(blk):
        return blk.Sh == 1.62 * (blk.Re * blk.Sc * blk.hydDiameter/blk.moduleLength) ** (1/3)
    m.gpm.ShConstr = pyo.Constraint(rule=sh_rule)

    def kl_rule(blk):
        return blk.kL == (blk.Sh * blk.inletDiffusivity) / (blk.hydDiameter + 1e-12)
    m.gpm.kLConstr = pyo.Constraint(rule=kl_rule)

    # Membrane physical parameters
    m.gpm.porosity = pyo.Param(initialize=0.7)
    m.gpm.tortuosity = pyo.Param(initialize=2.5)
    m.gpm.thickness = pyo.Param(initialize=50e-6)  # 50 microns
    m.gpm.gasDiffusivity = pyo.Param(initialize=2e-5)  # m2/s for NH3 in air

    m.gpm.km = pyo.Expression(expr=(m.gpm.porosity * m.gpm.gasDiffusivity) / (m.gpm.tortuosity * m.gpm.thickness))

    def overall_mass_transfer_rule(blk):
        return blk.kO == 1.0 / ((1.0 / (blk.kL + 1e-12)) + (1.0 / (blk.km + 1e-12)))
    m.gpm.kOverallConstr = pyo.Constraint(rule=overall_mass_transfer_rule)

    # Outlet chemistry and alpha
    def calc_z_out(blk):
        base_charge_conc = 2.0 * (blk.baseFlow / blk.molwtCaO) / (blk.volFlowIn + 1e-9)
        return blk.Z_net_out == blk.Z_net_in + base_charge_conc
    m.gpm.z_out_constr = pyo.Constraint(rule=calc_z_out)

    m.gpm.water_eq = pyo.Constraint(expr=m.gpm.H_out * m.gpm.OH_out == 1e-8)
    m.gpm.ammonia_eq = pyo.Constraint(expr=(10 ** (-m.gpm.pKa) * 1000.0) * m.gpm.NH4_out == m.gpm.H_out * m.gpm.NH3_out)
    m.gpm.tan_out_def = pyo.Constraint(expr=m.gpm.concOut / m.gpm.molwtN == m.gpm.NH3_out + m.gpm.NH4_out)
    m.gpm.charge_balance = pyo.Constraint(expr=m.gpm.Z_net_out + m.gpm.H_out + m.gpm.NH4_out == m.gpm.OH_out)
    m.gpm.ph_def = pyo.Constraint(expr=10 ** (3 - m.gpm.pH_out) == m.gpm.H_out)

    m.gpm.alpha_out = pyo.Expression(expr=m.gpm.NH3_out / (m.gpm.NH3_out + m.gpm.NH4_out + 1e-9))
    m.gpm.alpha_avg = pyo.Expression(expr=0.5 * (m.gpm.alpha_in + m.gpm.alpha_out))

    # Mass transfer performance with effective passes from recirculation
    def performance_rule(blk):
        exponent = -1.0 * blk.kO * blk.area * blk.alpha_avg / (blk.loopFlow + 1e-9)
        return blk.concOut == blk.concIn * pyo.exp(exponent)
    m.gpm.perf_constr = pyo.Constraint(rule=performance_rule)

    # Material/component balances -- mass balance
    m.gpm.materialBalance = pyo.Constraint(expr=m.gpm.massFlowIn == m.gpm.massFlowOut)
    # N_removed is a volumetric-concentration-based removal rate 
    m.gpm.N_rem_constr = pyo.Constraint(
        expr=m.gpm.N_removed == m.gpm.volFlowIn * (m.gpm.concInFresh - m.gpm.concOut))

    # Acid requirement from stoichiometric capture (H2SO4 equivalent basis)
    n_per_h2so4 = 2.0 * 14.01 / 98.08
    m.gpm.acidMassFlow = pyo.Expression(expr=m.gpm.acidFlowIn * m.gpm.acidDensity)
    m.gpm.H2SO4MassFlow = pyo.Expression(expr=m.gpm.acidMassFlow * m.gpm.w_acid_in)
    m.gpm.acidNCapa = pyo.Expression(expr=m.gpm.H2SO4MassFlow * n_per_h2so4)
    m.gpm.acid_stoich_eq = pyo.Constraint(expr=m.gpm.acidNCapa == m.gpm.N_removed)

    # -------------------------------------------------------------------
    # Water-vapor osmotic flux
    # -------------------------------------------------------------------
    m.gpm.gasConstR = pyo.Param(initialize=8.314, mutable=True)          # J/(mol.K)
    m.gpm.operatingTempK = pyo.Param(initialize=298.15, mutable=True)    # K
    m.gpm.waterPermeability = pyo.Param(initialize=1e-15, mutable=True)  # m3/(m2.s.Pa)
    m.gpm.osmoticCoefficient = pyo.Param(initialize=0.7, mutable=True)   # non-ideality correction

    # Feed-side total solute proxy: TAN only 
    m.gpm.feedTotalSoluteConc = pyo.Expression(expr=m.gpm.TAN_in_molar)  # mol/m3

    m.gpm.drawTotalNConc_molPerL = pyo.Expression(
        expr=(m.gpm.N_removed / m.gpm.molwtN) / (m.gpm.acidFlowIn * 1000.0 + 1e-9)
    )
    m.gpm.drawTotalSConc_molPerL = pyo.Expression(
        expr=(m.gpm.H2SO4MassFlow / m.gpm.molwtH2SO4) / (m.gpm.acidFlowIn * 1000.0 + 1e-9)
    )
    m.gpm.drawTotalSoluteConc = pyo.Expression(
        expr=(m.gpm.drawTotalNConc_molPerL + m.gpm.drawTotalSConc_molPerL) * 1000.0
    )  # mol/m3

    m.gpm.osmoticPressureFeed = pyo.Expression(
        expr=m.gpm.osmoticCoefficient * m.gpm.feedTotalSoluteConc * m.gpm.gasConstR * m.gpm.operatingTempK
    )  # Pa
    m.gpm.osmoticPressureDraw = pyo.Expression(
        expr=m.gpm.osmoticCoefficient * m.gpm.drawTotalSoluteConc * m.gpm.gasConstR * m.gpm.operatingTempK
    )  # Pa

    m.gpm.jWaterVapor = pyo.Expression(
        expr=m.gpm.waterPermeability * m.gpm.area * (m.gpm.osmoticPressureDraw - m.gpm.osmoticPressureFeed)
    )  # m3/s, positive = feed -> draw
    m.gpm.jWaterVaporMassFlow = pyo.Expression(expr=m.gpm.jWaterVapor * 1000.0)  # kg/s

    # Product-side bookkeeping
    m.gpm.saltMassFlow = pyo.Expression(expr=m.gpm.N_removed * (132.14 / 28.02))
    m.gpm.H2SO4Left = pyo.Expression(expr=m.gpm.H2SO4MassFlow - (m.gpm.N_removed / n_per_h2so4))

    # -------------------------------------------------------------------
    # Ammonium-sulfate solubility ceiling 
    # -------------------------------------------------------------------
    m.gpm.productMassFlowRaw = pyo.Expression(
        expr=(m.gpm.acidMassFlow * (1.0 - m.gpm.w_acid_in)) + m.gpm.H2SO4Left + m.gpm.saltMassFlow
             + m.gpm.jWaterVaporMassFlow
    )
    m.gpm.AS_solubility_g_per100gWater = pyo.Param(initialize=76.4, mutable=True)  # g AS/100g water, ~room temp
    m.gpm.wAS_max = pyo.Expression(
        expr=m.gpm.AS_solubility_g_per100gWater / (100.0 + m.gpm.AS_solubility_g_per100gWater)
    )
    m.gpm.wN_max = pyo.Expression(expr=m.gpm.wAS_max * (2.0 * m.gpm.molwtN / m.gpm.molwtAmSulfate))
    m.gpm.productMassFlowMinForSolubility = pyo.Expression(expr=m.gpm.N_removed / (m.gpm.wN_max + 1e-12))
    m.gpm.dilutionSmoothingEps = pyo.Param(initialize=1e-6, mutable=True)  # kg/s

    m.gpm.productMassFlow = pyo.Expression(
        expr=0.5 * (m.gpm.productMassFlowRaw + m.gpm.productMassFlowMinForSolubility)
             + 0.5 * pyo.sqrt(
                 (m.gpm.productMassFlowRaw - m.gpm.productMassFlowMinForSolubility) ** 2
                 + m.gpm.dilutionSmoothingEps ** 2
             )
    )
    m.gpm.extraDilutionWaterNeeded = pyo.Expression(expr=m.gpm.productMassFlow - m.gpm.productMassFlowRaw)
    m.gpm.N_wt_percent = pyo.Expression(expr=100.0 * m.gpm.N_removed / (m.gpm.productMassFlow + 1e-9))

    # Product pH estimate from NH4+ weak-acid equilibrium
    m.gpm.n_wt_frac = pyo.Expression(expr=m.gpm.N_wt_percent / 100.0)
    m.gpm.nh4_conc_mol_L = pyo.Expression(
        expr=(m.gpm.n_wt_frac * m.gpm.productDensity) / (m.gpm.molwtN * 1000.0)
    )
    m.gpm.ka_nh4 = pyo.Expression(expr=10 ** (-m.gpm.pKa))
    m.gpm.h_prod_mol_L = pyo.Expression(
        expr=0.5
        * (
            -m.gpm.ka_nh4
            + pyo.sqrt(m.gpm.ka_nh4**2 + 4.0 * m.gpm.ka_nh4 * m.gpm.nh4_conc_mol_L)
        )
    )
    m.gpm.product_pH_est = pyo.Expression(expr=-pyo.log(m.gpm.h_prod_mol_L + 1e-16) / pyo.log(10.0))

    m.gpm.h_outlet_mol_L = pyo.Expression(
        expr=0.5 * (-m.gpm.ka_nh4 + pyo.sqrt(m.gpm.ka_nh4**2 + 4.0 * m.gpm.ka_nh4 * m.gpm.NH4_out))
    )
    m.gpm.outlet_pH_computed = pyo.Expression(expr=-pyo.log(m.gpm.h_outlet_mol_L + 1e-16) / pyo.log(10.0))

    m.gpm.target_n_wt_frac = pyo.Expression(expr=m.gpm.targetNwtPercent / 100.0)
    m.gpm.target_nh4_conc_mol_L = pyo.Expression(
        expr=(m.gpm.target_n_wt_frac * m.gpm.productDensity) / (m.gpm.molwtN * 1000.0)
    )
    m.gpm.h_prod_target_mol_L = pyo.Expression(
        expr=0.5
        * (
            -m.gpm.ka_nh4
            + pyo.sqrt(m.gpm.ka_nh4**2 + 4.0 * m.gpm.ka_nh4 * m.gpm.target_nh4_conc_mol_L)
        )
    )
    m.gpm.product_pH_at_targetN = pyo.Expression(
        expr=-pyo.log(m.gpm.h_prod_target_mol_L + 1e-16) / pyo.log(10.0)
    )

    # Realistic pump model on total loop flow
    m.gpm.f_lam = pyo.Expression(expr=64.0 / (m.gpm.Re + 1e-9))
    m.gpm.f_turb = pyo.Expression(expr=0.3164 * (m.gpm.Re + 1e-9) ** (-0.25))
    m.gpm.f_blend = pyo.Expression(expr=1.0 / (1.0 + pyo.exp(-(m.gpm.Re - 3000.0) / 400.0)))
    m.gpm.fricFactor = pyo.Expression(expr=(1.0 - m.gpm.f_blend) * m.gpm.f_lam + m.gpm.f_blend * m.gpm.f_turb)

    m.gpm.dynamicHead = pyo.Expression(expr=(m.gpm.inletDensity * m.gpm.velocity ** 2) / 2.0)

    m.gpm.deltaP = pyo.Expression(
        expr=(m.gpm.fricFactor * (m.gpm.effLength / (m.gpm.hydDiameter + 1e-12)) + (m.gpm.minorK * m.gpm.modulesInSeries)) * m.gpm.dynamicHead
        + m.gpm.staticDP
    )

    m.gpm.pumpPower = pyo.Expression(expr=(m.gpm.loopFlow * m.gpm.deltaP) / (m.gpm.pumpEff * 1000.0))  # kW

    # Economics over lifetime
    m.gpm.capex = m.gpm.capexFactor * m.gpm.costReference * (m.gpm.area / m.gpm.areaReference)**m.gpm.costExponent * 3
    m.gpm.totalAcidCost = m.gpm.acidMassFlow * m.gpm.acidCost * m.daysOperation
    m.gpm.pumpOpex = m.gpm.pumpPower * m.elecPrice * (m.daysOperation / 3600.0)
    # opex is the draw-side acid cost + pump power.
    m.gpm.opex = m.gpm.totalAcidCost + m.gpm.pumpOpex