#------------------------------------------------------------------------------
# function:     nanofiltration.py                                             #
# Description:  Function to define nanofiltration (NF) / ion-selective        #
#               membrane model equations.                                     #
#                                                                             #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all NF / selective-membrane model equations          #
#------------------------------------------------------------------------------

import pyomo.environ as pyo

try:
    from . import getParams
except ImportError:
    import getParams


def nf(m):
    m.nf = pyo.Block()

    # Load parameters if present in params.xlsx, otherwise use safe defaults.
    nfParams = (
        getParams.params.get('Nanofiltration')
        or getParams.params.get('nf')
        or getParams.params.get('Selective Membrane')
        or {}
    )

    # --- Membrane / hydraulic parameters ---
    m.nf.membraneCost        = pyo.Param(initialize=nfParams.get('Membrane Cost', 500.0))           # $/m2
    m.nf.membraneLp          = pyo.Param(initialize=nfParams.get('Hydraulic Permeability', 5.0))   # L/m2/h/bar
    m.nf.pumpEfficiency      = pyo.Param(initialize=nfParams.get('Pump Efficiency', 0.75))          # fraction
    m.nf.capexFactor         = pyo.Param(initialize=nfParams.get('Capex Factor', 1.0))              # dimensionless
    m.nf.maxDeltaP           = pyo.Param(initialize=nfParams.get('Max Pressure Drop', 20.0), mutable=True)  # bar
    m.nf.targetRecovery      = pyo.Param(initialize=nfParams.get('Target Recovery', 0.70), mutable=True)  # fraction

    # Membrane replacement: NF elements typically replaced every ~3 years.
    m.nf.membraneReplFrac    = pyo.Param(initialize=nfParams.get('Membrane Replacement Fraction', 1.0/3.0), mutable=True)  # fraction/yr
    m.nf.membraneReplCostFac = pyo.Param(initialize=nfParams.get('Membrane Replacement Cost Factor', 1.0))  # multiplier on membraneCost*area

    # --- Observed (constant) rejection coefficients ---
    # rejection R_i = 1 - C_permeate,i / C_feed,i  (fraction of species i retained)
    m.nf.rejectionCa         = pyo.Param(initialize=nfParams.get('Ca Rejection', 0.93), mutable=True)          # fraction
    m.nf.rejectionMg         = pyo.Param(initialize=nfParams.get('Mg Rejection', 0.88), mutable=True)          # fraction

    # Monovalent (NH4+/TAN) rejection is pH-dependent via the ionic fraction
    m.nf.rejectionNH4Intrinsic = pyo.Param(initialize=nfParams.get('NH4 Intrinsic Rejection', 0.90), mutable=True)  # fraction

    # --- Osmotic pressure model: phi-corrected van 't Hoff, pi = phi * i * C * R * T ---
    m.nf.osmoticCoeffCa  = pyo.Param(initialize=1.577, mutable=True)  # bar/(kg/m3)
    m.nf.osmoticCoeffMg  = pyo.Param(initialize=2.601, mutable=True)  # bar/(kg/m3)
    m.nf.osmoticCoeffTAN = pyo.Param(initialize=3.186, mutable=True)  # bar/(kg-N/m3)

    # --- Inputs (link to upstream permeate) ---
    m.nf.massFlowIn      = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals, bounds=(0, None))  # kg/s
    m.nf.concInTAN       = pyo.Var(initialize=1.0,  within=pyo.NonNegativeReals)  # kg-N/m3 (TAN, mostly NH4+ at NF permeate pH)
    m.nf.concInCa        = pyo.Var(initialize=0.3,  within=pyo.NonNegativeReals)  # kg/m3 (dissolved Ca2+)
    m.nf.concInMg        = pyo.Var(initialize=0.1,  within=pyo.NonNegativeReals)  # kg/m3 (dissolved Mg2+)
    m.nf.pHIn            = pyo.Var(initialize=7.0,  within=pyo.NonNegativeReals, bounds=(0, 14))  # pH in

    m.nf.pKaNH3 = pyo.Param(initialize=9.25, mutable=True)
    m.nf.alphaIn = pyo.Expression(expr=1.0 / (1.0 + 10 ** (m.nf.pKaNH3 - m.nf.pHIn)))

    # --- Outputs: permeate (to GPM, NH4+-rich) and retentate (divalent-rich reject) ---
    m.nf.permeateMassFlow  = pyo.Var(initialize=8.0, within=pyo.NonNegativeReals, bounds=(0, None))  # kg/s
    m.nf.retentateMassFlow = pyo.Var(initialize=2.0, within=pyo.NonNegativeReals, bounds=(0, None))  # kg/s
    m.nf.concPermTAN       = pyo.Var(initialize=0.9,  within=pyo.NonNegativeReals)  # kg-N/m3
    m.nf.concRetTAN        = pyo.Var(initialize=1.0,  within=pyo.NonNegativeReals)  # kg-N/m3
    m.nf.concPermCa        = pyo.Var(initialize=0.03, within=pyo.NonNegativeReals)  # kg/m3
    m.nf.concRetCa         = pyo.Var(initialize=1.0,  within=pyo.NonNegativeReals)  # kg/m3
    m.nf.concPermMg        = pyo.Var(initialize=0.01, within=pyo.NonNegativeReals)  # kg/m3
    m.nf.concRetMg         = pyo.Var(initialize=0.3,  within=pyo.NonNegativeReals)  # kg/m3
    m.nf.pHOut             = pyo.Var(initialize=7.0,  within=pyo.NonNegativeReals, bounds=(0, 14))  # pH of permeate

    # --- Design / operating variables ---
    m.nf.area    = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals, bounds=(1e-6, None))  # m2
    m.nf.deltaP  = pyo.Var(initialize=2.0,  within=pyo.NonNegativeReals, bounds=(0, None))     # bar

    # -------------------- Liquid-phase volumetric flows --------------------
    m.nf.volFlowIn         = pyo.Expression(expr=m.nf.massFlowIn / m.model().liquidDensity)          # m3/s
    m.nf.permeateVolFlow   = pyo.Expression(expr=m.nf.permeateMassFlow / m.model().liquidDensity)    # m3/s
    m.nf.retentateVolFlow  = pyo.Expression(expr=m.nf.retentateMassFlow / m.model().liquidDensity)   # m3/s

    # ------------------------------------------------------------------
    # Overall flow balance -- mass balance
    # ------------------------------------------------------------------
    def overallBalanceRule(blk):
        return blk.massFlowIn == blk.permeateMassFlow + blk.retentateMassFlow
    m.nf.overallBalance = pyo.Constraint(rule=overallBalanceRule)

    m.nf.recovery = pyo.Var(initialize=0.7, within=pyo.NonNegativeReals, bounds=(0.001, 0.999))  # mass recovery fraction

    def recoveryDefRule(blk):
        return blk.permeateMassFlow == blk.recovery * blk.massFlowIn
    m.nf.recoveryDef = pyo.Constraint(rule=recoveryDefRule)

    def recoveryTargetRule(blk):
        return blk.recovery >= blk.targetRecovery
    m.nf.recoveryTarget = pyo.Constraint(rule=recoveryTargetRule)

    # ------------------------------------------------------------------
    # Osmotic pressure: feed-side value is the AVERAGE of inlet and retentate
    # ------------------------------------------------------------------
    m.nf.piFeedIn = pyo.Expression(
        expr=m.nf.osmoticCoeffTAN * m.nf.concInTAN
           + m.nf.osmoticCoeffCa  * m.nf.concInCa
           + m.nf.osmoticCoeffMg  * m.nf.concInMg
    )
    m.nf.piRetentate = pyo.Expression(
        expr=m.nf.osmoticCoeffTAN * m.nf.concRetTAN
           + m.nf.osmoticCoeffCa  * m.nf.concRetCa
           + m.nf.osmoticCoeffMg  * m.nf.concRetMg
    )
    m.nf.piFeedAvg = pyo.Expression(expr=0.5 * (m.nf.piFeedIn + m.nf.piRetentate))
    m.nf.piPermeate = pyo.Expression(
        expr=m.nf.osmoticCoeffTAN * m.nf.concPermTAN
           + m.nf.osmoticCoeffCa  * m.nf.concPermCa
           + m.nf.osmoticCoeffMg  * m.nf.concPermMg
    )
    m.nf.deltaPi = pyo.Expression(expr=m.nf.piFeedAvg - m.nf.piPermeate)

    # Enforce a minimum net driving pressure
    m.nf.minDrivingForce = pyo.Param(initialize=0.5, mutable=True)  # bar

    def positiveFlux_rule(blk):
        return blk.deltaP >= blk.deltaPi + blk.minDrivingForce
    m.nf.positiveFluxConstr = pyo.Constraint(rule=positiveFlux_rule)

    # ------------------------------------------------------------------
    # Membrane flux: Qp = A * Lp * (dP - dPi)
    # ------------------------------------------------------------------
    def membraneFluxRule(blk):
        return blk.permeateVolFlow == blk.area * blk.membraneLp * (blk.deltaP - blk.deltaPi) * (1e-3 / 3600.0)
    m.nf.membraneFlux = pyo.Constraint(rule=membraneFluxRule)

    def pressureLimitRule(blk):
        return blk.deltaP <= blk.maxDeltaP
    m.nf.pressureLimit = pyo.Constraint(rule=pressureLimitRule)

    # ------------------------------------------------------------------
    # Species split: constant observed rejection coefficients for Ca/Mg;
    # pH-dependent (via the ionic fraction) for TAN.
    # C_perm,i = (1 - R_i) * C_feed,i
    # ------------------------------------------------------------------
    m.nf.effectiveRejectionTAN = pyo.Expression(
        expr=m.nf.rejectionNH4Intrinsic * (1.0 - m.nf.alphaIn)
    )

    def permTANRule(blk):
        return blk.concPermTAN == (1.0 - blk.effectiveRejectionTAN) * blk.concInTAN
    m.nf.permTANDef = pyo.Constraint(rule=permTANRule)

    def permCaRule(blk):
        return blk.concPermCa == (1.0 - blk.rejectionCa) * blk.concInCa
    m.nf.permCaDef = pyo.Constraint(rule=permCaRule)

    def permMgRule(blk):
        return blk.concPermMg == (1.0 - blk.rejectionMg) * blk.concInMg
    m.nf.permMgDef = pyo.Constraint(rule=permMgRule)

    # ------------------------------------------------------------------
    # Species mass balances (close retentate concentrations)
    # Q_in*C_in = Qp*Cp + Qr*Cr
    # ------------------------------------------------------------------
    def tanBalanceRule(blk):
        return (blk.volFlowIn * blk.concInTAN
                == blk.permeateVolFlow * blk.concPermTAN + blk.retentateVolFlow * blk.concRetTAN)
    m.nf.tanBalance = pyo.Constraint(rule=tanBalanceRule)

    def caBalanceRule(blk):
        return (blk.volFlowIn * blk.concInCa
                == blk.permeateVolFlow * blk.concPermCa + blk.retentateVolFlow * blk.concRetCa)
    m.nf.caBalance = pyo.Constraint(rule=caBalanceRule)

    def mgBalanceRule(blk):
        return (blk.volFlowIn * blk.concInMg
                == blk.permeateVolFlow * blk.concPermMg + blk.retentateVolFlow * blk.concRetMg)
    m.nf.mgBalance = pyo.Constraint(rule=mgBalanceRule)

    # ------------------------------------------------------------------
    # pH held constant across feed/retentate/permeate
    # ------------------------------------------------------------------
    def pHOutRule(blk):
        return blk.pHOut == blk.pHIn
    m.nf.pHBalance = pyo.Constraint(rule=pHOutRule)

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------
    m.nf.nToPermeate_kgPerS = pyo.Expression(expr=m.nf.permeateVolFlow * m.nf.concPermTAN)
    m.nf.nToRetentate_kgPerS = pyo.Expression(expr=m.nf.retentateVolFlow * m.nf.concRetTAN)
    m.nf.nRecoveryToPermeate = pyo.Expression(
        expr=m.nf.nToPermeate_kgPerS / (m.nf.volFlowIn * m.nf.concInTAN + 1e-9)
    )
    m.nf.caRejectionToRetentate = pyo.Expression(
        expr=(m.nf.retentateVolFlow * m.nf.concRetCa)
             / (m.nf.volFlowIn * m.nf.concInCa + 1e-9)
    )
    m.nf.mgRejectionToRetentate = pyo.Expression(
        expr=(m.nf.retentateVolFlow * m.nf.concRetMg)
             / (m.nf.volFlowIn * m.nf.concInMg + 1e-9)
    )
    # Combined divalent concentrations -- just for reporting
    m.nf.concInDivalent   = pyo.Expression(expr=m.nf.concInCa + m.nf.concInMg)
    m.nf.concPermDivalent = pyo.Expression(expr=m.nf.concPermCa + m.nf.concPermMg)
    m.nf.concRetDivalent  = pyo.Expression(expr=m.nf.concRetCa + m.nf.concRetMg)
    m.nf.divalentRejectionToRetentate = pyo.Expression(
        expr=(m.nf.retentateVolFlow * m.nf.concRetDivalent)
             / (m.nf.volFlowIn * m.nf.concInDivalent + 1e-9)
    )

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------
    # CAPEX: membrane cost * area * 3 covers installation, vessels, and
    # peripherals ONLY (piping, instrumentation, housing)
    m.nf.capex = pyo.Expression(expr=m.nf.capexFactor * m.nf.membraneCost * m.nf.area * 3)

    m.nf.feedFlow_m3h = pyo.Expression(expr=m.nf.volFlowIn * 3600.0)  # m3/h
    m.nf.pumpPower = pyo.Expression(
        expr=(m.nf.feedFlow_m3h * m.nf.deltaP) / (36.0 * (m.nf.pumpEfficiency + 1e-9))
    )  # kW

    m.nf.projectYears = pyo.Expression(expr=m.daysOperation / (365.0 * 24.0 * 3600.0))  # years

    # Membrane replacement OPEX: recurring cost of replacing membrane elements
    m.nf.membraneReplOpex = pyo.Expression(
        expr=m.nf.membraneReplFrac * m.nf.membraneReplCostFac * m.nf.membraneCost
             * m.nf.area * m.nf.projectYears
    )

    m.nf.pumpOpex = pyo.Expression(
        expr=m.nf.pumpPower * m.elecPrice * (m.daysOperation / 3600.0)
    )
    m.nf.opex = pyo.Expression(expr=m.nf.pumpOpex + m.nf.membraneReplOpex)