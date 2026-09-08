#------------------------------------------------------------------------------
# function:     goMembraneDewatering.py                                       #
# Description:  GO membrane dewatering unit                                   #
#                                                                             #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all GO-membrane dewatering model equations           #
#------------------------------------------------------------------------------

import pyomo.environ as pyo

try:
	from . import getParams
except ImportError:
	import getParams



def goMembraneDewatering(m):
	m.go = pyo.Block()

	# Load parameters if present in params.xlsx, otherwise use safe defaults.
	go_params = (
		getParams.params.get('GO Membrane Dewatering')
		or {}
	)

	m.go.membraneCost = pyo.Param(initialize=go_params.get('Membrane Cost', 500.0))  # $/m2
	m.go.membraneLp = pyo.Param(initialize=go_params.get('Hydraulic Permeability', 50.0))  # L/m2/h/bar
	m.go.pumpEfficiency = pyo.Param(initialize=go_params.get('Pump Efficiency', 0.75))  # fraction
	m.go.capexFactor = pyo.Param(initialize=go_params.get('Capex Factor', 1.0))  # dimensionless
	m.go.targetSolids = pyo.Param(initialize=go_params.get('Target Solids', 0.25), mutable=True)  # wt frac
	m.go.maxDeltaP = pyo.Param(initialize=go_params.get('Max Pressure Drop', 30.0), mutable=True)  # bar
	m.go.minDrivingForce = pyo.Param(initialize=0.5, mutable=True)  # bar

	# --- Osmotic pressure coefficients 
	m.go.osmoticCoeffCa  = pyo.Param(initialize=1.577, mutable=True)  # bar/(kg/m3)
	m.go.osmoticCoeffMg  = pyo.Param(initialize=2.601, mutable=True)  # bar/(kg/m3)
	m.go.osmoticCoeffTAN = pyo.Param(initialize=3.186, mutable=True)  # bar/(kg-N/m3)

	# Dissolved-species rejection
	m.go.rejectionTAN = pyo.Param(initialize=0.05, mutable=True)  # fraction
	m.go.rejectionCa  = pyo.Param(initialize=0.05, mutable=True)  # fraction
	m.go.rejectionMg  = pyo.Param(initialize=0.05, mutable=True)  # fraction

	# -------------------- Primary state variables (mass basis) --------------------
	m.go.sludgeMassFlowIn  = pyo.Var(initialize=10.0, within=pyo.NonNegativeReals, bounds=(0, None))  # kg/s
	m.go.sludgeTSSin       = pyo.Var(initialize=0.08, within=pyo.NonNegativeReals, bounds=(0.0, 1.0))  # mass fraction
	m.go.sludgeMassFlowOut = pyo.Var(initialize=5.0,  within=pyo.NonNegativeReals, bounds=(0, None))  # kg/s (retentate)
	m.go.sludgeTSSout      = pyo.Var(initialize=0.20, within=pyo.NonNegativeReals, bounds=(0.0, 1.0))  # mass fraction
	m.go.permeateMassFlow  = pyo.Var(initialize=5.0,  within=pyo.NonNegativeReals, bounds=(0, None))  # kg/s
	m.go.sludgepHIn        = pyo.Var(initialize=7.0,  within=pyo.NonNegativeReals, bounds=(0, 14))  # pH of sludge in
	m.go.liquidpHOut       = pyo.Var(initialize=7.0,  within=pyo.NonNegativeReals, bounds=(0, 14))  # pH of permeate out

	# Dissolved-species concentrations 
	m.go.concInTAN  = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)  # kg-N/m3
	m.go.concInCa   = pyo.Var(initialize=0.3, within=pyo.NonNegativeReals)  # kg/m3
	m.go.concInMg   = pyo.Var(initialize=0.1, within=pyo.NonNegativeReals)  # kg/m3
	m.go.concRetTAN = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals)
	m.go.concRetCa  = pyo.Var(initialize=0.3, within=pyo.NonNegativeReals)
	m.go.concRetMg  = pyo.Var(initialize=0.1, within=pyo.NonNegativeReals)
	m.go.concPermTAN = pyo.Var(initialize=0.95, within=pyo.NonNegativeReals)
	m.go.concPermCa  = pyo.Var(initialize=0.29, within=pyo.NonNegativeReals)
	m.go.concPermMg  = pyo.Var(initialize=0.10, within=pyo.NonNegativeReals)

	# Design/operating variables
	m.go.area = pyo.Var(initialize=100.0, within=pyo.NonNegativeReals, bounds=(1e-6, None))  # m2
	m.go.deltaP = pyo.Var(initialize=2.0, within=pyo.NonNegativeReals, bounds=(0, None))  # bar

	# -------------------- mass balance  --------------------
	def material_balance_rule(blk):
		return blk.sludgeMassFlowIn == blk.sludgeMassFlowOut + blk.permeateMassFlow
	m.go.materialBalance = pyo.Constraint(rule=material_balance_rule)

	# Solids balance: mass-based directly, permeate assumed solids-free 
	def solids_balance_rule(blk):
		return blk.sludgeMassFlowIn * blk.sludgeTSSin == blk.sludgeMassFlowOut * blk.sludgeTSSout
	m.go.solidsBalance = pyo.Constraint(rule=solids_balance_rule)

	def target_solids_rule(blk):
		return blk.sludgeTSSout == blk.targetSolids
	m.go.targetSolidsConstraint = pyo.Constraint(rule=target_solids_rule)

	# -------------------- Liquid-phase volumetric flows --------------------
	# Dedicated liquid-phase density, used ONLY to convert liquid MASS flows
	# to volumetric flows for (a) the species concentration balances below
	# and (b) the membrane's hydraulic flux equation 
	m.go.liquidMassFlowIn  = pyo.Expression(expr=m.go.sludgeMassFlowIn * (1.0 - m.go.sludgeTSSin))
	m.go.liquidMassFlowOut = pyo.Expression(expr=m.go.sludgeMassFlowOut * (1.0 - m.go.sludgeTSSout))
	m.go.liquidVolFlowIn   = pyo.Expression(expr=m.go.liquidMassFlowIn / m.model().liquidDensity)   # m3/s
	m.go.liquidVolFlowOut  = pyo.Expression(expr=m.go.liquidMassFlowOut / m.model().liquidDensity)  # m3/s (retentate liquid)
	m.go.permeateVolFlow   = pyo.Expression(expr=m.go.permeateMassFlow / m.model().liquidDensity)   # m3/s (permeate ~ dilute liquid)

	# Bulk (whole-stream, solids+liquid) volumetric feed flow -- for pump
	# sizing ONLY, via the fixed standard bulk density
	m.go.feedVolFlowBulk = pyo.Expression(expr=m.go.sludgeMassFlowIn / m.model().sludgeDensity)  # m3/s

	# -------------------- Species balances ---------------------
	def tanBalanceRule(blk):
		return (blk.concInTAN * blk.liquidVolFlowIn
				== blk.concRetTAN * blk.liquidVolFlowOut + blk.concPermTAN * blk.permeateVolFlow)
	m.go.tanBalance = pyo.Constraint(rule=tanBalanceRule)

	def caBalanceRule(blk):
		return (blk.concInCa * blk.liquidVolFlowIn
				== blk.concRetCa * blk.liquidVolFlowOut + blk.concPermCa * blk.permeateVolFlow)
	m.go.caBalance = pyo.Constraint(rule=caBalanceRule)

	def mgBalanceRule(blk):
		return (blk.concInMg * blk.liquidVolFlowIn
				== blk.concRetMg * blk.liquidVolFlowOut + blk.concPermMg * blk.permeateVolFlow)
	m.go.mgBalance = pyo.Constraint(rule=mgBalanceRule)

	# Constant-rejection species split:
	# C_perm,i = (1 - R_i) * C_feed,i
	def permTANRule(blk):
		return blk.concPermTAN == (1.0 - blk.rejectionTAN) * blk.concInTAN
	m.go.permTANDef = pyo.Constraint(rule=permTANRule)

	def permCaRule(blk):
		return blk.concPermCa == (1.0 - blk.rejectionCa) * blk.concInCa
	m.go.permCaDef = pyo.Constraint(rule=permCaRule)

	def permMgRule(blk):
		return blk.concPermMg == (1.0 - blk.rejectionMg) * blk.concInMg
	m.go.permMgDef = pyo.Constraint(rule=permMgRule)

	# -------------------- Osmotic pressure --------------------
	m.go.piFeedIn = pyo.Expression(
		expr=m.go.osmoticCoeffTAN * m.go.concInTAN
		   + m.go.osmoticCoeffCa  * m.go.concInCa
		   + m.go.osmoticCoeffMg  * m.go.concInMg
	)
	m.go.piRetentate = pyo.Expression(
		expr=m.go.osmoticCoeffTAN * m.go.concRetTAN
		   + m.go.osmoticCoeffCa  * m.go.concRetCa
		   + m.go.osmoticCoeffMg  * m.go.concRetMg
	)
	m.go.piFeedAvg = pyo.Expression(expr=0.5 * (m.go.piFeedIn + m.go.piRetentate))
	m.go.piPermeate = pyo.Expression(
		expr=m.go.osmoticCoeffTAN * m.go.concPermTAN
		   + m.go.osmoticCoeffCa  * m.go.concPermCa
		   + m.go.osmoticCoeffMg  * m.go.concPermMg
	)
	m.go.deltaPi = pyo.Expression(expr=m.go.piFeedAvg - m.go.piPermeate)

	def positiveFlux_rule(blk):
		return blk.deltaP >= blk.deltaPi + blk.minDrivingForce
	m.go.positiveFluxConstr = pyo.Constraint(rule=positiveFlux_rule)

	# -------------------- Membrane flux  --------------------
	# Qp = A * Lp * (dP - dPi); mass follows by multiplying through the
	# dedicated liquid density (permeate is dilute liquid, negligible solids).
	def membrane_flux_rule(blk):
		return blk.permeateVolFlow == blk.area * blk.membraneLp * (blk.deltaP - blk.deltaPi) * (1e-3 / 3600.0)
	m.go.membraneFlux = pyo.Constraint(rule=membrane_flux_rule)

	def pressure_limit_rule(blk):
		return blk.deltaP <= blk.maxDeltaP
	m.go.pressureLimit = pyo.Constraint(rule=pressure_limit_rule)

	def pH_balance_rule(blk):
		return blk.liquidpHOut == blk.sludgepHIn 
	m.go.pHBalance = pyo.Constraint(rule=pH_balance_rule)

	# -------------------- Costs --------------------
	m.go.capex = pyo.Expression(expr=m.go.capexFactor * m.go.membraneCost * m.go.area * 3)  # 3 = membrane replacement + peripherals
	m.go.feedFlow_m3h = pyo.Expression(expr=m.go.feedVolFlowBulk * 3600.0)  # m3/h, bulk (slurry) basis for the pump
	m.go.pumpPower = pyo.Expression(expr=(m.go.feedFlow_m3h * m.go.deltaP) / (36.0 * (m.go.pumpEfficiency + 1e-9)))  # kW
	m.go.opex = pyo.Expression(expr=m.go.pumpPower * m.elecPrice * (m.daysOperation / 3600.0))  # $