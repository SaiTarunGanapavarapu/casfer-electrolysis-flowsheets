#------------------------------------------------------------------------------
# function:     storageTank.py                                               #
# Description:  Function to define storage tank model equations              #
#               Stores product output for 2 days                             #
#               Capital cost only (no operating cost)                        #
#                                                                            #
# Input:        - m : Pyomo concrete model                                   #
#                                                                            #
# Output:       - m with all storage tank model equations                    #
#                                                                            #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
try:
    from . import getParams
except ImportError:
    import getParams

def storageTank(m, blockName='st'):

    # create a named block on the model so multiple storage tanks can exist
    setattr(m, blockName, pyo.Block())
    blk = getattr(m, blockName)

    # Load parameters for storage tank from getParams
    storageTankParams = getParams.params.get('Storage Tank', {})

    def _get_param(key, default):
        val = storageTankParams.get(key, default)
        try:
            return float(val)
        except Exception:
            return default

    blk.costReference   = pyo.Param(initialize=_get_param('Cost Reference', 254842.0))   # $
    blk.volumeReference = pyo.Param(initialize=_get_param('Volume Reference', 249.83718))  # m3
    blk.capexFactor     = pyo.Param(initialize=_get_param('Capex Factor', 0.6))  # dimensionless
    blk.storageTimeHrs  = pyo.Param(initialize=48.0, mutable=True)  # 2 days = 48 hours
    blk.minCapex        = pyo.Param(initialize=5000.0, mutable=True)  # minimum capex

    # Fixed physical estimate of the stored product's bulk density, solely for capex
    blk.productDensity  = pyo.Param(initialize=_get_param('Product Density', 1200.0), mutable=True)  # kg/m3

    # -------------------- Primary state variables  --------------------
    blk.productMassFlowIn  = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals, bounds=(0, None))  # kg/s
    blk.productMassFlowOut = pyo.Var(initialize=1.0, within=pyo.NonNegativeReals, bounds=(0, None))  # kg/s
    blk.tankVolume         = pyo.Var(initialize=100, within=pyo.NonNegativeReals, bounds=(0, None))  # m3

    # Capital Cost
    blk.capex = blk.minCapex + 1.64 * blk.costReference * (blk.tankVolume / blk.volumeReference) ** blk.capexFactor

    # Operating Cost (zero for storage tank)
    blk.opex = pyo.Expression(expr=0.0)

    # Tank volume computation -- sized for storageTimeHrs of product storage. 
    def storageTankVolumeRule(b):
        return b.tankVolume == b.storageTimeHrs * 3600.0 * (b.productMassFlowIn / b.productDensity)  # unit is m3
    blk.storageTankVolume = pyo.Constraint(rule=storageTankVolumeRule)

    # Material Balance (pure mass balance, no density)
    def materialBalance(b):
        return b.productMassFlowOut == b.productMassFlowIn  # unit is kg/s
    blk.materialBalance = pyo.Constraint(rule=materialBalance)