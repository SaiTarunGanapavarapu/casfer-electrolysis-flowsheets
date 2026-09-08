#------------------------------------------------------------------------------
# function:     getStandardParams.py                                          #
# Description:  Function to load all the parameters in the ED model           #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all parameters                                       #
#                                                                             #
#------------------------------------------------------------------------------

import pyomo.environ as pyo
import src.singleUnitModels.getParams as getParams

def getStandardParams(m):

    # Load parameters from getParams
    standardParams = getParams.params['Standard Constants']

    # Define parameters for the standard unit operations
    m.daysOperation   = pyo.Param(initialize = standardParams['Lifetime'])   # lifetime of operation (s)
    m.elecPrice       = pyo.Param(initialize = standardParams['Electricity Price'])   # $/kWh
    m.accGravity      = pyo.Param(initialize = standardParams['Acceleration due to Gravity'])  # m/s2

    m.totalFlowIn     = pyo.Param(initialize = standardParams['Total Flow of Wastewater In'])  # m3/s
    m.sludgeTSSin     = pyo.Param(initialize = standardParams['Feed TSS'])           # fraction
    m.sludgeDensity   = pyo.Param(initialize = standardParams['Sludge Density'])     # kg/m3
    m.sludgeViscosity = pyo.Param(initialize = standardParams['Sludge Viscosity'])   # Pa.s

    m.liquidDensity   = pyo.Param(initialize = standardParams.get('Liquid Density', 1000.0), mutable=True)
    