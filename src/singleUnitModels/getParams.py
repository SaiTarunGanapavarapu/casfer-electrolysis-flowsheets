#------------------------------------------------------------------------------
# function:     getParams.py                                                  #
# Description:  Function to load all the parameters in single unit models     #
#                                                                             #
# Input:        - m : Pyomo concrete model                                    #
#                                                                             #
# Output:       - m with all parameters                                       #
#                                                                             #
#------------------------------------------------------------------------------

import pandas as pd
import os
import math
import warnings

# Get the directory where this script (getParams.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(current_dir, 'params.xlsx')

try:
    df = pd.read_excel(excel_path)
except Exception as e:
    raise RuntimeError(f"Failed to read '{excel_path}'. Original error: {e}")
# Normalize column names (strip whitespace)
df.columns = df.columns.str.strip()

# Build the params dictionary: keys are unit operation names
params = {}
for unit, group in df.groupby('Unit Operation'):
    # Build a dict of parameters for this unit and strip parameter names
    param_dict = group.set_index('Parameter')['Value'].to_dict()
    param_dict = {str(k).strip(): v for k, v in param_dict.items()}
    params[str(unit).strip()] = param_dict

# Backwards-compatibility aliases: some modules expect slightly different
# unit names (e.g. 'Prep Tank' vs 'Preparation Tank', 'Receive Tank' vs
# 'Receiving Tank'). Create aliases so both spellings work.
aliases = {
    'Prep Tank': 'Preparation Tank',
    'Receive Tank': 'Receiving Tank',
}
for alias, actual in aliases.items():
    if actual in params and alias not in params:
        params[alias] = params[actual]


def getParams(m=None):
    """Optional helper to attach params to a model instance.

    Usage:
      import getParams
      getParams.getParams(m)   # attaches `m.params = getParams.params`

    Returning the module-level `params` dict is supported too.
    """
    if m is not None:
        setattr(m, 'params', params)
    return params
