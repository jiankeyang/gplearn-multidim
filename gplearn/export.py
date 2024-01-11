from ._program import _Program
from ._multi_program import _MultiOutputProgram
from .fitness import _fitness_map, _Fitness
from .functions import _function_map, _Function, sig1 as sigmoid
from .export_utils.generator import Generator

import numpy as np
import sympy


def get_var_real():
    X0 = sympy.Symbol('X0', real=True)
    X1 = sympy.Symbol('X1', real=True)
    X2 = sympy.Symbol('X2', real=True)
    X3 = sympy.Symbol('X3', real=True)
    X4 = sympy.Symbol('X4', real=True)
    C = sympy.Symbol('C', positive=True)
    VarDict = {
        'X0': X0,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'C': C,
    }
    return VarDict

VarDict = get_var_real()

def to_sympy(program):
    """
    Convert a GP program to a SymPy expression.

    Parameters
    ----------
    est_gp : gplearn.genetic.SymbolicRegressor
        A fitted symbolic estimator.

    Returns
    -------
    expr : sympy expression
        The SymPy expression representing the fitted estimator.
    """
    if isinstance(program, _Program):
        return _to_sympy(program.program)
    elif isinstance(program, _MultiOutputProgram):
        return [_to_sympy(p) for p in program.program]
    else:
        raise ValueError('est_gp must be a fitted symbolic estimator')
    
def _to_sympy(program):
    symbol_list, var_list, coef_list = parse_program_to_list(program)
    print(symbol_list)
    print(var_list)
    print(coef_list)
    infix_expr = Generator.prefix_to_infix(symbol_list, coefficients=coef_list, variables=var_list)
    print(infix_expr)
    sympy_expr = Generator.infix_to_sympy(infix_expr, VarDict, 'simplify')
    return sympy_expr

def parse_program_to_list(program):
    symbol_list = list()
    var_list = list()
    coef_list = list()

    for i in program:
        if isinstance(i, int):
            symbol_list.append('X' + str(i))
            var_list.append('X' + str(i))
        elif isinstance(i, float):
            symbol_list.append(str(i))
            coef_list.append(str(i))
        else:
            if i.name == 'log':
                symbol_list.append('ln')
            elif i.name == 'neg':
                symbol_list.append('sub')
                symbol_list.append('0')
            else:
                symbol_list.append(i.name)

    var_list = list(set(var_list))
    coef_list = list(set(coef_list))
    return symbol_list, var_list, coef_list
