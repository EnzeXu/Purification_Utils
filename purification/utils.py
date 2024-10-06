import numpy as np
import sympy as sp

from concurrent.futures import ProcessPoolExecutor


def is_term_constant(term):
    """
    Whether a term in sympy is a constant
    :param term: a sympy equation
    :return: a boolean
    """
    return isinstance(term, (sp.Float, sp.Integer, sp.Rational))


def extract(expression):
    """
    Extract the `full_terms`, `terms`, and `coefficient_terms` (all in list) of an expression
    The order of each part is by str comparison of items in `terms`
    :param expression: a string. E.g., "3*y+2*sin(x)-3*x**2+2*x*y"
    :return: [2*sin(x), -3*x**2, 2*x*y, 3*y], [sin(x), x**2, x*y, y], [2, -3, 2, 3]
    """
    expr = sp.sympify(expression)
    raw_terms = list(sp.Add.make_args(expr))
    result = []

    for term in raw_terms:
        term = sp.sympify(term)
        coefficient_list = []
        non_coefficient_list = []
        for factor in sp.Mul.make_args(term):
            if is_term_constant(factor):
                coefficient_list.append(factor)
            else:
                non_coefficient_list.append(factor)
        coefficient_part = sp.sympify(sp.Mul(*[sp.sympify(item) for item in coefficient_list]))
        non_coefficient_part = sp.sympify(sp.Mul(*[sp.sympify(item) for item in non_coefficient_list]))
        result.append([term, non_coefficient_part, coefficient_part])

    result = sorted(result, key=lambda x: str(x[1]))
    full_terms = [item[0] for item in result]
    terms = [item[1] for item in result]
    coefficient_terms = [item[2] for item in result]
    return full_terms, terms, coefficient_terms


def evaluate_expression(expression_str, variable_list, variable_values):
    # print("debug:", expression_str, variable_list, variable_values)
    variables = sp.symbols(variable_list)
    expr = sp.sympify(expression_str)
    variable_dict = {var: val for var, val in zip(variables, variable_values)}
    result = expr.subs(variable_dict)
    result_value = result.evalf()
    return result_value


def purify_2d_sequential(eq, data, variable_list, threshold=0.05):
    # data is in shape (N, m). Here m is the dimension of the ODE system
    full_terms, terms, _ = extract(eq)
    n = data.shape[0]
    abs_value_array = np.zeros([n, len(full_terms)])
    abs_ratio_array = np.zeros([n, len(full_terms)])
    for i in range(n):
        for j, one_full_term in enumerate(full_terms):
            abs_value_array[i, j] = np.abs(evaluate_expression(one_full_term, variable_list, data[i]))
        for j in range(len(full_terms)):
            abs_ratio_array[i, j] = abs_value_array[i, j] / np.sum(abs_value_array[i])
    avg_ratio = np.average(abs_ratio_array, axis=0)
    purified_full_terms = [full_terms[i] for i in range(len(full_terms)) if avg_ratio[i] >= threshold]
    purified_eq = sp.sympify(sp.Add(*purified_full_terms))
    return purified_eq, avg_ratio, full_terms, terms

def purify_3d_sequential(eq, data, variable_list, threshold=0.05):
    # data is in shape (n_traj, N, m). Here m is the dimension of the ODE system
    full_terms, terms, _ = extract(eq)
    n_traj, n = data.shape[0], data.shape[1]
    abs_value_array = np.zeros([n_traj * n, len(full_terms)])
    abs_ratio_array = np.zeros([n_traj * n, len(full_terms)])
    for i_traj in range(n_traj):
        for i in range(n):
            for j, one_full_term in enumerate(full_terms):
                abs_value_array[i_traj * n + i, j] = np.abs(evaluate_expression(one_full_term, variable_list, data[i_traj, i]))
        for i in range(n):
            for j in range(len(full_terms)):
                abs_ratio_array[i_traj * n + i, j] = abs_value_array[i_traj * n + i, j] / np.sum(abs_value_array[i_traj * n + i])
    avg_ratio = np.average(abs_ratio_array, axis=0)
    purified_full_terms = [full_terms[i] for i in range(len(full_terms)) if avg_ratio[i] >= threshold]
    purified_eq = sp.sympify(sp.Add(*purified_full_terms))
    return purified_eq, avg_ratio, full_terms, terms


def process_traj(i_traj, data, full_terms, variable_list):
    n = data.shape[1]
    abs_value_array = np.zeros([n, len(full_terms)])
    abs_ratio_array = np.zeros([n, len(full_terms)])

    for i in range(n):
        for j, one_full_term in enumerate(full_terms):
            abs_value_array[i, j] = np.abs(evaluate_expression(one_full_term, variable_list, data[i_traj, i]))

    for i in range(n):
        for j in range(len(full_terms)):
            abs_ratio_array[i, j] = abs_value_array[i, j] / np.sum(abs_value_array[i])

    return abs_ratio_array


def purify_3d_parallel(eq, data, variable_list, threshold=0.05, max_workers=None):
    # data is in shape (n_traj, N, m). Here m is the dimension of the ODE system
    full_terms, terms, _ = extract(eq)
    n_traj = data.shape[0]

    abs_ratio_array = np.zeros([n_traj * data.shape[1], len(full_terms)])

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_traj, i_traj, data, full_terms, variable_list) for i_traj in range(n_traj)]

        for i_traj, future in enumerate(futures):
            abs_ratio_array[i_traj * data.shape[1]: (i_traj + 1) * data.shape[1]] = future.result()

    avg_ratio = np.average(abs_ratio_array, axis=0)
    purified_full_terms = [full_terms[i] for i in range(len(full_terms)) if avg_ratio[i] >= threshold]
    purified_eq = sp.sympify(sp.Add(*purified_full_terms))

    return purified_eq, avg_ratio, full_terms, terms



