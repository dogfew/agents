from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution, Bounds, minimize

# from firms.base_firm import BaseFirm
from market.market import Market
# from consumers.base_consumer import BaseConsumer

from solvers.tasks import *


def solve_float_problem(agent: 'BaseFirm' | 'BaseConsumer',
                        market: Market,
                        mode='production',
                        budget=None
                        ):
    """
    Решить в действительных числах задачу агента (фирмы/потребителя):
    f(v') - alpha * cost -> max
    where
        v' = sum_j V'_ij
        cost = sum_ij (V'_ij * P_ij)
    s.t.
        V'_ij <= V_ij forall i, j
        f(v') <= limit
        cost <= agent's budges

    :param agent: Фирма
    :param market: Рынок
    :param mode: Задача
    :param budget: Бюджет, исходя из которого принимаются решения
    :return: Результаты оптимизации
    """
    if mode == 'profit':
        problem = firms_profit_only_problem
        nonzero = agent.tech_matrix != 0
    elif mode == 'production':
        problem = firms_production_only_problem
        nonzero = agent.tech_matrix != 0
    elif mode == 'investment':
        problem = firms_investment_only_problem
        nonzero = agent.invest_matrix != 0
    elif mode == 'extended_investment':
        problem = firms_extended_investment_problem
        nonzero = agent.invest_matrix != 0
    elif mode == 'prod-invest':
        problem = firms_mixed_problem
        nonzero = (agent.invest_matrix != 0) | (agent.tech_matrix != 0)
    elif mode == 'consumption':
        problem = consumption_problem
        nonzero = agent.utility_matrix != 0
    else:
        raise NotImplementedError

    desired_shape = market.volume_matrix.shape
    volumes_flatten = np.where(nonzero,  market.volume_matrix, 0).flatten()
    prices_flatten = market.price_matrix.flatten()
    if mode != 'consumption':
        agents_prices = market.price_matrix[agent.id]
    else:
        agents_prices = None
    if agent.financial_resources < 0.0001:
        return np.zeros(desired_shape)
    if hasattr(agent, 'history') and agent.history.get(mode) is not None:
        x0 = np.maximum(
            np.minimum(
                agent.history[mode],
                volumes_flatten
            ),
            0)
        if prices_flatten @ x0 >= agent.financial_resources:
            x0 = np.zeros_like(volumes_flatten)
    else:
        x0 = np.zeros_like(volumes_flatten)
    try:
        # noinspection PyTypeChecker
        res = minimize(problem, x0=x0,
                       args=(agent, prices_flatten, agents_prices, desired_shape),
                       constraints=LinearConstraint(prices_flatten, ub=agent.financial_resources, keep_feasible=True),
                       bounds=Bounds(0, volumes_flatten, keep_feasible=True))
        x = res.x.reshape(desired_shape).astype(volumes_flatten.dtype)
    except Exception as e:
        print("Error", e)
        print('x0:', x0)
        print("prices:", market.price_matrix)
        print('volumes:', market.volume_matrix)
        print("FinRecources", agent.financial_resources)
        print("p@v", prices_flatten @ x0)
        x = np.zeros(desired_shape)
    if hasattr(agent, 'history'):
        agent.history[mode] = x.flatten()
    return x