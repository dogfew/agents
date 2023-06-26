import numpy as np
from scipy.optimize import minimize, Bounds


def solve_share_problem(consumer, stock_market):
    budget = consumer.financial_resources
    expected_profits = stock_market.profits_history
    n_firms = stock_market.n_firms

    def target(x):
        new_shareholders = stock_market.shareholders.copy()
        new_shareholders[:, consumer.id] += x * budget
        new_shares = (new_shareholders / new_shareholders.sum(axis=1)[:, None])[:, consumer.id]
        return - new_shares @ expected_profits

    res = minimize(target,
                   x0=np.full(n_firms, 1 / n_firms),
                   constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1},
                   bounds=Bounds(0, 1, keep_feasible=True))
    return res.x * budget
