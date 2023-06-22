import numpy as np


def firms_production_only_problem(volumes, firm, prices_flatten, prices_mean, desired_shape):
    v_reshaped = volumes.reshape(desired_shape).sum(axis=0)
    _, firms_out = firm.test_produce(firm.reserves + v_reshaped)
    gain = firms_out.sum()
    cost = prices_flatten @ volumes
    return - gain + cost * (firm.alpha if hasattr(firm, 'alpha') else 0.01)


def firms_investment_only_problem(volumes, firm, prices_flatten, prices_mean, desired_shape):
    v_reshaped = volumes.reshape(desired_shape).sum(axis=0)
    _, gain = firm.test_investment(firm.reserves + v_reshaped)
    cost = prices_flatten @ volumes
    gain_per_limit = firm.history['gains'] / np.maximum(firm.limits_history[-1], 1)
    # if gain == 0 and firm.limit < 2:
    #     return 0.001
    return - gain * firm.deprecation_steps * np.maximum(gain_per_limit, 1) + cost * 1e-6


def firms_extended_investment_problem(volumes, firm, prices_flatten, prices_mean, desired_shape):
    v_reshaped = volumes.reshape(desired_shape).sum(axis=0)
    _, gain = firm.test_investment(firm.reserves + v_reshaped)
    cost = prices_flatten @ volumes
    gain_per_limit = firm.history['gains'] / firm.limits_history[-1]
    return - gain * gain_per_limit * firm.deprecation_steps + cost


def firms_mixed_problem(volumes, firm, prices_flatten, prices_mean, desired_shape):
    v_reshaped = volumes.reshape(desired_shape).sum(axis=0)
    spent_reserves, delta_limit = firm.test_investment(firm.reserves + v_reshaped)
    _, firms_out = firm.test_produce(firm.reserves + v_reshaped - spent_reserves)
    gain = firms_out @ prices_flatten.reshape(desired_shape).mean(axis=0)
    cost = prices_flatten @ volumes
    return - gain - delta_limit * firm.beta + cost * (firm.alpha if hasattr(firm, 'alpha') else 1)


def consumption_problem(volumes, consumer, prices_flatten, prices_mean, desired_shape):
    v_reshaped = volumes.reshape(desired_shape).sum(axis=0)
    _, gain = consumer.test_consumption(v_reshaped)
    cost = prices_flatten @ volumes
    return - gain + cost


def firms_profit_only_problem_old(volumes, firm, prices_flatten, prices_mean, desired_shape):
    v_reshaped = volumes.reshape(desired_shape).sum(axis=0)
    _, firms_out = firm.test_produce(firm.reserves + v_reshaped)
    gain = firms_out @ prices_flatten.reshape(desired_shape).mean(axis=0)
    cost = prices_flatten @ volumes
    return - gain + cost


def firms_profit_only_problem(volumes, firm, prices_flatten, prices_mean, desired_shape):
    v_reshaped = volumes.reshape(desired_shape).sum(axis=0)
    _, firms_out = firm.test_produce(firm.reserves + v_reshaped)
    gain = firms_out @ prices_flatten.reshape(desired_shape)[firm.id]
    cost = prices_flatten @ volumes
    if gain == 0:
        return 0.0001 + cost
    return - gain + cost
