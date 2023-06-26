import numpy as np
from scipy.special import logsumexp

from solvers import solve_float_problem, solve_share_problem


class BaseConsumer:

    def __init__(self,
                 utility_matrix: np.ndarray,
                 satisfaction_level=10,
                 id_=0,
                 financial_resources: float = 0,
                 property: dict = None,
                 mpc=0.7
                 ):
        """
        :param utility_matrix: леонтьевская функция полезности домохозяйства
        :param financial_resources: объём финансовых ресурсов домохозяйства
        :param property: собственность домохозяйства (словарь)
        """
        self.utility_matrix = utility_matrix
        self.satisfaction_level = satisfaction_level
        self.id = id_
        self.financial_resources = financial_resources
        self.mpc = mpc
        self.total_utility = 0

    def buy(self, market):
        """
        Покупатель просто "сжигает" купленные товары
        :param market: рынок
        :return:
        """
        purchase_matrix = solve_float_problem(self, market, mode='consumption')
        sellers_gains = (purchase_matrix * market.price_matrix).sum(axis=1)  # получено продавцами
        total_cost = sellers_gains.sum()  # общая стоимость купленного
        new_reserves = purchase_matrix.sum(axis=0)  # вектор купленных ресурсов
        self.financial_resources -= total_cost
        self.total_utility += self.test_consumption(new_reserves)[1]

        market.process_purchases(purchase_matrix, sellers_gains)

    def step(self, market, stock_market=None):
        """Получить деньги с рынка"""
        if stock_market is not None:
            self.financial_resources += stock_market.process_gains(consumer_id=self.id)
            invest_resources = (1 - self.mpc) * self.financial_resources
            self.financial_resources *= self.mpc
        """Купить товары"""
        self.buy(market)
        """Инвестировать"""
        if stock_market is not None:
            self.financial_resources += invest_resources
            self.invest(market, stock_market)

    def test_consumption(self, reserves):
        """
        Посмотреть на значение функции полезности при данных закупках
        :return:
        """
        non_zero = self.utility_matrix != 0
        proportions = reserves[non_zero] / self.utility_matrix[non_zero]
        utility_value: int | float = np.minimum(np.maximum(-logsumexp(-proportions), 0), self.satisfaction_level)
        input_reserves = utility_value * self.utility_matrix
        return input_reserves, utility_value

    def invest(self, market, stock_market):
        investments = solve_share_problem(self, stock_market)
        self.financial_resources -= investments.sum()
        market.gains += investments
        stock_market.shareholders[:, self.id] += investments

    def __repr__(self):
        return f"Consumer(\n" \
               f"\tutility_array: {self.utility_matrix}\n" \
               f"\tTotal utility: {self.total_utility}\n" \
               f"\tFinance: {self.financial_resources})"
