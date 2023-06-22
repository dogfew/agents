import numpy as np
from scipy.special import logsumexp

from solvers.float_solver import solve_float_problem


class BaseConsumer:

    def __init__(self,
                 utility_matrix: np.ndarray,
                 satisfaction_level=10,
                 id_=0,
                 financial_resources: float = 0,
                 property: dict = None,
                 ):
        """
        :param utility_matrix: леонтьевская функция полезности домохозяйства
        :param financial_resources: объём финансовых ресурсов домохозяйства
        :param property: собственность домохозяйства (словарь)
        """
        self.utility_matrix = utility_matrix
        self.satisfaction_level = satisfaction_level
        self.id_ = id_
        self.financial_resources = financial_resources
        if property is not None:
            self.property = property
        else:
            self.property = dict()
        self.total_utility = 0

    def buy(self, market):
        """
        Покупатель просто "сжигает" купленные товары
        :param market: рынок
        :return:
        """
        purchase_matrix = solve_float_problem(self, market, mode='consumption')
        print(purchase_matrix)
        sellers_gains = (purchase_matrix * market.price_matrix).sum(axis=1)  # получено продавцами
        total_cost = sellers_gains.sum()  # общая стоимость купленного
        new_reserves = purchase_matrix.sum(axis=0)  # вектор купленных ресурсов
        self.financial_resources -= total_cost
        self.total_utility += self.test_consumption(new_reserves)[1]

        market.process_purchases(purchase_matrix, sellers_gains)

    def step(self, market):
        self.buy(market)

    def test_consumption(self, reserves):
        """
        Посмотреть на значение функции полезности при данных закупках
        :return:
        """
        print("reserves", reserves)
        non_zero = self.utility_matrix != 0
        print('non-zero', non_zero)
        proportions = reserves[non_zero] / self.utility_matrix[non_zero]
        utility_value: int | float = np.minimum(np.maximum(-logsumexp(-proportions), 0), self.satisfaction_level)
        input_reserves = utility_value * self.utility_matrix
        return input_reserves, utility_value

    def __repr__(self):
        return f"BaseConsumer(\n" \
               f"\tutility_array: {self.utility_matrix}\n" \
               f"\tTotal utility: {self.total_utility}\n" \
               f"\tFinance: {self.financial_resources})"
