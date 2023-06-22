import numpy as np
from scipy.special import logsumexp

from config import DEBUG
from solvers import solve_float_problem
import json


class BaseFirm:
    def __init__(self,
                 tech_matrix: np.ndarray,
                 out_matrix: np.ndarray,
                 invest_matrix: np.ndarray,
                 financial_resources: float = 0,
                 limit: int = 1,
                 id_: int = 0,
                 is_deprecating=False,
                 deprecation_steps=10,
                 sale_percent=0.9,
                 deprecation_array=None,
                 invest_anyway=True
                 ):
        """
        :param tech_matrix: Технологическая матрица (Затраты-1 ед. Выпуск)
        :param out_matrix: Матрица выпуска (из нулей и единиц)
        :param invest_matrix: Инвестиционная матрица (Затраты-Инвестиции)
        :param financial_resources: Объём финансовых ресурсов
        :param limit: Максимальный объём производства фирмы
        :param sale_percent: Доля резервов, которые фирма выставляет за одну итерацию
        :param deprecation_steps: количество шагов, пока основной капитал не пропадёт
        :attribute reserves: Объём запасов фирмы
        """
        self.invest_anyway = invest_anyway
        self.tech_matrix = tech_matrix
        self.out_matrix = out_matrix
        self.invest_matrix = invest_matrix
        self.financial_resources = financial_resources
        self.__limit = limit
        self.reserves = np.zeros(shape=tech_matrix.shape, dtype=self.tech_matrix.dtype)
        self.id = id_
        self.is_deprecating = is_deprecating
        self.deprecation_steps = deprecation_steps
        self.sale_percent = sale_percent
        if deprecation_array is None:
            self.deprecation_array = np.full(limit, deprecation_steps, dtype=int)
        else:
            self.deprecation_array = deprecation_array

        self.history = {
            'volume': 0,
            'investment': None,
            'gains': 1,
            'gains_history': [],
            'price': None,
            'reserves': None,
            'financial_resources': 0,
            'previous_volumes': np.zeros_like(out_matrix),
            'previous_produced': None,
            'previous_demand': 0,
        }
        self.min_limit = 0

    @property
    def limit(self):
        return len(self.deprecation_array)

    @property
    def next_limit(self):
        return len(self.deprecation_array[self.deprecation_array >= 1])

    def define_prices(self, market):
        """
        Определить цены для рынка
        :param market:
        :return:
        """
        mask = self.out_matrix != 0
        # средние цены
        # текущие цены фирмы
        firms_prices = market.price_matrix[self.id]
        # текущие объёмы фирмы на рынке
        current_firms_volumes = market.volume_matrix[self.id]
        # объёмы фирмы на предыдущем шаге
        last_firms_volumes = self.history['previous_volumes']
        # текущий спрос на продукцию фирмы
        current_demand = last_firms_volumes - current_firms_volumes
        previous_demand = self.history['previous_demand'] + 1e-10

        increase_delta = np.select(
            condlist=[
                (previous_demand <= 1e-9) & (current_demand <= 1e-9) & (current_firms_volumes <= 1e-9),
                (previous_demand <= 1e-9) & (current_demand <= 1e-9) & (current_firms_volumes > 1e-9),
            ],
            choicelist=[
                +1,
                -1
            ],
            default=(current_demand - previous_demand) / previous_demand,
        )
        increase_delta *= 0.25
        new_prices = firms_prices + increase_delta
        new_prices = np.minimum(np.maximum(market.MIN_PRICE, new_prices), market.MAX_PRICE).round(4)
        new_prices = np.where(mask, new_prices, 0)
        market.process_prices(self.id, new_prices)

        self.history['price'] = new_prices
        self.history['previous_demand'] = current_demand

    def sell(self, market):
        """
        Выставить произведенные товары на рынок

        param market: Рынок
        :return:
        """
        goods = np.where(self.out_matrix != 0, self.reserves * self.sale_percent, 0)
        market.process_sales(self.id, goods)
        self.reserves -= goods
        self.history['previous_volumes'] = market.volume_matrix[self.id]
        self.history['previous_produced'] = goods

    def buy_decision(self, mode, market):
        """
        Принятие частного решения о покупке чего-либо
        :param mode: режим: ['profit', 'production', 'investment'] - цель покупки
        :param market: рынок
        :return:
        """
        purchase_matrix = solve_float_problem(self, market, mode)  # матрица объёмов купленного у каждой фирмы
        sellers_gains = (purchase_matrix * market.price_matrix).sum(axis=1)  # получено продавцами
        total_cost = sellers_gains.sum()  # общая стоимость купленного
        new_reserves = purchase_matrix.sum(axis=0)  # вектор купленных ресурсов

        # обновить атрибуты фирмы
        self.financial_resources -= total_cost
        self.reserves += new_reserves

        # обновить атрибуты рынка
        market.process_purchases(purchase_matrix, sellers_gains)

    def step(self, market):
        """
        Шаг фирмы в модели
        :param market:
        :param process_gain:
        :return:
        """
        pass

    def test_produce(self, reserves):
        """
        Посмотреть изменения запасов при производстве

        :param reserves: резервы фирмы
        :return: new_reserves - изменённые запасы
        """
        non_zero = self.tech_matrix != 0
        proportions = reserves[non_zero] / self.tech_matrix[non_zero]
        commodity_amount: float = np.maximum(np.minimum(-logsumexp(-proportions), self.limit), 0)
        input_reserves, new_reserves = commodity_amount * self.tech_matrix, commodity_amount * self.out_matrix
        return input_reserves, new_reserves

    def produce(self):
        """
        Произвести товары по технологии Леонтьева
        :return commodity_ammount: количество произведенного фирмой товара
        """
        input_reserves, out_reserves = self.test_produce(self.reserves)
        self.history['volume'] = out_reserves
        self.reserves -= input_reserves
        self.reserves += out_reserves

        prod_proportion = np.nansum(self.history['volume'] / self.out_matrix) / self.limit
        optimal_limit = np.ceil(np.maximum(prod_proportion * self.limit, 1))
        need_new_limits = np.maximum(optimal_limit - self.next_limit, 0)
        self.history['max_delta_limit'] = np.maximum(need_new_limits, 1)
        if prod_proportion >= 0.95:
            self.history['max_delta_limit'] += np.log1p(self.limit)
            self.history['investment_problem'] = 'extended_investment'
        return out_reserves

    def test_investment(self, reserves):
        """
        Посмотреть на изменения в лимите при инвестициях (СГЛАЖЕННЫЕ)
        :param reserves:
        :return:
        """
        non_zero = self.invest_matrix != 0
        proportions = reserves[non_zero] / self.invest_matrix[non_zero]
        soft_delta_limit: float = np.minimum(np.maximum(-logsumexp(-proportions), 0), self.history['max_delta_limit'])
        return soft_delta_limit * self.invest_matrix, soft_delta_limit

    def invest(self):
        """
        Инвестиции фирмы
        """
        if self.invest_anyway:
            self.history['max_delta_limit'] = np.maximum(self.history['max_delta_limit'], 1)
        _, soft_delta_limit = self.test_investment(self.reserves)
        delta_limit = np.floor(soft_delta_limit).astype(int)
        self.reserves -= delta_limit * self.invest_matrix
        self.deprecation_array = np.hstack((self.deprecation_array,
                                            np.full(delta_limit, self.deprecation_steps)))
        return self.limit

    def __repr__(self):
        f = lambda y: np.array2string(y, formatter={'float_kind': lambda x: "%.2f" % x})
        representation = f"Фирма {self.id} (" \
                         f"\n  Технологическая матрица: {f(self.tech_matrix)}" \
                         f"\n  Матрица выхода:          {f(self.out_matrix)}" \
                         f"\n  Инвестиционная матрица:  {f(self.invest_matrix)}" \
                         f"\n  Запасы:                  {f(self.reserves)}" \
                         f"\n  Амортизация:             {f(self.deprecation_array)}" \
                         f"\n  Лимит:                   {self.limit}" \
                         f"\n  Финансы:                 {self.financial_resources}" \
                         f")"
        return representation

    def deprecation(self):
        """
        Амортизация. Уменьшить срок службы основного капитала на единицу. Если срок службы основного капитала
        меньше нуля, то уменьшить лимиты фирмы.
        :return:
        """
        if self.is_deprecating:
            self.deprecation_array = self.deprecation_array[self.deprecation_array >= 1] - 1
        if self.limit == 0 and self.min_limit > 0:
            self.deprecation_array = np.full(self.min_limit, self.deprecation_steps)
