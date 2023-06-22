import numpy as np

from firms.base_firm import BaseFirm
from scipy.special import expit


class ComplexFirm(BaseFirm):
    def __init__(self,
                 tech_matrix: np.ndarray,
                 out_matrix: np.ndarray,
                 invest_matrix: np.ndarray,
                 financial_resources: float = 0,
                 limit: int = 1,
                 id_: int = 0,
                 is_deprecating=False,
                 deprecation_steps=30,
                 profit_rate=0,
                 production_rate=0.8,
                 sale_percent=1.,
                 min_limit=0,
                 prod_name=None,
                 invest_anyway=True
                 ):
        super().__init__(tech_matrix,
                         out_matrix,
                         invest_matrix,
                         financial_resources,
                         limit,
                         id_,
                         is_deprecating,
                         deprecation_steps,
                         sale_percent,
                         None,
                         invest_anyway)
        self.profit_rate = profit_rate
        self.production_rate = production_rate
        self.investment_rate = np.float64(1.) - self.production_rate
        self.profits = 0
        self.production_resources = self.financial_resources * self.production_rate
        self.investment_resources = self.financial_resources * (1 - self.production_rate)

        self.min_limit = min_limit

        self.gains_history = []
        self.limits_history = []

        if prod_name is None:
            self.prod_name = np.argsort(self.out_matrix)[-1]
        else:
            self.prod_name = prod_name

    def step(self, market):
        """
        Шаг фирмы в модели
        :param market: Рынок
        """
        self.production_resources = self.financial_resources * self.production_rate
        self.investment_resources = self.financial_resources * self.investment_rate
        self.financial_resources = 0.
        """Получение выигрышей с рынка"""
        gains = market.process_gains(self.id)  # получить выигрыш с рынка.

        self.history['gains'] = gains  # сохранение выигрышей в истории
        self.gains_history.append(gains)
        self.limits_history.append(self.limit)
        # self.limits_history = self.limits_history[-self.deprecation_steps * 2:]
        # self.gains_history = self.gains_history[-self.deprecation_steps * 2:]

        # изменение прибылей
        d_profits = gains * self.profit_rate
        gains -= d_profits
        self.profits += d_profits
        d_prod_resources = gains * self.production_rate
        d_investment_resources = gains * self.investment_rate
        self.production_resources += d_prod_resources
        self.investment_resources += d_investment_resources
        gains -= d_prod_resources
        gains -= d_investment_resources
        assert -1e-10 < gains < 1e-10
        """Амортизация"""
        self.deprecation()
        """Определение цен"""
        self.define_prices(market)
        """Производство"""
        self.financial_resources, self.production_resources = self.production_resources, 0
        self.history['investment_problem'] = 'investment'
        if self.limit > 0:
            if self.history.get('production') is not None:
                self.buy_decision('profit', market)
            else:
                self.buy_decision('production', market)
            self.produce()
            self.production_resources, self.financial_resources = self.financial_resources, 0
            """Продажа товаров"""
            self.history['reserves'] = self.reserves
            self.sell(market)
        else:
            self.history['max_delta_limit'] = 1
            self.history['volume'] = np.zeros_like(self.out_matrix)
            self.history['previous_volumes'] = np.zeros_like(self.out_matrix)
            self.investment_resources += self.financial_resources
            self.financial_resources = 0

        """Инвестиции"""
        self.financial_resources, self.investment_resources = self.investment_resources, 0
        if self.financial_resources > 0:
            if self.history['investment_problem'] == 'investment':
                self.buy_decision('investment', market)
            else:
                self.buy_decision('extended_investment', market)
        self.invest()
        self.investment_resources = self.financial_resources
        self.financial_resources = self.investment_resources + self.production_resources

    def __repr__(self):
        f = lambda y: np.array2string(y, formatter={'float_kind': lambda x: "%.2f" % x})
        representation = f"Фирма {self.id} (" \
                         f"\n  Технологическая матрица: {f(self.tech_matrix)}" \
                         f"\n  Матрица выхода:          {f(self.out_matrix)}" \
                         f"\n  Инвестиционная матрица:  {f(self.invest_matrix)}" \
                         f"\n  Запасы:                  {f(self.reserves)}" \
                         f"\n  Амортизация:             {f(self.deprecation_array)}" \
                         f"\n  Лимит:                   {self.limit}" \
                         f"\n  Процент на продажу:      {self.sale_percent}" \
                         f"\n  Отчисления на прибыль:   {self.profit_rate}" \
                         f"\n  Финансы:                 {self.financial_resources}" \
                         f")"
        return representation


class ComplexSellFirm(ComplexFirm):
    def __init__(self,
                 tech_matrix: np.ndarray,
                 out_matrix: np.ndarray,
                 invest_matrix: np.ndarray,
                 financial_resources: float = 0,
                 limit: int = 1,
                 id_: int = 0,
                 is_deprecating=False,
                 deprecation_steps=15,
                 profit_rate=0,
                 production_rate=0.8,
                 sale_percent=1.,
                 min_limit=0,
                 prod_name=None,
                 invest_anyway=True
                 ):
        super().__init__(tech_matrix,
                         out_matrix,
                         invest_matrix,
                         financial_resources,
                         limit,
                         id_,
                         is_deprecating,
                         deprecation_steps,
                         profit_rate,
                         production_rate,
                         sale_percent,
                         min_limit,
                         prod_name,
                         invest_anyway
                         )
        super().__init__(tech_matrix, out_matrix, invest_matrix,
                         financial_resources, limit, id_,
                         is_deprecating, deprecation_steps,
                         profit_rate, production_rate, sale_percent, min_limit, prod_name)
        self.history['step'] = 0

    def define_prices(self, market):
        """
        Определить цены для рынка
        :param market:
        :return:
        """
        mask = self.out_matrix != 0
        # средние цены
        mean_prices = market.mean_prices
        # текущие цены фирмы
        firms_prices = market.price_matrix[self.id]
        # текущие объёмы фирмы на рынке
        current_firms_volumes = market.volume_matrix[self.id]
        # объёмы фирмы на предыдущем шаге
        last_firms_volumes = self.history['previous_volumes']
        # текущий спрос на продукцию фирмы
        current_demand = last_firms_volumes - current_firms_volumes
        previous_demand = self.history['previous_demand'] + 1e-10

        increase_percent = np.select(
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
        increase_percent = expit(increase_percent)
        increase_percent -= 0.5
        new_prices = firms_prices * (1 + increase_percent)
        new_prices = np.minimum(np.maximum(market.MIN_PRICE, new_prices), market.MAX_PRICE).round(4)
        new_prices = np.where(mask, new_prices, 0)
        market.process_prices(self.id, new_prices)

        self.history['price'] = new_prices
        self.history['previous_demand'] = current_demand


class RegulatedFirm(ComplexFirm):
    def __init__(self,
                 tech_matrix: np.ndarray,
                 out_matrix: np.ndarray,
                 invest_matrix: np.ndarray,
                 financial_resources: float = 0,
                 limit: int = 1,
                 id_: int = 0,
                 is_deprecating=False,
                 deprecation_steps=15,
                 profit_rate=0,
                 production_rate=0.8,
                 sale_percent=1.,
                 min_limit=0,
                 prod_name=None,
                 invest_anyway=True
                 ):
        super().__init__(tech_matrix,
                         out_matrix,
                         invest_matrix,
                         financial_resources,
                         limit,
                         id_,
                         is_deprecating,
                         deprecation_steps,
                         profit_rate,
                         production_rate,
                         sale_percent,
                         min_limit,
                         prod_name,
                         invest_anyway
                         )
        self.history['previous_volumes'] = np.zeros_like(out_matrix)
        self.history['step'] = 0

    def define_prices(self, market):
        """
        Определить цены для рынка
        :param market:
        :return:
        """
        self.history['price'] = market.price_matrix[self.id]
