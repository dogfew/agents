import numpy as np


class Market:
    def __init__(self, n_firms: int, n_commodities: int,
                 min_price: float=0.01,
                 max_price: float=100):
        """
        :param n_firms: число фирм
        :param n_commodities: число видов товаров
        """
        self.n_firms: int = n_firms
        self.n_commodities: int = n_commodities

        self._v_matrix: np.ndarray = np.zeros((n_firms, n_commodities), dtype=np.float64)
        self._p_matrix: np.ndarray = np.zeros((n_firms, n_commodities), dtype=np.float64)
        self.gains: np.ndarray = np.zeros(n_firms, dtype=np.float64)
        self.MIN_PRICE = min_price
        self.MAX_PRICE = max_price

    @property
    def volume_matrix(self):
        return self._v_matrix.round(5)

    @property
    def price_matrix(self):
        return np.nan_to_num(self._p_matrix).round(5)

    @property
    def mean_prices(self):
        weights = self.volume_matrix
        weights /= weights.sum(axis=0)
        res = np.nansum(self._p_matrix * weights, axis=0)
        return res

    @property
    def total_volumes(self):
        return self._v_matrix.sum(axis=0)

    def add_firms(self, n_firms: int = 1):
        """
        Добавить новые фирмы на рынок (экстенсивный рост)

        :param n_firms: число новых фирм
        """
        self.n_firms += n_firms
        self._v_matrix = np.vstack(self._v_matrix, np.zeros(n_firms, self.n_commodities))
        self._p_matrix = np.vstack(self._p_matrix, np.zeros(n_firms, self.n_commodities))

    def add_commodities_types(self, n_commodities: int = 1):
        """
        Добавить новые виды товаров на рынок (экстенсивный рост)

        :param n_commodities: сколько новых видов товаров будет

        """
        self.n_commodities += n_commodities
        self._v_matrix = np.vstack(self._v_matrix, np.zeros(self.n_commodities, n_commodities))
        self._p_matrix = np.vstack(self._p_matrix, np.zeros(self.n_commodities, n_commodities))

    def process_purchases(self,
                          purchase_matrix: np.ndarray,
                          sellers_gains: np.ndarray):
        """
        Обновить параметры рынка после покупок

        :param purchase_matrix: Матрица с объёмами купленного
        :param sellers_gains: Выигрыши продавцов
        :return:
        """
        self._v_matrix -= purchase_matrix
        self.gains += sellers_gains
        assert np.all(self._v_matrix >= - 0.0001), f"Ошибка в коде! Где-то купили больше товаров, чем можно было" \
                                                   f"\nVolumes: {self.volume_matrix}"

    def process_sales(self,
                      firm_id: int,
                      volumes: np.ndarray,
                      ):
        """
        Добавить новые товары на рынок (интенсивный рост)

        :param firm_id: id фирмы (номер строки)
        :param volumes: объёмы товаров
        :return:
        """

        assert firm_id < self.n_firms, "Недопустимый firm_id"
        self._v_matrix[firm_id] += volumes
        assert np.all(self._v_matrix >= -1e-3), f"Ошибка в коде! Где-то купили больше товаров, чем можно было" \
                                                f"\nVolumes: {self.volume_matrix}"

    def process_prices(self,
                       firm_id: int,
                       prices: np.ndarray):
        """
        Добавить новые товары на рынок (интенсивный рост)

        :param firm_id: id фирмы (номер строки)
        :param prices: цены товаров
        :return:
        """

        assert firm_id < self.n_firms, "Недопустимый firm_id"
        self._p_matrix[firm_id] = prices

    def process_gains(self, firm_id):
        firms_gain = self.gains[firm_id].copy()
        self.gains[firm_id] -= firms_gain
        return firms_gain

    def __repr__(self):
        representation = f"Market (n_firms: {self.n_firms}, n_commodities: {self.n_commodities})" \
                         f"\n\tVolume Matrix:\n{self._v_matrix}" \
                         f"\n\tPrice Matrix:\n{self._p_matrix}" \
                         f"\n\tGain Matrix:\n{self._gains}"
        return representation


class StockMarket:
    def __init__(self, n_firms: int, n_consumers: int):
        self.n_firms: int = n_firms
        self.n_consumers: int = n_consumers

        self.__profits = np.zeros((n_firms, n_consumers))
        self.__shareholders = np.zeros((n_firms, n_consumers))

    def process_gains(self, consumer_id):
        gains = self.__profits[:, consumer_id]
        self.__profits[:, consumer_id] = 0
        return gains.sum()

    def add_profit(self, firm_id):
        pass
