import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.special import softmax


def generate_dict(n):
    return {k: [] for k in range(n)}


def gini_coefficient(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


class Simulation:
    def __init__(self,
                 market,
                 firms,
                 consumers,
                 stock_market=None,
                 base_income=lambda x: x * 10,
                 tech_percent=0,
                 tech_steps=1,
                 tech_abs=0,
                 random_seed=0):
        """
        :param market: рынок
        :param firms: фирмы
        :param consumers: потребители
        :param base_income: функция базового дохода
        :param random_seed:
        """
        self.firms = firms
        self.consumers = consumers
        self.market = market
        self.stock_market = stock_market
        self.base_income = base_income
        self.n_firms = self.market.n_firms
        self.n_commodities = self.market.n_commodities
        self.total_steps = 0
        self.history = dict(
            volumes=[],
            prices=[],
            limits=[],
            finance=[],
            gains=[],
            profitability=[],
            firms_profitability=generate_dict(len(firms)),
            firms_gains=generate_dict(len(firms)),
            firms_volumes=generate_dict(len(firms)),
            firms_dvolumes=generate_dict(len(firms)),
            firms_prices=generate_dict(len(firms)),
            firms_limits=generate_dict(len(firms)),
            firms_steps=generate_dict(len(firms)),
            firms_finance=generate_dict(len(firms)),
            consumers_gini=[],
            firms_hirshman=[],
            firms_gini=[]
        )
        self.agents = firms + consumers
        self.tech_percent = tech_percent
        self.tech_abs = tech_abs
        self.tech_steps = tech_steps
        self.rng = np.random.default_rng(random_seed)

    def simulate(self, num_iters, shuffle=True, show_pbar=True):
        """
        Запустить симуляцию
        :param num_iters: количество шагов
        :param shuffle: перемешивать ли порядок хода
        :param show_pbar: показывать ли progress-bar
        :return:
        """
        agents_range = np.arange(len(self.agents))
        pbar = tqdm(total=num_iters) if show_pbar else None
        for i in range(num_iters):
            self.step(agents_range)
            if show_pbar:
                pbar.set_description(f"\rШаг: {1 + i}/{num_iters} | Агентов: {len(self.agents)}")
                pbar.update()
            if shuffle:
                self.rng.shuffle(agents_range)

    def process_agent_step(self, agent):
        agent.step(self.market, self.stock_market)

    def step(self, agents_range=None, shuffle=True):
        """
        Сделать шаг для всех агентов
        :param agents_range: порядок шагов для каждого агента
        :return:
        """
        if agents_range is None:
            agents_range = np.arange(len(self.agents))
        if shuffle:
            self.rng.shuffle(agents_range)
        v_current = np.zeros_like(self.market.volume_matrix, dtype=np.float64)
        p_current = np.zeros_like(self.market.price_matrix, dtype=np.float64)
        l_current = np.zeros_like(self.market.price_matrix, dtype=np.float64)
        g_current = np.zeros_like(self.market.price_matrix, dtype=np.float64)
        f_current = np.zeros_like(self.market.price_matrix, dtype=np.float64)
        fixed_capital_current = np.zeros_like(self.market.price_matrix, dtype=np.float64)
        floating_capital_current = np.zeros_like(self.market.price_matrix, dtype=np.float64)
        for j, id_ in enumerate(agents_range):
            agent = self.agents[id_]
            self.process_agent_step(agent)

            if not hasattr(agent, 'tech_matrix'):
                agent.financial_resources += self.base_income(self.market.mean_prices).sum()
            else:
                v_current[id_] += self.market.volume_matrix[id_]
                p_current[id_] = self.market.mean_prices
                l_current[id_] += np.where(agent.out_matrix != 0, agent.limit, 0)
                g_current[id_] += np.where(agent.out_matrix != 0, agent.history['gains'], 0)
                f_current[id_] += np.where(agent.out_matrix != 0, agent.financial_resources, 0)

                fixed_capital = np.sum(agent.deprecation_array / agent.deprecation_steps)
                fixed_capital *= (self.market.mean_prices @ agent.invest_matrix)
                fixed_capital_current[id_] += np.where(agent.out_matrix != 0, fixed_capital, 0)
                floating_capital_current[id_] += np.where(agent.out_matrix != 0, agent.history['floating_capital'], 0)
                self.update_history(agent, id_, j)
        self.history['volumes'].append(v_current.sum(axis=0).round(4))
        self.history['dvolumes'] = np.diff(self.history['volumes'], axis=0, prepend=0)
        self.history['prices'].append(p_current.mean(axis=0))
        self.history['limits'].append(l_current.sum(axis=0))
        self.history['finance'].append(f_current.sum(axis=0))
        self.history['gains'].append(g_current.sum(axis=0))
        self.history['profitability'].append(
            np.minimum(1,
                       g_current.sum(axis=0) / (
                               floating_capital_current.sum(axis=0) + fixed_capital_current.sum(axis=0) + 1e-5
                       )))
        if self.stock_market is not None:
            incomes = self.stock_market.consumers_income
            income_gini = gini_coefficient(incomes)
            if np.isnan(income_gini) and self.total_steps > 0:
                self.history['consumers_gini'].append(self.history['consumers_gini'][-1])
            else:
                self.history['consumers_gini'].append(gini_coefficient(incomes))
        firms_gini = gini_coefficient(fixed_capital_current.sum(axis=1) + floating_capital_current.sum(axis=1))
        if np.isnan(firms_gini) and self.total_steps > 0:
            self.history['firms_gini'].append(self.history['firms_gini'][-1])
        else:
            self.history['firms_gini'].append(firms_gini)
        total_sales_percent = g_current.sum(axis=0)[None, :]
        sales_percents_square = np.where(total_sales_percent > 1e-10,
                                         (g_current / total_sales_percent) ** 2,
                                         0).round(4)
        new_hirshman = np.sqrt(np.sum(sales_percents_square, axis=0).round(2))
        if self.total_steps > 0:
            self.history['firms_hirshman'].append(np.where(new_hirshman != 0, new_hirshman,
                                                           self.history['firms_hirshman'][-1]))
        else:
            self.history['firms_hirshman'].append(new_hirshman)
        self.total_steps += 1
        if (self.tech_percent != 0 or self.tech_abs != 0) and (self.total_steps % self.tech_steps) == 0:
            for firm in self.firms:
                firm.out_matrix += firm.out_matrix * self.tech_percent / 100.
                firm.out_matrix += np.where(firm.out_matrix != 0, self.tech_abs, 0)
                firm.out_matrix = np.minimum(firm.out_matrix, 100)

    def update_history(self, agent, id_, j):
        h = agent.history
        self.history['firms_limits'][id_].append(agent.limit)
        self.history['firms_gains'][id_].append(h['gains'].round(4))
        self.history['firms_volumes'][id_].append(0 if h['volume'] is None else h['volume'].sum().round(4))
        self.history['firms_dvolumes'][id_] = np.diff(self.history['firms_volumes'][id_], axis=0, prepend=0)
        self.history['firms_prices'][id_].append(
            (np.nan_to_num(h['price']) * (agent.out_matrix != 0)).sum().round(4)
        )
        self.history['firms_steps'][id_].append(j)
        self.history['firms_finance'][id_].append(agent.financial_resources)
        fixed_capital = np.sum(agent.deprecation_array / agent.deprecation_steps)
        fixed_capital *= (self.market.mean_prices @ agent.invest_matrix)
        self.history['firms_profitability'][id_].append(
            np.minimum(agent.history['gains'] / (agent.history['floating_capital'] + fixed_capital + 1e-5), 1)
        )

    def save(self, dir=r'history/main/'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        columns = [f'Фирма {i}' for i in range(self.n_firms)]
        index = [f'Итерация {j + 1}' for j in range(len(self.history['limits'][0]))]
        volumes_df = pd.DataFrame(self.history['volumes'],
                                  columns=[f'Товар {i}' for i in range(self.n_commodities)])
        volumes_df.index = index
        volumes_df.to_csv(f'{dir}volumes.csv')
        prices_df = pd.DataFrame(self.history['prices'],
                                 columns=[f'Средняя цена {i}' for i in range(self.n_commodities)], )
        prices_df.index = index
        prices_df.to_csv(f'{dir}prices.csv')
        for k, v in self.history.items():
            if k in ['volumes', 'prices']:
                continue
            temp_df = pd.DataFrame(v).round(3)
            temp_df.columns = columns
            temp_df.to_csv(f"{dir}{k}.csv", index_label=False)

    def plot(self, dir=r'history/main/', save=True):

        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        pd.read_csv(f"{dir}prices.csv").plot(ax=ax[1][0], title='Prices')
        pd.read_csv(f"{dir}volumes.csv").plot(ax=ax[1][1], title='Volumes')
        pd.read_csv(f'{dir}limits.csv').plot(ax=ax[0][0], title='Limits')
        pd.read_csv(f'{dir}gains.csv').plot(ax=ax[0][1], title='Gains')
        pd.read_csv(f'{dir}financial_resources.csv').plot(ax=ax[0][2], title='Finance')
        temp_df = pd.read_csv(f'{dir}volumes.csv')
        temp_df[temp_df.columns[1:]] = temp_df[temp_df.columns[1:]].diff()
        temp_df.plot(ax=ax[1][2], title='delta Volumes')
        plt.show()
        if save:
            fig.savefig(f'{dir}res.png', dpi=200)


class SimulationGovernment(Simulation):
    """
    Симуляция с участием государства
    """

    def __init__(self,
                 market,
                 firms,
                 consumers,
                 stock_market=None,
                 base_income=lambda x: x * 10,
                 tech_percent=0,
                 tech_steps=1,
                 tech_abs=0,
                 random_seed=0,
                 profit_tax=0,
                 direct_tax_firm=0,
                 direct_tax_consumer=0,
                 profit_taxation_border=0.5,
                 direct_taxation_border=0.5,
                 soft_weights=True
                 ):
        """
        :param profit_tax: Доля налога на прибыль
        :param direct_tax_firm: Доля прямого налогообложения фирмы
        :param direct_tax_consumer: Доля прямого налогообложения потребителей
        :param taxation_border: Барьер для налогообложения
        """
        super().__init__(market,
                         firms,
                         consumers,
                         stock_market,
                         base_income,
                         tech_percent,
                         tech_steps,
                         tech_abs,
                         random_seed)
        self.profit_tax = profit_tax
        self.profit_taxation_border = profit_taxation_border
        self.direct_taxation_border = direct_taxation_border
        self.direct_tax_firm = direct_tax_firm
        self.direct_tax_consumer = direct_tax_consumer
        self.soft_weights = soft_weights

    def process_agent_step(self, agent):
        finances_array = np.array([firm.financial_resources for firm in self.firms]) + self.market.gains
        if self.soft_weights:
            weights = softmax(-finances_array, axis=0)
        else:
            weights = np.full_like(finances_array, 1 / self.n_firms, dtype=np.float64)
        total_finance = finances_array.sum()
        """Если агент - фирма"""
        if hasattr(agent, 'tech_matrix'):
            agent_share = agent.financial_resources / total_finance
            if agent_share >= self.profit_taxation_border:
                agent_gains = self.market.gains[agent.id]
                taxed = agent_gains * self.profit_tax
                self.market.gains[agent.id] -= taxed
                self.market.gains += taxed * weights
            if agent_share >= self.direct_taxation_border:
                taxed = agent.financial_resources * self.direct_tax_firm
                agent.financial_resources -= taxed
                self.market.gains += taxed * weights
        agent.step(self.market, self.stock_market)
