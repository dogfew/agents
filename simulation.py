import numpy as np
import pandas as pd
from scipy.special import softmax

from consumers.base_consumer import BaseConsumer
from market import Market
from simulation import Simulation
from firms import ComplexFirm, ComplexSellFirm, BaseFirm
import warnings

warnings.simplefilter('ignore')

economy_tech_matrix = np.array(
    [[0, 0.5, 0.3],
     [0.4, 0, 0.4],
     [0.002, 0.3, 0]]
)
economy_invest_matrix = np.array(
    [[0, 0.01, 0],
     [0, 0.1, 0],
     [0, 1, 0]]
)
n_industies, n_commodities = economy_tech_matrix.shape
economy_out_matrix = np.eye(n_industies) * 5
industries_dist = np.ones(n_industies)
industries_dist = industries_dist / industries_dist.sum()
industries_idx = np.arange(n_industies)
start_limits = lambda: 5
start_reserves = lambda: np.ones(n_industies) * 0
finances = lambda: 1
n_firms = 6
industry_types = np.tile(industries_idx, n_firms // n_industies + 1)[:n_firms]
firm_type = ComplexSellFirm


def main():
    market = Market(n_firms=n_firms, n_commodities=n_commodities)
    market._p_matrix = np.ones_like(market.price_matrix)
    firms = []
    for i in range(n_firms):
        industry_type = industry_types[i]
        firm = firm_type(tech_matrix=economy_tech_matrix[industry_type],
                         out_matrix=economy_out_matrix[industry_type],
                         invest_matrix=economy_invest_matrix[industry_type],
                         limit=start_limits(),
                         id_=i,
                         financial_resources=finances(), deprecation_steps=5,
                         is_deprecating=True)
        firm.reserves = start_reserves()
        firms.append(firm)
    consumers = []
    # consumers = [BaseConsumer(utility_matrix=np.array([0, 0, 1]), id_=0)]
    simulation = Simulation(market, firms, consumers)
    simulation.simulate(5)
    simulation.save()
    simulation.plot()
    for firm in firms:
        print(firm.production_resources)
        print(firm.investment_resources)
        print(firm.financial_resources)
        print()

#    simulate(market, firms, consumers, num_iters=n_iters, dir=r'history/main/')

if __name__ == '__main__':
    main()
    # import pandas as pd
    # print(pd.read_csv(r'history/main/prices.csv'))
    # print(pd.read_csv(r'history/main/firms_prices.csv').tail(1).T)
