from matplotlib import pyplot as plt
from shiny import App, reactive, render, ui
import numpy as np
import pandas as pd
from shinywidgets import register_widget
from consumers.base_consumer import BaseConsumer
from market import Market
from simulation import Simulation, SimulationGovernment
from firms import ComplexSellFirm, ComplexFirm, RegulatedFirm
from io import StringIO
from asyncio import sleep
import json
from warnings import simplefilter
from app_ui import APP_UI
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import plotly.express as px

simplefilter('ignore')


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    np.set_printoptions(suppress=True)

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app_ui = APP_UI


def server(input, output, session):
    economy_tech_matrix = np.array([[]])
    economy_invest_matrix = np.array([[]])
    simulation = Simulation(Market(n_firms=1, n_commodities=1), [], [])

    @output
    @render.table
    def num_firms():
        file_text = StringIO(input.num_firms())
        data = pd.read_csv(file_text)
        firms_dist.set(np.array(data).flatten())
        data.columns = pd.MultiIndex.from_tuples([('Количество фирм в секторе', i) for i in data.columns])
        return data

    @output
    @render.table
    def tech_matrix():
        file_text = StringIO(input.tech_matrix())
        data = pd.read_csv(file_text)
        tech_matrix.set(np.array(data))
        prod_names.set(list(data.columns))
        data.columns = pd.MultiIndex.from_tuples([('Технологическая матрица', i) for i in data.columns])
        return data

    @output
    @render.table
    def utility():
        file_text = StringIO(input.utility())
        data = pd.read_csv(file_text)
        utility_reactive.set(np.array(data))
        data.columns = pd.MultiIndex.from_tuples([('Полезность', i) for i in data.columns])
        return data

    @output
    @render.table
    def target_price_vector():
        file_text = StringIO(input.target_price_vector())
        data = pd.read_csv(file_text)
        target_price_vector_reactive.set(np.array(data))
        data.columns = pd.MultiIndex.from_tuples([('Цена', i) for i in data.columns])
        return data

    @output
    @render.table
    def invest_matrix():
        file_text = StringIO(input.invest_matrix())
        data = pd.read_csv(file_text)
        invest_matrix.set(np.array(data))
        data.columns = pd.MultiIndex.from_tuples([('Инвестиционная матрица', i) for i in data.columns])
        return data

    configurated = reactive.Value(False)
    output_multiplier = reactive.Value(1)
    utility_reactive = reactive.Value(np.array([0, 0, 1]))
    firms_dist = reactive.Value(np.array([]))
    invest_matrix = reactive.Value(economy_invest_matrix)
    tech_matrix = reactive.Value(economy_tech_matrix)
    simulation_reactive = reactive.Value(simulation)
    history = reactive.Value(simulation.history.copy())
    step_timeout = reactive.Value(0)
    clever_firms = reactive.Value(True)
    n_consumers = reactive.Value(1)
    prod_names = reactive.Value(['Товар A', 'Товар B', 'Товар C'])
    target_price_vector_reactive = reactive.Value(np.array([1, 2, 3]))

    """plotly виджет"""
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=(
                            'Средне-взвешенные цены', 'Объёмы', 'Изменения объёмов', 'Лимиты', 'Финансы', 'Выигрыши'), )
    for i in range(1, 3):
        for j in range(1, 4):
            fig.add_scatter(y=[], row=i, col=j)

    fig.update_layout(height=1000, width=1700, title_text='Секторальные графики', paper_bgcolor='white',
                      plot_bgcolor='white')
    fig.update_xaxes(gridcolor='black')
    fig.update_yaxes(gridcolor='black')
    plotly_widget = go.FigureWidget(fig)
    register_widget("plotly_widget", plotly_widget)

    def make_step():
        if not input.y():
            return
        if not configurated():
            ui.notification_show('Прежде, чем запускать модель, нажмите кнопку Сконфигурировать', type='error')
            ui.update_switch('y', value=False)
            return

        sps = input.steps_per_second()
        with reactive.isolate():
            step_timeout.set(step_timeout() + sps)
            if step_timeout() < 30:
                return
        simulation = simulation_reactive()
        simulation.step(shuffle=input.shuffle())
        update_widget(simulation)
        simulation_reactive.set(simulation)
        history.set(simulation.history.copy())

        with reactive.isolate():
            step_timeout.set(0)

    reactive.poll(make_step, interval_secs=1 / 30)

    @reactive.Effect
    @reactive.event(input.window)
    def update_window():
        simulation_ = simulation_reactive()
        if input.y() or simulation_.total_steps == 0:
            return
        update_widget(simulation_)

    def update_widget(simulation):
        window_size = input.window()
        with plotly_widget.batch_update():
            i = 0
            idx = np.arange(simulation.total_steps)
            for col in ['prices', 'volumes', 'dvolumes', 'limits', 'finance', 'gains']:
                for lst in np.array(simulation.history[col]).T:
                    plotly_widget.data[i].y = lst
                    plotly_widget.data[i].x = idx
                    i += 1
            for col in ['prices', 'volumes', 'dvolumes', 'limits', 'finance', 'gains']:
                for lst in np.array(simulation.history[col]).T:
                    plotly_widget.data[i].y = np.convolve(lst, np.ones(window_size) / window_size, mode='valid').round(4)
                    plotly_widget.data[i].x = idx
                    i += 1
            #
            # for col in np.array(simulation.history['volumes']).T:
            #     plotly_widget.data[i].y = col
            #     plotly_widget.data[i].x = idx
            #     i += 1
            #
            # for col in np.array(simulation.history['dvolumes']).T:
            #     plotly_widget.data[i].y = col
            #     plotly_widget.data[i].x = idx
            #     i += 1
            # for col in np.array(simulation.history['limits']).T:
            #     plotly_widget.data[i].y = col
            #     plotly_widget.data[i].x = idx
            #     i += 1
            # for firm in simulation.firms:
            #     plotly_widget.data[i].y = list(plotly_widget.data[i].y) + [firm.limit]
            #     plotly_widget.data[i].x = idx
            #     i += 1
            # for firm in simulation.firms:
            #     plotly_widget.data[i].y = list(plotly_widget.data[i].y) + [firm.financial_resources]
            #     plotly_widget.data[i].x = idx
            #     i += 1
            # for firm in simulation.firms:
            #     plotly_widget.data[i].y = list(plotly_widget.data[i].y) + [firm.history['gains']]
            #     plotly_widget.data[i].x = idx
            #     i += 1

    @reactive.Effect
    @reactive.event(input.step)
    async def make_n_steps():
        """СДЕЛАТЬ ШАГИ МОДЕЛИ!!!"""
        n_steps = input.x()
        if n_steps is None:
            m = ui.modal(
                f"Шаг, который указан вами, неправильный. Попробуйте, например, 10.",
                title="Недопустимый шаг",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
            return

        if configurated() == 0:
            m = ui.modal(
                f"Прежде, чем запускать модель, нажмите кнопку 'Сконфигурировать'",
                title="Не сконфигурирована модель!",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
            return
        simulation = simulation_reactive()
        with ui.Progress(min=0, max=n_steps) as p:
            p.set(message="Делаем шаги", detail="Делаем шаги модели.")
            for i_ in range(n_steps):
                p.set(i_, message=f"{i_ + 1}/{n_steps}")
                await sleep(0)
                simulation.step(shuffle=input.shuffle())
        update_widget(simulation)
        simulation_reactive.set(simulation)
        history.set(simulation_reactive().history.copy())

    @reactive.Effect
    @reactive.event(input.add_reserves)
    def update_reserves():
        n_reserves = input.add_reserves_value()
        if n_reserves is None:
            m = ui.modal(
                f"Резервы, которые вы раздаёте, должны быть числом. Попробуйте, например, 10.",
                title="Недопустимоое количество резервов",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
            return
        simulation = simulation_reactive()
        for firm in simulation.firms:
            firm.reserves += n_reserves
        simulation_reactive.set(simulation)
        history.set(simulation_reactive().history.copy())

    @reactive.Effect
    @reactive.event(input.add_finances)
    def update_finances():
        n_reserves = input.add_finances_value()
        if n_reserves is None:
            m = ui.modal(
                f"Деньги, которые вы раздаёте, должны быть числом. Попробуйте, например, 10.",
                title="Недопустимоое количество денег",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
            return
        simulation = simulation_reactive()
        for firm in simulation.firms:
            firm.production_resources += n_reserves * firm.production_rate
            firm.investment_resources += n_reserves * (1 - firm.production_rate)
            firm.financial_resources += n_reserves
        simulation_reactive.set(simulation)
        history.set(simulation_reactive().history.copy())

    @output
    @render.ui
    def ui_select():
        return ui.input_checkbox_group(
            "display_firms",
            label=f"Выберите фирмы",
            choices=[i for i in range(len(simulation_reactive().firms))],
            selected=None,
            inline=True,
        )

    @output
    @render.text
    def display_firm():
        history()
        firms = simulation_reactive().firms
        f = lambda i: str(firms[int(i)]) + (
                    '\n' + json.dumps(firms[int(i)].history, indent=4, cls=NumpyEncoder)) * input.json()
        out = '\n\n'.join(map(f, input.display_firms()))
        return out

    @output
    @render.plot(alt="plots")
    def plots():
        sumulation = simulation_reactive()
        history_ = history()
        columns = [f'Фирма {firm.id} ({firm.prod_name})' for firm in sumulation.firms]

        fig, ax = plt.subplots(2, 3, figsize=(30, 15))
        volumes_df = pd.DataFrame(history_.get('volumes'),
                                  columns=prod_names())
        prices_df = pd.DataFrame(history_.get('prices'),
                                 columns=prod_names())
        temp_df = volumes_df.copy().diff()
        limits_df = pd.DataFrame(history_.get('firms_limits'))
        gains_df = pd.DataFrame(history_.get('firms_gains'))
        finance_df = pd.DataFrame(history_.get('firms_finance'))

        limits_df.columns = columns
        gains_df.columns = columns
        finance_df.columns = columns

        volumes_df.plot(ax=ax[1, 1], title='Объёмы')
        limits_df.plot(ax=ax[0, 0], title='Лимиты')
        finance_df.plot(ax=ax[0, 2], title='Финансы')
        prices_df.plot(ax=ax[1, 0], title='Средние цены')
        gains_df.plot(ax=ax[0, 1], title='Выигрыши')
        temp_df.plot(ax=ax[1, 2], title='Изменения объёмов', alpha=0.5)

    @reactive.Effect
    @reactive.event(input.start, ignore_init=True)
    def change_simulation():
        try:
            rng = np.random.default_rng(0)
            economy_tech_matrix = tech_matrix()
            assert np.all(economy_tech_matrix >= 0), "В матрице Леонтьева нельзя использовать отрицательные числа"
            economy_invest_matrix = invest_matrix()
            assert np.all(economy_invest_matrix >= 0), "В матрице Леонтьева нельзя использовать отрицательные числа"
            n_industies, n_commodities = economy_tech_matrix.shape
            economy_out_matrix = np.eye(n_industies) * output_multiplier()
            assert np.all(firms_dist() >= 0), "Количество фирм не может быть отрицательным"
            assert economy_tech_matrix.shape == economy_invest_matrix.shape, "Количество отраслей во всех матрицах должно быть одинаковым"
            assert firms_dist().shape[0] == economy_invest_matrix.shape[0], "Неверное количество фирм по отраслям"
            firms = []
            i = 0
            if input.set_target_price() and input.regulation():
                firm_type = RegulatedFirm
            else:
                firm_type = ComplexSellFirm if clever_firms() else ComplexFirm
            print(firm_type)
            limits = np.arange(input.start_limits()[0], input.start_limits()[1] + 1)
            reserves = np.arange(input.start_reserves()[0], input.start_reserves()[1] + 1)
            finances = np.arange(input.start_finances()[0], input.start_finances()[1] + 1)
            sales_percents = np.arange(input.sales_percents()[0], input.sales_percents()[1] + 1) / 100
            production_rates = np.arange(input.production_percent()[0], input.production_percent()[1] + 1) / 100
            profit_rates = np.arange(input.profit_rates()[0], input.profit_rates()[1] + 1) / 100
            iterable = zip(economy_tech_matrix, economy_out_matrix, economy_invest_matrix, firms_dist(), prod_names())
            for current_tech, current_out, current_invest, current_firms_number, prod_name in iterable:
                for j in range(current_firms_number):
                    firm = firm_type(tech_matrix=current_tech,
                                     out_matrix=current_out,
                                     invest_matrix=current_invest,
                                     limit=rng.choice(limits),
                                     id_=i,
                                     financial_resources=rng.choice(finances),
                                     is_deprecating=input.deprecation(),
                                     deprecation_steps=input.deprecation_steps(),
                                     min_limit=input.min_limit(),
                                     sale_percent=rng.choice(sales_percents),
                                     production_rate=rng.choice(production_rates),
                                     profit_rate=rng.choice(profit_rates) * int(input.finmarket()),
                                     prod_name=prod_name,
                                     invest_anyway=input.invest_anyway()
                                     )
                    firm.reserves = np.full(n_commodities, rng.choice(reserves), dtype=np.float64)
                    firms.append(firm)
                    i += 1
            market_ = Market(n_firms=len(firms),
                             n_commodities=n_commodities,
                             min_price=input.MIN_PRICE(),
                             max_price=input.MAX_PRICE()
                             )
            market_._p_matrix = np.ones_like(market_.price_matrix)
            if input.set_target_price():
                market_._p_matrix *= target_price_vector_reactive()
            consumers_ = []
            if input.consumers():
                consumers_ = [BaseConsumer(utility_matrix=utility_reactive().flatten(), id_=j)
                              for j in range(n_consumers())]
            if input.set_taxation():
                simulation_ = SimulationGovernment(
                    market_,
                    firms,
                    consumers_,
                    base_income=lambda x: x * input.base_income(),
                    profit_tax=input.profit_tax() / 100,
                    direct_tax_firm=input.direct_tax_firm() / 100,
                    direct_taxation_border=input.direct_taxation_border() / 100,
                    profit_taxation_border=input.profit_taxation_border() / 100,
                    soft_weights=input.soft_weights()
                )
            else:
                simulation_ = Simulation(market_, firms, consumers_, base_income=lambda x: x * input.base_income())
            simulation_reactive.set(simulation_)
            history.set(simulation_reactive().history.copy())
            ui.notification_show("Модель успешно сконфигурирована!", type='message')
            configurated.set(True)
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            with plotly_widget.batch_update():
                plotly_widget.data = []
                for i, x in enumerate([f"Цены", "Объёмы", "Δ Объёмов"]):
                    for col, color in zip(prod_names(), colors):
                        plotly_widget.add_trace(go.Scatter(x=[], y=[], name=f"{col}",
                                                           visible='legendonly',
                                                           showlegend=i == 0,
                                                           legendgroup=f'{col}',
                                                           line=go.scatter.Line(dash='dash'),
                                                           marker={'color': color}), row=1, col=i + 1)
                for i, x in enumerate([f"Лимиты", "Финансы", "Выручка"]):
                    for col, color in zip(prod_names(), colors):
                        plotly_widget.add_trace(go.Scatter(x=[], y=[], name=f"{col}",
                                                           visible='legendonly',
                                                           showlegend=False,
                                                           legendgroup=f"{col}",
                                                           line=go.scatter.Line(dash='dash'),
                                                           marker={'color': color}), row=2, col=i + 1)

                for i, x in enumerate([f"Цены", "Объёмы", "Δ Объёмов"]):
                    for col, color in zip(prod_names(), colors):
                        plotly_widget.add_trace(go.Scatter(x=[], y=[], name=f"MA {col}", legendgroup=f'MA {col}',
                                                           showlegend=i == 0,
                                                           marker={'color': color},
                                                           ), row=1, col=i + 1)
                for i, x in enumerate([f"Лимиты", "Финансы", "Выручка"]):
                    for col, color in zip(prod_names(), colors):
                        plotly_widget.add_trace(go.Scatter(x=[], y=[], name=f"MA {col}", legendgroup=f"MA {col}",
                                                           showlegend=False,
                                                           marker={'color': color}), row=2, col=i + 1)
        except Exception as e:
            ui.notification_show(f"Модель не удалось сконфигурировать! {e}", type='error')

    @reactive.Effect
    @reactive.event(input.klever)
    def process_clever():
        clever_firms.set(input.klever())

    @reactive.Effect
    @reactive.event(input.output_multiplier)
    def process_multiplier():
        output_multiplier.set(input.output_multiplier())

    @reactive.Effect
    @reactive.event(input.n_consumers)
    def process_consumers():
        n_consumers.set(input.n_consumers())

    @output
    @render.table(index=True)
    def market_prices():
        history()
        df = pd.DataFrame(simulation_reactive().market.price_matrix)
        df.loc[-1] = simulation_reactive().market.mean_prices
        df.index = [f'Фирма {i}' for i in df.index[:-1]] + ['В среднем']
        df.columns = pd.MultiIndex.from_tuples([('Цены', f'{i}') for i in prod_names()])
        return df

    @output
    @render.table(index=True)
    def market_volumes():
        history()
        df = pd.DataFrame(simulation_reactive().market.volume_matrix)
        df.loc[-1] = df.sum(axis=0)
        df.index = [f'Фирма {i}' for i in df.index[:-1]] + ['Итого']
        df.columns = pd.MultiIndex.from_tuples([('Объёмы', f'{i}') for i in prod_names()])
        return df

    @output
    @render.table(index=True)
    def finance():
        history()
        simulation = simulation_reactive()
        firms_ = simulation.firms
        consumers_ = simulation.consumers
        gains = simulation_reactive().market.gains
        df = pd.DataFrame([firm.financial_resources for firm in firms_] +
                          [consumer.financial_resources for consumer in consumers_])
        df = df.assign(limits=[firm.limit for firm in firms_] + [None for _ in consumers_])
        df = df.assign(reserves=[firm.reserves.sum() for firm in firms_] + [None for _ in consumers_])
        df = df.assign(gains=list(gains) + [None for _ in consumers_])
        df = df.T.assign(total=np.nansum(df, axis=0)).T
        df.index = [f'Фирма {i}' for i in range(len(firms_))] + \
                   [f'Потребитель {i}' for i in range(len(consumers_))] + \
                   ['Итого']
        df.columns = ['Финансы', 'Лимиты', 'Суммарные запасы', 'Выручка']
        return df.round(3)


app = App(app_ui, server)
