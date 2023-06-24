import seaborn as sns
from shiny import ui
from shinywidgets import output_widget, register_widget
import shinyswatch
from warnings import simplefilter

# simplefilter('ignore')
sns.set_style("whitegrid")
sns.color_palette('pastel')
sns.set_context('talk', font_scale=0.8)

DARK = False
if DARK:
    sns.set(rc={'axes.facecolor': '#222222',
                'figure.facecolor': '#222222',
                'xtick.color': 'white', 'ytick.color': 'white',
                'text.color': 'white'})
    theme = shinyswatch.theme.darkly
else:
    theme = shinyswatch.theme.minty
APP_UI = ui.page_fluid(
    theme(),
    ui.h1("Моделирование динамики экономики в зависимости от производственных технологий"),
    ui.h4("Запустить модель"),
    ui.div(
        {'style': 'display: flex;margin-bottom: 1%;'},
        ui.div(
            {'style': 'margin-right: 2%;'},
            ui.input_switch('y', 'Автоматически делать шаги', False),
            ui.panel_conditional('input.y',
                                 ui.input_slider('steps_per_second', 'Шагов в секунду', 1, 30, value=10)
                                 )
        ),
        ui.div(
            {'style': 'margin-right: 2%;'},
            ui.input_numeric("x", "Количество шагов:", 10),
            ui.input_action_button("step", "Сделать шаги!")),
        ui.div(
            {'style': 'margin-right: 2%;'},
            ui.input_numeric("add_reserves_value", "Объём запасов:", 0),
            ui.input_action_button("add_reserves", "Раздать ресурсы!", style='margin-right: 1%;')
        ),
        ui.div(
            ui.input_numeric("add_finances_value", "Объём денег:", 0),
            ui.input_action_button("add_finances", "Напечатать деньги!", style='margin-right: 1%;')
        )

    ),
    ui.h4("Меню"),
    ui.navset_tab(
        ui.nav("Конфигурация",
               ui.div(
                   {'style': 'display: flex;margin-top:1%;'},

                   ui.input_action_button("start", "Сконфигурировать"),
               ),
               ui.div(
                   {'style': 'display: flex;margin-top:1%;'},
                   ui.div(
                       ui.input_text_area("tech_matrix", "Tехнологическая матрица",
                                          value="a, b\n0, 1\n1, 0\n",
                                          height='150px'),
                       ui.output_table("tech_matrix"),
                   ),
                   ui.div(
                       {'style': 'margin-left: 1%;'},
                       ui.input_text_area("invest_matrix", "Инвестиционная матрица",
                                          value="a, b\n0, 1\n1, 0\n",
                                          height='150px'),
                       ui.output_table("invest_matrix"),
                   ),
                   ui.div(
                       {'style': 'margin-left: 1%;'},
                       ui.input_text_area("num_firms", "Кол-во  фирм",
                                          value="a, b\n2, 2",
                                          height='150px'),
                       ui.output_table("num_firms"),
                   )),
               ui.div(
                   {'style': 'display: flex;margin-left:1%;margin-top:2%;'},
                   ui.div(
                       {'style': 'display: inline-block;margin-left:1%;'},
                       ui.input_switch("klever", "Гладкие цены", True),
                       ui.input_switch("shuffle", "Перемешивать порядок ходов"),
                       ui.input_slider("output_multiplier", "Множитель выпуска", value=1.5, min=1, max=10, step=0.25),
                       ui.input_switch('advanced_sliders', "Продвинутые настройки фирм", value=False),
                       ui.panel_conditional('input.advanced_sliders',
                                            ui.input_slider("start_limits", "Изначальные лимиты", value=(5, 5), min=0,
                                                            max=20),
                                            ui.input_slider("start_reserves", "Изначальные запасы", value=(5, 5), min=0,
                                                            max=20),
                                            ui.input_slider("start_finances", "Изначальные финансы", value=(5, 5),
                                                            min=0, max=100, step=5),
                                            ui.input_slider("sales_percents", "Доля резервов на продажу",
                                                            value=(100, 100), min=0, max=100),
                                            ui.input_slider("production_percent", "Доля финансов на производство",
                                                            value=(80, 80), min=0, max=100),
                                            ui.input_switch("invest_anyway", "Не ограничивать объём инвестиций")

                                            ),
                       ui.input_switch('advanced_sliders_market', "Продвинутые настройки рынка", value=False),
                       ui.panel_conditional('input.advanced_sliders_market',
                                            ui.input_numeric("MIN_PRICE", "Минимальные цены на рынке:", 0.01),
                                            ui.input_numeric("MAX_PRICE", "Максимальные цены на рынке:", 10)
                                            )

                   ),
                   ui.div(
                       {'style': 'display: block;margin-left:1%;'},
                       ui.input_switch("deprecation", "Амортизация"),
                       ui.panel_conditional(
                           "input.deprecation",
                           ui.input_slider("deprecation_steps", "Время жизни предприятия", min=1, max=30, value=2),
                           ui.input_slider("min_limit", "Минимальный лимит", min=0, max=5, value=0),

                       ),
                   ),
                   ui.div(
                       {'style': 'display: block;margin-left:1%;'},
                       ui.input_switch("consumers", "Потребители", value=False),
                       ui.panel_conditional('input.consumers',
                                            ui.input_slider("n_consumers", "Количество потребителей", value=2, min=0,
                                                            max=10),
                                            ui.input_slider("base_income", "Базовый доход (в благах)",
                                                            min=0, max=100, value=0),
                                            ui.input_slider("base_income", "Уровень удовлетворения",
                                                            min=0, max=100, value=0),
                                            ui.div(
                                                {'style': 'margin-left: 1%; display: flex;'},
                                                ui.input_text_area("utility", "Функция полезности потребителей",
                                                                   value="a, b\n0, 1",
                                                                   height='80px'),
                                                ui.output_table("utility"),
                                            ),
                                            ui.input_switch("finmarket", "Собственность", value=False),
                                            ui.panel_conditional('input.finmarket',
                                                                 ui.input_slider("profit_rates", "Отчисления в прибыль",
                                                                                 min=0, max=100, value=(15, 15)),
                                                                 ui.input_slider("mpc", "MPC",
                                                                                 min=0, max=100, value=(70, 70))
                                                                 ),
                                            ),
                   ),
                   ui.div(
                       {'style': 'display: block;margin-left:1%;'},
                       ui.input_switch("regulation", "Регулирование", value=False),
                       ui.panel_conditional('input.regulation',
                                            ui.input_switch('set_target_price', 'Установить жесткие цены'),
                                            ui.panel_conditional('input.set_target_price',
                                                                 ui.div(
                                                                     ui.input_text_area("target_price_vector",
                                                                                        "Вектор цен",
                                                                                        value="a, b\n1, 1",
                                                                                        height='80px'),
                                                                     ui.output_table("target_price_vector"),
                                                                 ),
                                                                 ),
                                            ui.input_switch('set_taxation', 'Налоги'),
                                            ui.panel_conditional('input.set_taxation',
                                                                 ui.div(
                                                                     ui.input_slider("profit_taxation_border",
                                                                                     "Барьер для налога на прибыль",
                                                                                     value=25, min=0, max=100),
                                                                     ui.input_slider("profit_tax",
                                                                                     "% налога на прибыль",
                                                                                     value=25, min=0, max=100),
                                                                     ui.input_slider("direct_taxation_border",
                                                                                     "Барьер для прямого налога для фирм",
                                                                                     value=50, min=0, max=100),
                                                                     ui.input_slider("direct_tax_firm",
                                                                                     "% прямого налога",
                                                                                     value=50, min=0, max=100),
                                                                     ui.input_switch('soft_weights',
                                                                                     'SoftMax распределение',
                                                                                     value=True),

                                                                 ),
                                                                 )
                                            ),
                   ),
               ),
               ),
        ### графики
        ui.nav("Графики",
               ui.panel_main(
                   ui.output_plot("plots", height='800px'),
               ),
               ),
        ### графики
        ui.nav("Секторальные графики",
               ui.div(
                   # {"style": "display: flex; margin-left: 25%"},
                   ui.input_slider('window', 'Размер окна скользящих средних', min=1, max=10, value=1),
               ),
               # ui.panel_main(
               output_widget("plotly_widget", height='100%'),
               # ),
               ),
        ui.nav('Состояние рынка',
               ui.div(
                   {"style": 'display: flex;'},
                   ui.output_table("market_volumes", style='margin-right: 3%'),
                   ui.output_table("market_prices", style='margin-right: 3%'),
                   ui.output_table("finance", style='margin-right: 3%')
               ),
               ui.panel_conditional('input.finmarket',
                                    ui.div(
                                        {"style": 'display: flex;'},
                                        ui.output_table("property_table", style='margin-right: 3%'),
                                        ui.output_table("property_table_shares", style='margin-right: 3%'),
                                    )
                                    )
               )
        ,
        ui.nav(
            "Фирмы",
            ui.div(
                ui.output_ui("ui_select"),
                ui.input_switch('json', "JSON")),
            ui.output_text_verbatim("display_firm")

        )
        ,
    )
)
