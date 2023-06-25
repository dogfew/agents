import seaborn as sns
from shiny import ui
from shinywidgets import output_widget
import shinyswatch

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
    ui.head_content(
        ui.tags.script(
            src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
        ),
        ui.tags.script(
            "if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub]);"
        ),
    ),
    theme(),
    ui.h1("Моделирование динамики экономики и неравенства в зависимости от производственных технологий"),
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
                                            ui.input_switch("invest_anyway", "Не ограничивать объём инвестиций", True)

                                            ),
                       ui.input_switch('advanced_sliders_market', "Продвинутые настройки рынка", value=False),
                       ui.panel_conditional('input.advanced_sliders_market',
                                            ui.input_numeric("MIN_PRICE", "Минимальные цены на рынке:", 0.01),
                                            ui.input_numeric("MAX_PRICE", "Максимальные цены на рынке:", 10)
                                            ),
                       ui.input_switch('tech_progress', "Технологический прогресс", value=False),
                       ui.panel_conditional('input.tech_progress',
                                            ui.input_slider("tech_percent", "% прироста производительности",
                                                            value=0, min=0, max=10),
                                            ui.input_slider("tech_abs", "Абсолютный прирост производительности",
                                                            value=0, min=0, max=1, step=0.05),
                                            ui.input_slider("tech_steps", "Количество шагов до прироста",
                                                            value=1, min=1, max=100),

                                            )

                   ),
                   ui.div(
                       {'style': 'display: block;margin-left:1%;'},
                       ui.input_switch("deprecation", "Амортизация"),
                       ui.panel_conditional(
                           "input.deprecation",
                           ui.input_slider("deprecation_steps", "Время жизни основного капитала", min=1, max=30,
                                           value=2),
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
                                                                 ui.input_slider("mpc", "Доля дохода, тратящаяся на потребление",
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
               ui.input_slider('window', 'Размер окна скользящих средних', min=1, max=10, value=1),
               output_widget('a'),
               ui.navset_tab(
                   ui.nav('Отраслевые', output_widget("plotly_widget"), ),
                   ui.nav('По фирмам', output_widget("plotly_widget2"), ),
                   ui.nav('Неравенство', output_widget("plotly_widget3"), )

               ),
               ),
        ui.nav('Состояние рынка',
               ui.div(
                   {"style": 'display: flex;'},
                   ui.output_table("market_volumes", style='margin-right: 3%'),
                   ui.output_table("market_prices", style='margin-right: 3%'),
                   ui.output_table("finance", style='margin-right: 3%')
               ),
               ui.panel_conditional('input.finmarket & input.consumers',
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
        , ui.nav(
            "Описание",
            ui.div(
                {"style": 'margin-top: 1%;margin-right:1%;'},
                ui.h3("Введение"),
                ui.p("""
В данной работе моделируется экономика, где производство задано с помощью матрицы Леонтьева, и есть фирмы, распределённые
по разным отраслям в этой матрице. Эта матрица, как и число фирм в каждой отрасли, задаётся при инициализации модели, и 
может быть любого размера, однако большее число фирм и отраслей приводит к большему времени вычислений. 
Среда, где фирмы взаимодействуют между собой, в данной модели называется рынком. 
Опционально в модели может фигурировать другой вид агентов: потребители, функция которых состоит в 
непроизводительном потреблении и инвестициях в фирмы. Инвестиции в фирмы осуществляются через рынок ценных бумаг. 

Чтобы модель заработала, фирмы изначально должны иметь определенное количество запасов, из которых осуществят производство.
"""),
                ui.h3("Рынок"),
                ui.p("""
Рынок представляет из себя среду, где фирмы взаимодействуют между собой. Там хранятся произведённые каждой фирмой
товары в матрице 
\[ V \\in \\mathbb{R}^{\\text{n_firms} \\times \\text{n_commodities}} \]
а также назначенные каждой фирмой цены в матрице
\[ P \\in \\mathbb{R}^{\\text{n_firms} \\times \\text{n_commodities}} \]

Таким образом, цены на рынке индивидуальны для каждой фирмы. Также на рынке сосредоточена выручка фирм: она увеличивается
в зависимости от стоимости проданных фирмой благ, и в начале каждого хода фирма получает эту выручку. 
"""),
                ui.h3("Фирмы"),
                ui.p("""
Фирма - это основной агент модели, который на каждом шаге определяет цены, 
совершает закупки запасов для основного и оборотного капитала, производит блага, инвестирует, а также выставляет 
произведенные блага на рынок. В зависимости от того, к какой отрасли принадлежит фирма, у неё будут разные
вектора инвестиций, затрат и выпуска. 
"""),
                ui.h5("Производство"),
                ui.p("""
Фирмы производят по производственной технологии Леонтьева, по сути, преобразуя вектор одних благ
в вектор некоторых других благ:
$$ \\text{out}(\\mathbf{x}) = \\text{minimum}(
\min (\\mathbf{x} / \\mathbf{\\text{tech_vector}}) \cdot \\mathbf{\\text{out_vector}}, \\text{limit} )$$

причём максимальный выпуск фирмы на каждый данный момент ограничен "лимитами" фирмы, которые представляют
из себя некоторый аналог основного капитала. 
В данном интерфейсе
вектор выхода имеет только один ненулевой элемент из-за настроек, но сама архитектура модели даёт возможность 
сделать вектор выхода любым. 

На первом шаге фирма просто старается
максимизировать объём производства, а затем
объёмы производства фирма определяет исходя из задачи максимизации прибыли. В задаче максимизации ожидаемой прибыли, 
фирма оценивает выручку по назначенным ей ценам и по объёму производства продукции, и определяет объём благ, которые
надо купить, в рамках доли своего бюджета, который она тратит на производство (это гиперпараметр модели, по умолчанию
равный 0.8):

$$ \pi^e (V, P) = \\text{out}(\\sum_{i} V_{ij}) \cdot p_{\\text{firm}} - \\sum_{ij} V_{ij} \\cdot P_{ij} 
\\rightarrow \max_{V} \\\\ s.t. \\sum_{ij} V_{ij} \\cdot P_{ij} \le \\text{firms_budget} \cdot \\text{prod_proportion}$$

После покупки фирма преобразует их по своей производственной технологии, и увеличивает свои запасы. 
"""),
                ui.h5("Цены"),
ui.p("""Фирмы меняют цены в зависимости от изменения спроса. Они могут менять его либо линейно, либо
на заранее определённую величину. 
$$ 
p^{t} = 
\\begin{cases}
p_{t-1} \cdot (1 + \sigma(\\frac{D_{t-1} - D_{t-2}}{D_{t-2}}) - 0.5) & \\text{'Линейный' случай} \\\\
p + \delta \\frac{D_{t-1} - D_{t-2}}{D_{t-2}} & \\text{Иначе} 
\\end{cases}
$$
$$ \\text{где } \\sigma(x) \\text{- логистическая функция, которая была выбрана из-за своих свойств, которые автором сочлись 
подходящими для описания ценовой реакции фирм на изменение спроса.} $$

$$ \delta \\text{ - гиперпараметр, равный 0.25} $$
"""),
                ui.h5("Продажа товаров"),
                ui.p("""
Фирма продаёт определенную часть своих запасов после того, как осуществляет производство. 
По умолчанию она продаёт 100%, но эту долю можно изменить в настройках. 
"""),
                ui.h5("Инвестиции"),
                ui.p("""
Объём производства фирмы, как говорилось ранее, ограничен лимитами. 
Инвестиции фирма осуществляет после производства и продажи произведенных товаров. Она стремится максимизировать 
прибыль, которую она получит от своих инвестиций на протяжении
"deprecation_steps" (по умолчанию 10, это гиперпараметр, соответствующий времени 
жизни основного капитала), и минимизировать затраты на инвестиции, покупая необходимые блага 
на рынке. Формула для инвестиций следующая:
 $$ \\text{invest}(\\mathbf{x}) = \\lfloor \min (\\mathbf{x} / \\mathbf{\\text{invest_vector}}), \\text{max_invest} 
 ) \\rfloor $$
Объём инвестиций определяет, сколько единиц основного капитала, которые существуют на протяжении "deprecation_steps"
(если есть амортизация), получит фирма. 
В случае, если есть угроза того, что основной капитал совсем пропадёт, фирма осуществляет инвестиции вне зависимости
от их прибыльности. 
"""),
                ui.h5("Амортизация"),
                ui.p("""
                Если в модели включена амортизация, основной капитал фирмы существует на протяжении
                "deprecation_steps" периодов, и спустя это время пропадает. 
"""),
                ui.h3("Потребители"),
                ui.p("""
В этой модели потребители осуществляют непроизводственное потребление в зависимости от своей функции 
полезности, которая также является леонтьевской.          
$$ \\text{u}(\\mathbf{x}) = 
\min (\\mathbf{x} / \\mathbf{\\text{utility_vector}}) $$

Потребители тратят часть своего дохода на потребление, а оставшийся доход тратят на инвестиции в фирмы. Пропорции, 
по которым они распределяют свой доход, фиксированы (по умолчанию 0.7 на потребление, 0.3 на сбережения).
"""),
                ui.h5("Доход потребителей"),
                ui.p("""
Опционально, потребители могут получать базовый доход в виде стоимости определенного количества благ из их 
функции полезности. Также, если в модели есть собственность, то 
часть выручки фирм распределяется между потребителями. Доля, которую получит потребитель из "доходной" части
выручки фирмы, соответствует его суммарным инвестициям в эту фирму. 
"""),
                ui.h5("Инвестиции потребителей"),
                ui.p("""Часть выручки фирм распределяется между потребителями в зависимости от их суммарных инвестиций.
Каждый потребитель стремится инвестировать свои средства так, чтобы максимизировать доход от собственности, исходя
из последних прибылей фирм. 
решает следующую задачу:
$$ \\sum_{i=1}^n \\frac{S_{ki} + x_i}{x_i + \sum_{j} S_{ij}} \cdot \pi_i \\rightarrow \max_{\\mathbf{x}} \\\\
s.t. \\sum x_i = \\text{budget} $$

После этого он осуществляет инвестиции, и инвестированные им деньги распределяются между фирмами. Информация о суммарных
объёмах инвестиций обновляется. 
"""),
                ui.h3("Регулирование"),
                ui.p("""
В связи с тем, что в данной модели часто наблюдается тенденция к монополизации, которая 
приводит к тому, что все финансы скапливаются у одной отрасли, вследствие чего остальные отрасли не могут ничего
производить, и производство в экономике останавливается, предложены разные 
варианты регулирования, чтобы посмотреть, может ли оно помочь, среди них: фиксированные цены, прямые налоги на
фирмы и налоги на прибыль фирм. 
"""),
            )
        )
    )
)
