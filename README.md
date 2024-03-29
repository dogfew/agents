# Моделирование динамики экономики и неравенства в зависимости от производственных технологий

## Запуск

Надо склонировать репозиторий:
```
git clone https://github.com/dogfew/agents
```
Для запуска нужен Python, а также установленные библиотеи из `requirements.txt`: shiny, shinywidgets, plotly, numpy, scipy и т.д.
Установка зависимостей:

```
pip install -r requiremets.txt
```
Запуск:

```
shiny run --reload
```
Дальше надо настроить модель, нажать кнопку "Сконфигурировать" и сделать шаги. 

## Введение

В данной работе моделируется экономика, где производство задано с помощью матрицы Леонтьева, и есть фирмы, распределённые
по разным отраслям в этой матрице. Эта матрица, как и число фирм в каждой отрасли, задаётся при инициализации модели, и 
может быть любого размера, однако большее число фирм и отраслей приводит к большему времени вычислений. 
Среда, где фирмы взаимодействуют между собой, в данной модели называется рынком. 
Опционально в модели может фигурировать другой вид агентов: потребители, функция которых состоит в 
непроизводительном потреблении и инвестициях в фирмы. Инвестиции в фирмы осуществляются через рынок ценных бумаг. 

Чтобы модель заработала, фирмы изначально должны иметь определенное количество запасов, из которых осуществят производство.

## Рынок 

Рынок представляет из себя среду, где фирмы взаимодействуют между собой. Там хранятся произведённые каждой фирмой
товары в матрице
```math
V \in \mathbb{R}^{\text{n\_firms} \times \text{n\_commodities}}
```
а также назначенные каждой фирмой цены в матрице
```math
P \in \mathbb{R}^{\text{n\_firms} \times \text{n\_commodities}}
```

Таким образом, цены на рынке индивидуальны для каждой фирмы. Также на рынке сосредоточена выручка фирм: она увеличивается
в зависимости от стоимости проданных фирмой благ, и в начале каждого хода фирма получает эту выручку.


## Фирмы

Фирма - это основной агент модели, который на каждом шаге определяет цены,
совершает закупки запасов для основного и оборотного капитала, производит блага, инвестирует, а также выставляет
произведенные блага на рынок. В зависимости от того, к какой отрасли принадлежит фирма, у неё будут разные
вектора инвестиций, затрат и выпуска.


### Производство

Фирмы производят по производственной технологии Леонтьева, по сути, преобразуя вектор одних благ
в вектор некоторых других благ:
```math
\text{out}(\mathbf{x}) = \text{minimum}(
\min (\mathbf{x} / \mathbf{\text{tech\_vector}}) \cdot \mathbf{\text{out\_vector}}, \text{limit} )
```

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

```math
 \pi^e (V, P) = \text{out}(\sum_{i} V_{ij}) \cdot p_{\text{firm}} - \sum_{ij} V_{ij} \cdot P_{ij}
\rightarrow \max_{V}
```
```math
s.t. \sum_{ij} V_{ij} \cdot P_{ij} \le \text{firms\_budget} \cdot \text{prod\_proportion}
```

После покупки фирма преобразует их по своей производственной технологии, и увеличивает свои запасы.


### Цены

Фирмы меняют цены в зависимости от изменения спроса. Они могут менять его либо линейно, либо
на заранее определённую величину.
```math
p_{t} =
\begin{cases}
p_{t-1} \cdot (1 + \sigma(\frac{D_{t-1} - D_{t-2}}{D_{t-2}}) - 0.5) & \text{'Линейный' случай} \\
p_{t-1} + \delta \frac{D_{t-1} - D_{t-2}}{D_{t-2}} & \text{Иначе}
\end{cases}
```
```math
\text{где } \sigma(x) \text{- логистическая функция}
```
Логистическая функция была выбрана из-за своих свойств, которые автором сочлись
подходящими для описания ценовой реакции фирм на изменение спроса.
```math 
\delta - \text{- гиперпараметр, равный 0.25}
``` 


### Продажа товаров

Фирма продаёт определенную часть своих запасов после того, как осуществляет производство.
По умолчанию она продаёт 100%, но эту долю можно изменить в настройках.


### Инвестиции

Объём производства фирмы, как говорилось ранее, ограничен лимитами.
Инвестиции фирма осуществляет после производства и продажи произведенных товаров. Она стремится максимизировать
прибыль, которую она получит от своих инвестиций на протяжении
"deprecation_steps" (по умолчанию 10, это гиперпараметр, соответствующий времени
жизни основного капитала), и минимизировать затраты на инвестиции, покупая необходимые блага
на рынке. Формула для инвестиций следующая:
```math
\text{invest}(\mathbf{x}) = \lfloor \min (\mathbf{x} / \mathbf{\text{invest\_vector}}), \text{max\_invest}
 ) \rfloor
```
Объём инвестиций определяет, сколько единиц основного капитала, которые существуют на протяжении "deprecation_steps"
(если есть амортизация), получит фирма.
В случае, если есть угроза того, что основной капитал совсем пропадёт, фирма осуществляет инвестиции вне зависимости
от их прибыльности.

### Амортизация

Если в модели включена амортизация, основной капитал фирмы существует на протяжении "deprecation_steps" периодов, и спустя это время пропадает.

## Потребители

В этой модели потребители осуществляют непроизводственное потребление в зависимости от своей функции
полезности, которая также является леонтьевской.
```math
\text{u}(\mathbf{x}) =
\min (\mathbf{x} / \mathbf{\text{utility\_vector}})
```

Потребители тратят часть своего дохода на потребление, а оставшийся доход тратят на инвестиции в фирмы. Пропорции,
по которым они распределяют свой доход, фиксированы (по умолчанию 0.7 на потребление, 0.3 на сбережения).

### Доход потребителей

Опционально, потребители могут получать базовый доход в виде стоимости определенного количества благ из их
функции полезности. Также, если в модели есть собственность, то
часть выручки фирм распределяется между потребителями. Доля, которую получит потребитель из "доходной" части
выручки фирмы, соответствует его суммарным инвестициям в эту фирму.

### Инвестиции потребителей

Часть выручки фирм распределяется между потребителями в зависимости от их суммарных инвестиций.
Каждый потребитель стремится инвестировать свои средства так, чтобы максимизировать доход от собственности, исходя
из последних прибылей фирм.
решает следующую задачу:
```math
\sum_{i=1}^n \frac{S_{ki} + x_i}{x_i + \sum_{j} S_{ij}} \cdot \pi_i \rightarrow \max_{\mathbf{x}}
```
```math
s.t. \sum x_i = \text{budget}
```

После этого он осуществляет инвестиции, и инвестированные им деньги распределяются между фирмами. Информация о суммарных
объёмах инвестиций обновляется.


## Регулирование

В связи с тем, что в данной модели часто наблюдается тенденция к монополизации, которая
приводит к тому, что все финансы скапливаются у одной отрасли, вследствие чего остальные отрасли не могут ничего
производить, и производство в экономике останавливается, предложены разные
варианты регулирования, чтобы посмотреть, может ли оно помочь, среди них: фиксированные цены, прямые налоги на
фирмы и налоги на прибыль фирм.
