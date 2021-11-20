<div align='center'><h1> ⚡ Raifhack DS 2021 ⚡ </h1></div>

RaifhackDS - онлайн-хакатон Райффайзенбанка в области Data Science. 

В данном репозитории представлено решение задачи хакатона в виде [jupyter ноутбука](./notebooks), в дальнейшем адаптированное под [пакет python](./raiflib) и demo-сервис по оценке коммерческой недвижимости.

<hr>
<h3 align="center"><a href='' target="_blank">Попробовать demo-сервис</a><h3>

## 🔎 Описание
Банки выдают кредиты, в том числе, под залог недвижимости.  Каждый объект залога оценивают: жилые с помощью ИТ (например, SRG group или «Мобильный оценщик»), а вот коммерческие — чаще вручную. Полноценного автоматизированного инструмента для оценки коммерческой недвижимости пока никто не анонсировал.

## 🎯 Задача
Разработать алгоритм оценки стоимости коммерческой недвижимости.

## 💡 Решение
За время проведения хакатона и период работы песочницы для дорешивания удалось реализовать следующие идеи и приемы, дополнившие базовые признаки, предоставленные организаторами:
- преобразование признаков смешанного типа (этажей) в категориальную переменную
- очистка от выбросов
- объединение коррелирующих признаков (`reform_mean_year_building_500`, `reform_mean_year_building_1000`) в один признак (`reform_mean_year_building`)
- добавление географических признаков, таких как расстояние до столицы (`distance_to_moscow`), расстояние до центра региона (`distance_to_region_center`)
- логарифмирование целевой переменной (`per_square_meter_price`) и площади помещения (`total_square`) для приведения распределений к виду, близкому к нормальному

В качестве модели был выбран `CatBoostRegressor`.
Валидация проводилась разбиением на 5 фолдов.

## 🏁 Результаты
Решение, представленное в проекте занимает в [таблице песочницы](https://apply.raifhack.ru/competition) **93 место** из 379.

<center>

| Место | Команда           |  Скор    | Итоговый скор|
|:-----: |:----------------: | -----:   |        -----:|
| 1     | Козуэй-Бей	    |  1.0325  | 0.9077       |
| 2     | Curious insight   |  1.1027  | 0.9324       |
| 3     | Subsoul           |  1.1271  | 0.9573       |
|       |    ...                                      |
|**93** | **kks_ml**        |**1.2313**| **1.1930**   |
|       |    ...                                      |
| 379   |    TeamBBZ        | 0        |  0           |

</center>

Считаю, что для первого соревнования результат хороший. Хочется так же поблагодарить организаторов соревнований за качественное мероприятие, высокую интерактивность и яркое after-party!


## 🖊 Контакты
По всем вопросам:
- @m1trm - telegram
