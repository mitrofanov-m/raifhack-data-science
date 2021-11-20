import pandas as pd
import re
from raiflib.utils import UNKNOWN_VALUE


def prepare_categorical(df: pd.DataFrame) -> None:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, выборка
    :return: None
    """
    fillna_cols = ['region','city','street','realty_type', 'floor']
    df[fillna_cols] = df[fillna_cols].fillna(UNKNOWN_VALUE)


def prepare_regions(df: pd.DataFrame) -> None:
    """
    Убирает из регионов города, в которых менее 3 объявлений
    :param df: dataframe, выборка
    :return: None
    """
    city_offer_counts = df[df.price_type == 0] \
        .groupby(['region', 'city'], as_index=False) \
        .agg({'id': 'count'})

    city_offer_indeces_series = df.reset_index()[df.price_type == 0] \
        .groupby(['region', 'city'])['index'].apply(list)
    city_offer_indeces = pd.DataFrame(city_offer_indeces_series.index.tolist(), columns=['region', 'city'])
    city_offer_indeces['indeces'] = city_offer_indeces_series.tolist()
    low_city_offers = pd.merge(city_offer_counts, city_offer_indeces, on=['region', 'city']).query('id < 3')
    bad_idx_list = [idx for city_indeces in low_city_offers.indeces.tolist() for idx in city_indeces]
    bad_df = df.index.isin(bad_idx_list)
    df = df[~bad_df]
    

def _get_float_floor(string):
    if re.search('(?<![0-9\-])1(?![0-9])', string):
        return 1
    elif re.search('(подвал|цоколь|-1)', string):
        return -1
    else:
        try:
            return re.search('(?<![0-9])-*\d+(?![0-9])', string).group()
        except:
                return 0


def _categorize_floor(floor):
    # заменить на код из ноутбука
    if isinstance(floor, str):
        floor = _get_float_floor(floor)

    try:
        if floor < 0:
            return 0
        elif floor == 1 or floor == 2:
            return floor
        elif floor in range(3,6):
            return 3
        elif floor in range(5,11):
            return 4
        elif floor in range(10,21):
            return 5
        elif floor in range(20,51):
            return 6
        elif floor > 50:
            return 7
        else:
            return 8
    except:
        return 8


def prepare_floor(floors: pd.Series):
    """ Предобработка этажей для оптимальной работы модели
    Categorical features must be encoded as non-negative integers (int) less than Int32.MaxValue (2147483647). 
    It is best to use a contiguous range of integers started from zero.
    """
    return floors.apply(_categorize_floor).astype('int32')


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полный цикл редобработки данных перед кодированием.
    :param df: dataframe, выборка
    :return: dataframe
    """
    new_df = df.copy()
    prepare_categorical(new_df)
    new_df['floor'] = prepare_floor(new_df['floor'])

    return new_df


def prepare_train_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полный цикл редобработки обучающих данных перед кодированием.
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    new_df = prepare_df(df)
    prepare_regions(new_df)

    return new_df
