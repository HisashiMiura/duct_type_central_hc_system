import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta

import central_system as cs
import envelope
import read_conditions
import appendix
import read_load


def get_spec(cn: int):

    return {
        # 地域による違い
        1: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        2: [2, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        3: [3, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        4: [4, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        5: [5, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        6: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        7: [7, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        8: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        # 住戸規模による違い（基準：ケース1,6,8）
        9: [1, 15.0, 10.0, 60.06, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        10: [6, 15.0, 10.0, 60.06, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        11: [8, 15.0, 10.0, 60.06, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        # 躯体性能による違い（基準：ケース1,6,8）（重複あり）
        12: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'small', 1800.0, 1800.0, True, False],
        13: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'middle', 1800.0, 1800.0, True, False],
        14: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'large', 1800.0, 1800.0, True, False],
        15: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'small', 1800.0, 1800.0, True, False],
        16: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'middle', 1800.0, 1800.0, True, False],
        17: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'large', 1800.0, 1800.0, True, False],
        18: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'small', 1800.0, 1800.0, True, False],
        19: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        20: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'large', 1800.0, 1800.0, True, False],
        21: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'small', 1800.0, 1800.0, True, False],
        22: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'middle', 1800.0, 1800.0, True, False],
        23: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'large', 1800.0, 1800.0, True, False],
        24: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'small', 1800.0, 1800.0, True, False],
        25: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'middle', 1800.0, 1800.0, True, False],
        26: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'large', 1800.0, 1800.0, True, False],
        27: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'small', 1800.0, 1800.0, True, False],
        28: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'middle', 1800.0, 1800.0, True, False],
        29: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'large', 1800.0, 1800.0, True, False],
        30: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'small', 1800.0, 1800.0, True, False],
        31: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        32: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'large', 1800.0, 1800.0, True, False],
        33: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'small', 1800.0, 1800.0, True, False],
        34: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'middle', 1800.0, 1800.0, True, False],
        35: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'large', 1800.0, 1800.0, True, False],
        36: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'small', 1800.0, 1800.0, True, False],
        37: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'middle', 1800.0, 1800.0, True, False],
        38: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 's55', 'large', 1800.0, 1800.0, True, False],
        39: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'small', 1800.0, 1800.0, True, False],
        40: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'middle', 1800.0, 1800.0, True, False],
        41: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h4', 'large', 1800.0, 1800.0, True, False],
        42: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'small', 1800.0, 1800.0, True, False],
        43: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, False],
        44: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'large', 1800.0, 1800.0, True, False],
        45: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'small', 1800.0, 1800.0, True, False],
        46: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'middle', 1800.0, 1800.0, True, False],
        47: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11more', 'large', 1800.0, 1800.0, True, False],
        # 定格風量による違い（基準：ケース1,6,8)
        48: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 900.0, 900.0, True, False],
        49: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 900.0, 900.0, True, False],
        50: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 900.0, 900.0, True, False],
        # ダクト断熱による違い（基準：ケース1,6,8)
        51: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, False, False],
        52: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, False, False],
        53: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, False, False],
        # VAVシステムによる違い（基準：ケース1,6,8)
        54: [1, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, True],
        55: [6, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, True],
        56: [8, 29.81, 51.34, 120.08, 266.0 / 90.0, 'h11', 'middle', 1800.0, 1800.0, True, True],
    }[cn]


# region graph

# region convert function


def get_raw(v, name):
    return [(v, name)]


def get_integration(v, name):
    return [(np.sum(v.reshape(365, 24).T, axis=0), name)]


def get_average(v: np.ndarray, name):
    return [(np.mean(v.reshape(365, 24).T, axis=0), name)]


def get_three_characteristics(v, name):
    return [(np.min(v.reshape(365, 24).T, axis=0), name + '_min'),
            (np.mean(v.reshape(365, 24).T, axis=0), name + '_mean'),
            (np.max(v.reshape(365, 24).T, axis=0), name + '_max')]


def get_five_characteristics(v, name):
    return [
        (np.min(v.reshape(365, 24).T, axis=0), name + '_min'),
        (np.percentile(v.reshape(365, 24).T, 25, axis=0), name + '_25percentile'),
        (np.percentile(v.reshape(365, 24).T, 50, axis=0), name + '_mean'),
        (np.percentile(v.reshape(365, 24).T, 75, axis=0), name + '_75percentile'),
        (np.max(v.reshape(365, 24).T, axis=0), name + '_max')]


# endregion


def date_xs_range(op):

    start = datetime.strptime('2018-01-01', '%Y-%m-%d')

    if op == 'raw':
        return np.array([start + timedelta(hours=n) for n in range(8760)])
    else:
        return np.array([start + timedelta(n) for n in range(365)])


def draw_graph(y_title, ys, op: str ='ave', display_date: str = 'year'):

    plt.style.use('seaborn-whitegrid')

    xs = date_xs_range(op)

    fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(1, 1, 1)

    f = {
        'ave': get_average,
        'itg': get_integration,
        'a3': get_three_characteristics,
        'a5': get_five_characteristics,
        'raw': get_raw,
    }[op]

    for y in ys:
        ysds = f(np.array(y[0]), y[1])
        for ysd in ysds:
            ax.plot(xs, ysd[0], label=ysd[1])

    if display_date == 'year':
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
    if display_date == 'year':
        ax.set_xlim(datetime.strptime('2018-01-01', '%Y-%m-%d'), datetime.strptime('2019-01-01', '%Y-%m-%d'))
    else:
        start_date = datetime.strptime('2018/' + display_date, '%Y/%m/%d')
        end_date = start_date + timedelta(days=1)
        ax.set_xlim(start_date, end_date)
    ax.set_ylabel(y_title)
    plt.legend()
    plt.show()


def draw_sum_bar_graph(x_title, ys):

    fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(1, 1, 1)

    values = [np.sum(y[0]) for y in ys]
    titles = [y[1] for y in ys]
    xs = np.arange(len(ys))

    ax.barh(xs+0.5, values, tick_label=titles)
    ax.set_yticks(xs + 0.5)
    ax.set_xlabel(x_title)
    plt.show()


# endregion


def get_main_value(cn: int) -> dict:
    spec = get_spec(cn)
    system_spec = {
        'input': 'default',
        'is_duct_insulated': spec[9],
        'vav_system': spec[10]
    }
    return cs.get_main_value(region=spec[0],
                             a_mr=spec[1], a_or=spec[2], a_a=spec[3], r_env=spec[4],
                             insulation=spec[5], solar_gain=spec[6],
                             system_spec=system_spec)

