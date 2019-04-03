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


def get_spec(cn: int) -> (int, envelope.FloorArea, envelope.Spec, cs.SystemSpec):

    spec = {
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

    region = spec[0]
    floor_area = envelope.FloorArea(a_mr=spec[1], a_or=spec[2], a_a=spec[3], r_env=spec[4])
    envelope_spec = envelope.Spec(insulation=spec[5], solar_gain=spec[6])
    cap_rtd_h, cap_rtd_c = appendix.get_rated_capacity(region, floor_area)
    system_spec = cs.SystemSpec(cap_rtd_h=cap_rtd_h, cap_rtd_c=cap_rtd_c,
                                supply_air_rtd_h=spec[7], supply_air_rtd_c=spec[8],
                                is_duct_insulated=spec[9], vav_system=spec[10])

    return region, floor_area, envelope_spec, system_spec


# region convert function


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


# region graph


def date_xs_range():

    start = datetime.strptime('2018-01-01', '%Y-%m-%d')

    return np.array([start + timedelta(n) for n in range(365)])


def draw_graph(y_title, ys, op: str ='ave'):

    plt.style.use('seaborn-whitegrid')

    xs = date_xs_range()

    fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(1, 1, 1)

    f = {
        'ave': get_average,
        'itg': get_integration,
        'a3': get_three_characteristics,
        'a5': get_five_characteristics,
    }[op]

    for y in ys:
        ysds = f(y[0], y[1])
        for ysd in ysds:
            ax.plot(xs, ysd[0], label=ysd[1])

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim(datetime.strptime('2018-01-01', '%Y-%m-%d'), datetime.strptime('2019-01-01', '%Y-%m-%d'))
    ax.set_ylabel(y_title)
    plt.legend()
    plt.show()


def draw_sum_bar_graph(x_title, ys):

    fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(1, 1, 1)

    values = [np.sum(y[0]) for y in ys]
    titles = [y[1] for y in ys]
    xs = np.arange(len(ys))

    ax.barh(xs+0.5, values)
    ax.set_yticks(xs + 0.5)
    ax.set_yticklabels(titles)
    plt.show()


# endregion


# region external conditions

def get_outdoor_temperature(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_conditions.read_temperature(region)


def get_outdoor_absolute_humidity(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_conditions.read_absolute_humidity(region)


def get_direct_solar(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_conditions.read_direct_solar(region)


def get_horizontal_sky_solar(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_conditions.read_horizontal_sky_solar(region)


def get_night_radiation(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_conditions.read_night_radiation(region)


def get_horizontal_solar(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_conditions.get_horizontal_solar(region)


def get_sat_temperature(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_conditions.get_sat_temperature(region)

# endregion


# region heat load


def get_heating_load(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    if region == 8:
        return np.full((12, 8760), 0.0)
    else:
        return read_load.get_heating_load(region, envelope_spec, floor_area)


def get_sensible_cooling_load(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_load.get_sensible_cooling_load(region, envelope_spec, floor_area)


def get_latent_cooling_load(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return read_load.get_latent_cooling_load(region, envelope_spec, floor_area)


# endregion


# region ducting

def get_attic_temperature_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_attic_temperature(region)[0]


def get_attic_temperature_for_cooling(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_attic_temperature(region)[1]


def get_duct_ambient_air_temperature(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_duct_ambient_air_temperature(floor_area.total, region, system_spec)


def get_total_duct_length(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_duct_length(floor_area.total)[2]

# endregion


# region supply air volume

def get_mechanical_ventilation(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_mechanical_ventilation(floor_area)


def get_heating_output_for_supply_air_estimation(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_heating_output_for_supply_air_estimation(region, floor_area, envelope_spec)


def get_cooling_output_for_supply_air_estimation(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_cooling_output_for_supply_air_estimation(region, floor_area, envelope_spec)


def get_heat_source_supply_air_volume_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_heat_source_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)


def get_heat_source_supply_air_volume_for_cooling(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_heat_source_supply_air_volume_for_cooling(region, floor_area, envelope_spec, system_spec)


def get_each_supply_air_volume_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_each_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)


def get_each_supply_air_volume_for_cooling(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_each_supply_air_volume_for_cooling(region, floor_area, envelope_spec, system_spec)

# endregion


# region non occupant room temperature

def get_non_occupant_room_temperature_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_non_occupant_room_temperature_for_heating(region, floor_area, envelope_spec, system_spec)


def get_non_occupant_room_temperature_for_cooling(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_non_occupant_room_temperature_for_cooling(region, floor_area, envelope_spec, system_spec)

# endregion


def get_maximum_output_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_maximum_output_for_heating(region, floor_area, envelope_spec, system_spec)


def get_maximum_output_for_cooling(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_maximum_output_for_cooling(region, floor_area, envelope_spec, system_spec)


def get_treated_untreated_heat_load_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_treated_untreated_heat_load_for_heating(region, floor_area, envelope_spec, system_spec)


def get_treated_untreated_heat_load_for_cooling(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_treated_untreated_heat_load_for_cooling(region, floor_area, envelope_spec, system_spec)


def get_heat_source_heating_output(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_heat_source_heating_output(region, floor_area, envelope_spec, system_spec)


def get_heat_source_sensible_cooling_output(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_heat_source_cooling_output(region, floor_area, envelope_spec, system_spec)[0]


def get_heat_source_latent_cooling_output(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_heat_source_cooling_output(region, floor_area, envelope_spec, system_spec)[1]


def get_decided_outlet_supply_air_temperature_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_decided_outlet_supply_air_temperature_for_heating(region, floor_area, envelope_spec, system_spec)


def get_air_conditioned_temperature_for_heating(cn: int) -> np.ndarray:
    return np.full(8760, cs.get_air_conditioned_temperature_for_heating())


def get_non_occupant_room_load(cn: int) -> np.ndarray:

    region, floor_area, envelope_spec, system_spec = get_spec(cn)

    temp_nor = cs.get_non_occupant_room_temperature_for_heating(region, floor_area, envelope_spec, system_spec)
    temp_or = np.full(8760, cs.get_air_conditioned_temperature_for_heating())

    # specific heat of air, J/kgK
    c = cs.get_specific_heat()

    # air density, kg/m3
    rho = cs.get_air_density()

    # supply air volume for heating, m3/h (5 rooms * 8760 times)
    v_supply_h_each = cs.get_each_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)
    # total supply air volume for heating, m3/h (8760 times)
    v_supply_h = np.sum(v_supply_h_each, axis=0)

    return (temp_or - temp_nor) * v_supply_h * c * rho * 10**(-6)


def get_heat_loss_through_partition_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_heat_loss_through_partition_for_heating(region, floor_area, envelope_spec, system_spec)


def get_duct_heat_loss_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_duct_heat_loss_for_heating(region, floor_area, envelope_spec, system_spec)


def get_treated_load_for_heating(cn: int) -> np.ndarray:
    region, floor_area, envelope_spec, system_spec = get_spec(cn)
    return cs.get_treated_load_for_heating(region, floor_area, envelope_spec, system_spec)




