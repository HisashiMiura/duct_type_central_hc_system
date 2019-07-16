from typing import List, Tuple
import numpy as np
import pandas as pd

import read_conditions
import envelope
import read_load
import appendix
from appendix import SystemSpec


# region functions


# region system spec

def get_rated_capacity(region: int, a_a: float) -> (float, float):
    """
    calculate rated heating and cooling capacity of heat source
    Args:
        region: region, 1-8
        a_a: total floor area
    Returns:
        rated heating capacity, W
        rated cooling capacity, W
    """
    return appendix.get_rated_capacity(region, a_a)

# endregion


# region house spec

def get_non_occupant_room_floor_area(a_mr: float, a_or: float, a_a: float, r_env: float) -> float:
    """
    calculate the non occupant room floor area
    Args:
        a_mr: main occupant room floor area, m2
        a_or: other occupant room floor area, m2
        a_a: total floor area, m2
        r_env: the ratio of the total envelope area to the total floor area
    Returns:
        non occupant room floor area, m2
    """

    # make envelope.FloorArea class
    floor_area = envelope.FloorArea(a_mr, a_or, a_a, r_env)

    return floor_area.nor


def get_referenced_floor_area() -> np.ndarray:
    """
    get the referenced floor area of 12 rooms
    Returns:
        the referenced floor area, m2, (12 rooms)
    """

    return envelope.get_referenced_floor_area()


def get_floor_area(a_mr: float, a_or: float, a_a: float, r_env: float) -> np.ndarray:
    """
    calculate the floor area of the evaluated house
    Args:
        a_mr: main occupant floor area, m2
        a_or: other occupant floor area, m2
        a_a: total floor area, m2
        r_env: ratio of the envelope total area to the total floor area, -
    Returns:
        floor area of the evaluated house, m2, (12 rooms)
    """

    # make envelope.FloorArea class
    floor_area = envelope.FloorArea(a_mr, a_or, a_a, r_env)

    # floor area of the evaluated house, m2 (12 rooms)
    return envelope.get_hc_floor_areas(floor_area)


def get_partition_area(a_hcz: np.ndarray, a_mr, a_or, a_nr, r_env) -> np.ndarray:
    """
    calculate the areas of the partition
    Args:
        a_hcz: the partition area looking from each occupant rooms to the non occupant room, m2, (5 rooms)
        a_mr: main occupant room floor area, m2
        a_or: other occupant room floor area, m2
        a_nr: non occupant room floor area, m2
        r_env: ratio of the envelope total area to the total floor area, -
    Returns:
        the areas of the partitions, m2
    """

    # calculate the partition area between main occupant room and non occupant room, m2
    a_part_mr = a_hcz[0:1] * r_env * a_nr / (a_or + a_nr)

    # calculate the partition areas between 4 other occupant rooms and non occupant room, m2
    a_part_or = a_hcz[1:5] * r_env * a_nr / (a_mr + a_nr)

    # concatenate
    return np.concatenate((a_part_mr, a_part_or))


def get_heat_loss_coefficient_of_partition() -> float:
    """
    return the heat loss coefficient of the partition
    Returns:
        heat loss coefficient of the partition, W/m2K
    """
    return 1 / 0.46


def get_envelope_spec(region: int, insulation: str, solar_gain: str) -> (float, float, float):
    """
    get Q value, mu_h value, mu_c value
    Args:
        region: region, 1-8
        insulation: insulation level. specify the level as string following below:
            's55': Showa 55 era level
            'h4': Heisei 4 era level
            'h11': Heisei 11 era level
            'h11more': more than Heisei 11 era level
        solar_gain: solar gain level. specify the level as string following below.
            'small': small level
            'middle': middle level
            'large': large level
    Returns:
        Q value, W/m2K
        mu_h value, (W/m2)/(W/m2)
        mu_c value, (W/m2)/(W/m2)
    """

    # make envelope.Spec class
    envelope_spec = envelope.Spec(insulation, solar_gain)

    # Q value, W/m2K
    q = envelope_spec.get_q_value(region=region)

    # mu value, (W/m2)/(W/m2)
    mu_h = envelope_spec.get_mu_h_value(region=region)
    mu_c = envelope_spec.get_mu_c_value(region=region)

    return q, mu_h, mu_c


def get_mechanical_ventilation(a_hcz_r: np.ndarray, a_hcz: np.ndarray) -> np.ndarray:
    """
    calculate mechanical ventilation of each 5 rooms
    Args:
        a_hcz_r: the referenced heating and cooling zone floor area, m2, (12 rooms)
        a_hcz: the referenced heating and cooling zone floor area, m2, (12 rooms)
    Returns:
        supply air volume of mechanical ventilation, m3/h, (5 rooms)
    """

    # referenced mechanical ventilation volume, m3/h
    v_vent_r = np.array([60.0, 20.0, 40.0, 20.0, 20.0])

    # referenced floor area of the occupant room(sliced 0 to 5)
#    a_hcz_r = envelope.get_referenced_floor_area()[0:5]
    a_hcz_r = a_hcz_r[0:5]

    # floor area of the occupant room(sliced 0 to 5)
#    a_hcz = envelope.get_hc_floor_areas(zone_floor_area)[0:5]
    a_hcz = a_hcz[0:5]

    return v_vent_r * a_hcz / a_hcz_r

# endregion


# region general property

def get_air_density() -> float:
    """
    air density
    Returns:
        air density, kg/m3
    """
    return 1.2


def get_specific_heat() -> float:
    """
    specific heat of air
    Returns:
        specific heat of air, J/kgK
    """
    return 1006.0


def get_evaporation_latent_heat() -> float:
    """
    get latent heat of evaporation at 28 degree C
    because this value is used for the calculation of the latent cooling load
    Returns:
        latent heat of evaporation, kJ/kg
    """
    theta = 28.0
    return 2500.8 - 2.3668 * theta


def get_calender() -> np.ndarray:
    """
    get calender
    Returns:
        calender with 'weekday' and 'holiday' (8760 times)
    """

    df = pd.read_csv('schedule_data/schedule.csv')
    return np.repeat(df['暖冷房'].values, 24)

# endregion


# region occupant usage

def get_heating_and_cooling_schedule(region: int) -> (np.ndarray, np.ndarray):
    """
    get the heating and cooling schedule
    operation represents True as boolean type
    Args:
        region: region, 1-8
    Returns:
        (heating schedule, cooling schedule)
        heating schedule, operation day represents True, (8760 times)
        cooling schedule, operation day represents True, (8760 times)
    """

    heating_period = np.array([
                                  [True] * 157 + [False] * 110 + [True] * 98,
                                  [True] * 154 + [False] * 115 + [True] * 96,
                                  [True] * 150 + [False] * 123 + [True] * 92,
                                  [True] * 149 + [False] * 125 + [True] * 91,
                                  [True] * 134 + [False] * 149 + [True] * 82,
                                  [True] * 110 + [False] * 198 + [True] * 57,
                                  [True] * 85 + [False] * 245 + [True] * 35,
                                  [True] * 0 + [False] * 365 + [True] * 0,
                              ][region - 1])

    cooling_period = np.array([
                                  [False] * 190 + [True] * 53 + [False] * 122,
                                  [False] * 195 + [True] * 48 + [False] * 122,
                                  [False] * 190 + [True] * 53 + [False] * 122,
                                  [False] * 190 + [True] * 53 + [False] * 122,
                                  [False] * 186 + [True] * 57 + [False] * 122,
                                  [False] * 149 + [True] * 117 + [False] * 99,
                                  [False] * 134 + [True] * 152 + [False] * 79,
                                  [False] * 83 + [True] * 265 + [False] * 17,
                              ][region - 1])

    return np.repeat(heating_period, 24), np.repeat(cooling_period,24)


def get_n_p(a_mr: float, a_or: float, a_nr: float, calender: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    calculate number of peopele
    Args:
        a_mr: main occupant room floor area, m2
        a_or: other occupant room floor area, m2
        a_nr: non occupant room floor area, m2
        calender: calender with 'weekday' and 'holiday'
    Returns:
        (n_p, n_p_mr, n_p_or, n_p_nr)
        n_p: total number of people (8760 times)
        n_p_mr: number of people in main occupant room (8760 times)
        n_p_or: number of people in other occupant room (8760 times)
        n_p_nr: number of people in non occupant room (8760 times)
    """

    n_p_mr_wd = np.array([0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 1, 1, 0, 0, 1, 2, 2, 3, 3, 2, 1, 1]) * a_mr / 29.81
    n_p_or_wd = np.array([4, 4, 4, 4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 3]) * a_or / 51.34
    n_p_nr_wd = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * a_nr / 38.93
    n_p_mr_hd = np.array([0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 2, 1, 0, 0, 2, 3, 3, 4, 2, 2, 1, 0]) * a_mr / 29.81
    n_p_or_hd = np.array([4, 4, 4, 4, 4, 4, 4, 3, 1, 2, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 3]) * a_or / 51.34
    n_p_nr_hd = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * a_nr / 38.93

    n_p_mr = np.tile(n_p_mr_wd, 365) * (calender == '平日') + np.tile(n_p_mr_hd, 365) * (calender == '休日')
    n_p_or = np.tile(n_p_or_wd, 365) * (calender == '平日') + np.tile(n_p_or_hd, 365) * (calender == '休日')
    n_p_nr = np.tile(n_p_nr_wd, 365) * (calender == '平日') + np.tile(n_p_nr_hd, 365) * (calender == '休日')

    n_p = n_p_mr + n_p_or + n_p_nr

    return n_p, n_p_mr, n_p_or, n_p_nr


def get_q_gen(a_mr: float, a_or: float, a_nr: float, calender: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    calculate heat generation, W
    Args:
        a_mr: main occupant room floor area, m2
        a_or: other occupant room floor area, m2
        a_nr: non occupant room floor area, m2
        calender: calender with 'weekday' and 'holiday'
    Returns:
        (q_gen, q_gen_mr, q_gen_or, q_gen_nr)
        q_gen: total heat generation, W (8760 times)
        q_gen_mr: heat generation in main occupant room, W (8760 times)
        q_gen_or: heat generation in other occupant room, W (8760 times)
        q_gen_nr: heat generation in non occupant room, W (8760 times)
    """

    q_gen_mr_wd = np.array([
        66.9, 66.9, 66.9, 66.9, 66.9, 66.9, 123.9, 383.6, 323.2, 307.3, 134.8, 66.9,
        286.7, 271.2, 66.9, 66.9, 236.9, 288.6, 407.8, 383.1, 423.1, 339.1, 312.9, 278.0
    ]) * a_mr / 29.81

    q_gen_or_wd = np.array([
        18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 398.2, 18.0, 18.0,
        18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 53.0, 53.0, 115.5, 103.0, 258.3, 137.3
    ]) * a_or / 51.34

    q_gen_nr_wd = np.array([
        41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 126.1, 249.9, 158.3, 191.3, 117.5, 41.5,
        42.5, 89.0, 41.5, 41.5, 105.8, 105.8, 112.1, 118.5, 155.7, 416.1, 314.8, 174.9
    ]) * a_nr / 38.93

    q_gen_mr_hd = np.array([
        66.9, 66.9, 66.9, 66.9, 66.9, 66.9, 66.9, 66.9, 440.5, 443.3, 515.1, 488.9,
        422.9, 174.4, 66.9, 66.9, 237.8, 407.8, 383.1, 326.8, 339.1, 339.1, 312.9, 66.9
    ]) * a_mr / 29.81

    q_gen_or_hd = np.array([
        18, 18, 18, 18, 18, 18, 18, 18, 35.5, 654.3, 223, 223,
        53, 18, 18, 18, 93, 93, 55.5, 18, 270, 168.8, 270, 18
    ]) * a_or / 51.34

    q_gen_nr_hd = np.array([
        41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 281.3, 311, 269.5, 100.4, 106.7,
        98.5, 55.8, 41.5, 41.5, 158.4, 171.3, 82.7, 101.4, 99.5, 255.1, 232.1, 157.8
    ]) * a_nr / 38.93

    q_gen_mr = np.tile(q_gen_mr_wd, 365) * (calender == '平日') + np.tile(q_gen_mr_hd, 365) * (calender == '休日')
    q_gen_or = np.tile(q_gen_or_wd, 365) * (calender == '平日') + np.tile(q_gen_or_hd, 365) * (calender == '休日')
    q_gen_nr = np.tile(q_gen_nr_wd, 365) * (calender == '平日') + np.tile(q_gen_nr_hd, 365) * (calender == '休日')

    q_gen = q_gen_mr + q_gen_or + q_gen_nr

    return q_gen, q_gen_mr, q_gen_or, q_gen_nr


def get_w_gen(a_mr: float, a_or: float, a_nr: float, calender: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    calculate moisture generation, g/h
    Args:
        a_mr: main occupant room floor area, m2
        a_or: other occupant room floor area, m2
        a_nr: non occupant room floor area, m2
        calender: calender with 'weekday' and 'holiday'
    Returns:
        (w_gen, w_gen_mr, w_gen_or, w_gen_nr)
        w_gen: total moisture generation, g/h (8760 times)
        w_gen_mr: moisture generation in main occupant room, g/h (8760 times)
        w_gen_or: moisture generation in other occupant room, g/h (8760 times)
        w_gen_nr: moisture generation in non occupant room, g/h (8760 times)
    """

    w_gen_mr_wd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]) * a_mr / 29.81

    w_gen_or_wd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ]) * a_or / 51.34

    w_gen_nr_wd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ]) * a_nr / 38.93

    w_gen_mr_hd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]) * a_mr / 29.81

    w_gen_or_hd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ]) * a_or / 51.34

    w_gen_nr_hd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ]) * a_nr / 38.93

    w_gen_mr = np.tile(w_gen_mr_wd, 365) * (calender == '平日') + np.tile(w_gen_mr_hd, 365) * (calender == '休日')
    w_gen_or = np.tile(w_gen_or_wd, 365) * (calender == '平日') + np.tile(w_gen_or_hd, 365) * (calender == '休日')
    w_gen_nr = np.tile(w_gen_nr_wd, 365) * (calender == '平日') + np.tile(w_gen_nr_hd, 365) * (calender == '休日')

    w_gen = w_gen_mr + w_gen_or + w_gen_nr

    return w_gen, w_gen_mr, w_gen_or, w_gen_nr


def get_v_local(calender: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    calculate local ventilation amount, m3/h
    Args:
        calender: calender with 'weekday' and 'holiday'
    Returns:
        (v_local, v_local_mr, v_local_or, v_local_nr)
        v_local: total local ventilation amount, m3/h (8760 times)
        v_local_mr: local ventilation amount in main occupant room, m3/h (8760 times)
        v_local_or: local ventilation amount in other occupant room, m3/h (8760 times)
        v_local_nr: local ventilation amount in non occupant room, m3/h (8760 times)
    """

    v_local_mr_wd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 150.0, 150.0, 0.0,
        0.0, 0.0, 0.0])
    v_local_or_wd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0])
    v_local_nr_wd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 2.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 52.0,
        25.0, 102.8])
    v_local_mr_hd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 75.0, 0.0, 0.0, 0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 150.0, 150.0, 0.0, 0.0,
        0.0, 0.0, 0.0])
    v_local_or_hd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0])
    v_local_nr_hd = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 1.2, 1.2, 0.0, 0.0, 0.0, 0.0, 2.0, 75.8, 25.0, 2.0, 0.8, 25.0,
        27.0, 100.8])

    v_local_mr = np.tile(v_local_mr_wd, 365) * (calender == '平日') + np.tile(v_local_mr_hd, 365) * (calender == '休日')
    v_local_or = np.tile(v_local_or_wd, 365) * (calender == '平日') + np.tile(v_local_or_hd, 365) * (calender == '休日')
    v_local_nr = np.tile(v_local_nr_wd, 365) * (calender == '平日') + np.tile(v_local_nr_hd, 365) * (calender == '休日')

    v_local = v_local_mr + v_local_or + v_local_nr

    return v_local, v_local_mr, v_local_or, v_local_nr


def get_theta_set() -> (float, float):
    """
    get set temperature
    Returns:
        set temperature for heating, degree C
        set temperature for cooling, degree C
    """

    theta_set_h = 20.0
    theta_set_c = 27.0

    return theta_set_h, theta_set_c


def get_x_set() -> float:
    """
    get set absolute humidity
    Returns:
        set absolute humidity for cooling, kg/kgDA
    """

    x_set_c = 0.013425743  # 27℃ 60% の時の絶対湿度

    return x_set_c

# endregion


# region external conditions

def get_outdoor_temperature(region: int) -> np.ndarray:
    """
    get outdoor temperature
    Args:
        region: region, 1-8
    Returns:
        outdoor temperature, degree C, (8760 times)
    """

    return read_conditions.read_temperature(region)


def get_absolute_humidity(region: int) -> np.ndarray:
    """
    get outdoor absolute humidity
    Args:
        region: region, 1-8
    Returns:
        outdoor absolute humidity, kg/kgDA, (8760 times)
    """

    return read_conditions.read_absolute_humidity(region)


def get_relative_humidity(theta_ex: np.ndarray, x_ex: np.ndarray) -> np.ndarray:
    """
    get outdoor relative humidity
    Args:
        outdoor temperature, degree C, (8760 times)
        outdoor absolute humidity, kg/kgDA, (8760 times)
    Returns:
        outdoor relative humidity, %, (8760 times)
    """

    return read_conditions.get_relative_humidity(theta_ex, x_ex)


def get_horizontal_solar(region: int) -> np.ndarray:
    """
    get horizontal solar radiation
    Args:
        region: region, 1-8
    Returns:
        horizontal solar radiation, W/m2, (8760 times)
    """

    return read_conditions.get_horizontal_solar(region)


def get_sat_temperature(region: int) -> np.ndarray:
    """
    get SAT temperature
    Args:
        region: region, 1-8
    Returns:
        SAT temperature, degree C, (8760 times)
    """

    return read_conditions.get_sat_temperature(region)

# endregion


# region circulating air flow

def get_heating_output_for_supply_air_estimation(
        a_a: float, q: float, mu_h: float, v_vent: np.ndarray,
        theta_ex: np.ndarray, j: np.ndarray,
        heating_period: np.ndarray, n_p: np.ndarray, q_gen: np.ndarray, v_local: np.ndarray,
        theta_set_h: float):
    """
    get heating output for supply air estimation
    Args:
        a_a: total floor area, m2
        q: Q value, W/m2K
        mu_h: mu_h value, (W/m2)/(W/m2)
        v_vent: mechanical ventilation, m3/h, (5 rooms)
        theta_ex: outdoor temperature, degree C (8760 times)
        j: horizontal solar radiation, W/m2K (8760 times)
        heating_period: heating schedule (8760 times)
        n_p: number of people (8760 times)
        q_gen: heat generation, W (8760 times)
        v_local: local ventilation amount, m3/h (8760 times)
        theta_set_h: set temperature for heating, degree C
    Returns:
        heating output for supply air estimation, MJ/h (8760 times)
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    q_d_hs_h = np.maximum(
        (((q - 0.35 * 0.5 * 2.4) * a_a + c * rho * (v_local + sum(v_vent)) / 3600) * (theta_set_h - theta_ex)
         - mu_h * a_a * j - q_gen - n_p * 79.0) * 3600 * 10 ** (-6),
        0.0) * heating_period

    return q_d_hs_h


def get_cooling_output_for_supply_air_estimation(
        a_a: float, q: float, mu_c: float, v_vent: np.ndarray,
        theta_ex: np.ndarray, x_ex: np.ndarray, j: np.ndarray,
        cooling_period: np.ndarray, n_p: np.ndarray, q_gen: np.ndarray, w_gen: np.ndarray, v_local: np.ndarray,
        theta_set_c: float, x_set_c: float):
    """
    get heating output for supply air estimation
    Args:
        a_a: total floor area, m2
        q: Q value, W/m2K
        mu_c: mu_c value, (W/m2)/(W/m2)
        v_vent: mechanical ventilation, m3/h, (5 rooms)
        theta_ex: outdoor temperature, degree C (8760 times)
        x_ex: outdoor absolute humidity, kg/kg(DA) (8760 times)
        j: horizontal solar radiation, W/m2K (8760 times)
        cooling_period: cooling schedule (8760 times)
        n_p: number of people (8760 times)
        q_gen: heat generation, W (8760 times)
        w_gen: moisture generation, g/h (8760 times)
        v_local: local ventilation amount, m3/h (8760 times)
        theta_set_c: set temperature for cooling, degree C
        x_set_c: set absolute humidigy for cooling, kg/kgDA
    Returns:
        heating output for supply air estimation, MJ/h (8760 times)
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # latent heat of evaporation, kJ/kg
    l_wtr = get_evaporation_latent_heat()

    q_d_hs_cs = np.maximum(
        (((q - 0.35 * 0.5 * 2.4) * a_a + c * rho * (v_local + sum(v_vent)) / 3600) * (
                    theta_ex - theta_set_c)
         + mu_c * a_a * j + q_gen + n_p * 51.0) * 3600 * 10 ** (-6),
        0.0) * cooling_period

    q_d_hs_cl = np.maximum(
        (((v_local + sum(v_vent)) * rho * (x_ex - x_set_c) * 10 ** 3 + w_gen) * l_wtr
         + n_p * 40.0 * 3600) * 10 ** (-6), 0.0) * cooling_period

    return q_d_hs_cs + q_d_hs_cl


def get_heating_output_for_supply_air_estimation2(
        region: int, insulation: str, solar_gain: str, a_mr: float, a_or: float, a_a: float, r_env: float
) -> np.ndarray:
    """
    calculate heating output for supply air estimation
    Args:
        region: region
        insulation: insulation level
        solar_gain: solar gain level
        a_mr: main occupant room floor area, m2
        a_or: other occupant room floor area, m2
        a_a: total floor area, m2
        r_env: ratio of total envelope area to total floor area
    Returns:
        heating output for supply air estimation, MJ/h
    """

    # make envelope.FloorArea class
    floor_area = envelope.FloorArea(a_mr, a_or, a_a, r_env)

    # make envelope.Spec class
    envelope_spec = envelope.Spec(insulation, solar_gain)

    # heating load, MJ/h
    l_h = read_load.get_heating_load(region, envelope_spec, floor_area)

    q_dash_hs_h = np.sum(l_h, axis=0)

    # This operation is not described in the specification document
    # The supply air has lower limitation. This operation does not eventually effect the result.
    return np.vectorize(lambda x: x if x > 0.0 else 0.0)(q_dash_hs_h)


def get_cooling_output_for_supply_air_estimation2(
        region: int, insulation: str, solar_gain: str, a_mr: float, a_or: float, a_a: float, r_env: float
) -> np.ndarray:
    """
    calculate the cooling output for supply air estimation
    Args:
        region: region
        insulation: insulation level
        solar_gain: solar gain level
        a_mr: main occupant room floor area, m2
        a_or: other occupant room floor area, m2
        a_a: total floor area, m2
        r_env: ratio of total envelope area to total floor area
    Returns:
        sensible and latent cooling output for supply air estimation, MJ/h
    """

    # make envelope.FloorArea class
    floor_area = envelope.FloorArea(a_mr, a_or, a_a, r_env)

    # make envelope.Spec class
    envelope_spec = envelope.Spec(insulation, solar_gain)

    # sensible cooling load, MJ/h
    l_cs = read_load.get_sensible_cooling_load(region, envelope_spec, floor_area)

    # latent cooling load, MJ/h
    l_cl = read_load.get_latent_cooling_load(region, envelope_spec, floor_area)

    q_dash_hs_c = np.sum(l_cs, axis=0) + np.sum(l_cl, axis=0)

    # This operation is not described in the specification document
    # The supply air has lower limitation. This operation does not eventually effect the result.
    return np.vectorize(lambda x: x if x > 0.0 else 0.0)(q_dash_hs_c)


def get_minimum_air_volume(v_vent: np.ndarray) -> float:
    """
    calculate minimum air volume
    Args:
        v_vent: supply air volume of mechanical ventilation, m3/h, (5 rooms)
    Returns:
        minimum supply air volume of the system, m3/h, which is constant value through year.
    """

    return v_vent.sum()


def get_rated_output(cap_rtd_h: float, cap_rtd_c: float) -> (float, float):
    """
    calculate the rated heating and cooling output
    Args:
        cap_rtd_h: rated heating capacity, W
        cap_rtd_c: rated cooling capacity, W
    Returns:
        rated heating output, MJ/h
        rated cooling output, MJ/h
    """

    q_hs_rtd_h = cap_rtd_h * 3600 * 10 ** (-6)
    q_hs_rtd_c = cap_rtd_c * 3600 * 10 ** (-6)

    return q_hs_rtd_h, q_hs_rtd_c


def get_heat_source_supply_air_volume(
        heating_period: np.ndarray, cooling_period: np.ndarray,
        q_d_hs_h: np.ndarray, q_d_hs_c: np.ndarray, q_hs_rtd_h: float, q_hs_rtd_c: float,
        v_hs_min: float, v_hs_rtd_h: float, v_hs_rtd_c: float) -> np.ndarray:
    """
    calculate the supply air volume
    Args:
        heating_period: heating schedule (8760 times)
        cooling_period: cooling schedule (8760 times)
        q_d_hs_h: heating output of the system for estimation of the supply air volume, MJ/h
        q_d_hs_c: cooling output of the system for estimation of the supply air volume, MJ/h
        q_hs_rtd_h: rated heating output, MJ/h
        q_hs_rtd_c: rated cooling output, MJ/h
        v_hs_min: minimum supply air volume, m3/h
        v_hs_rtd_h: rated (maximum) supply air volume, m3/h
        v_hs_rtd_c: rated (maximum) supply air volume, m3/h
    Returns:
        supply air volume, m3/h (8760 times)
    """

    def get_v(q, q_hs_rtd, v_hs_rtd):
        if q < 0.0:
            return v_hs_min
        elif q < q_hs_rtd:
            return (v_hs_rtd - v_hs_min) / q_hs_rtd * q + v_hs_min
        else:
            return v_hs_rtd

    # supply air volume of heat source for heating and cooling, m3/h
    v_d_hs_supply_h = np.vectorize(get_v)(q_d_hs_h, q_hs_rtd_h, v_hs_rtd_h)
    v_d_hs_supply_c = np.vectorize(get_v)(q_d_hs_c, q_hs_rtd_c, v_hs_rtd_c)

    return v_d_hs_supply_h * heating_period + v_d_hs_supply_c * cooling_period


def get_supply_air_volume_valance(a_hcz: np.ndarray) -> np.ndarray:
    """
    calculate supply air volume valance
    Args:
        a_hcz: floor area of heating and cooling zones, m2, (12 rooms)
    Returns:
        the ratio of the supply air volume valance for each 5 rooms (0.0-1.0)
    """

    # slice the list. 1: main occupant room, 2-5: other occupant rooms
    occupant_rooms_floor_area = a_hcz[0:5]

    # calculate the ratio
    return occupant_rooms_floor_area / np.sum(occupant_rooms_floor_area)


def get_each_supply_air_volume_not_vav_adjust(
        r_supply_des: np.ndarray, v_hs_supply: np.ndarray, v_vent: np.ndarray) -> np.ndarray:
    """
    calculate each supply air volume without VAV adjust system
    Args:
        r_supply_des: supply air volume valance, (5 rooms)
        v_hs_supply: total supply air volume, m3/h
        v_vent: mechanical ventilation, m3/h (5 rooms)
    Returns:
        supply air volume, m3/h (5 rooms * 8760 times)
    """

    # supply air volume valance
    r_supply_des = r_supply_des.reshape(1, 5).T

    # mechanical ventilation, m3/h (5 rooms * 1 value)
    v_vent = v_vent.reshape(1, 5).T

    return np.maximum(v_hs_supply * r_supply_des, v_vent)

# endregion


# region load

def get_load(region: int, insulation: str, solar_gain: str, a_mr: float, a_or: float, a_a: float, r_env: float) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    get heating load, and sensible and latent cooling load
    Args:
        region: region
        insulation: insulation level
        solar_gain: solar gain level
        a_mr: main occupant room floor area, m2
        a_or: other occupant room floor area, m2
        a_a: total floor area, m2
        r_env: ratio of total envelope area to total floor area
    Returns:
        heating load, MJ/h (12 rooms * 8760 times)
        sensible cooling load, MJ/h (12 rooms * 8760 times)
        latent cooling load, MJ/h (12 rooms * 8760 times)
    """

    # make envelope.FloorArea class
    floor_area = envelope.FloorArea(a_mr, a_or, a_a, r_env)

    # make envelope.Spec class
    envelope_spec = envelope.Spec(insulation, solar_gain)

    # heating load, MJ/h
    l_h = read_load.get_heating_load(region, envelope_spec, floor_area)

    # sensible cooling load, MJ/h
    l_cs = read_load.get_sensible_cooling_load(region, envelope_spec, floor_area)

    # latent cooling load, MJ/h
    l_cl = read_load.get_latent_cooling_load(region, envelope_spec, floor_area)

    return l_h, l_cs, l_cl


def get_air_conditioned_room_temperature(
        heating_period: np.ndarray, cooling_period: np.ndarray,
        theta_ex: np.ndarray, theta_set_h: float, theta_set_c: float) -> np.ndarray:
    """
    calculate air conditioned room temperature
    Args:
        heating_period: heating schedule, operation day represents True, (8760 times)
        cooling_period: cooling schedule, operation day represents True, (8760 times)
        theta_ex: outdoor temperature, degree C, (8760 times)
        theta_set_h: set temperature for heating, degree C
        theta_set_c: set temperature for cooling, degree C
    Returns:
        air conditioned room temperature, degree C, (8760 times)
    """

    theta_ac_m = np.clip(theta_ex, theta_set_h, theta_set_c)

    middle_period = (heating_period == cooling_period)

    return theta_set_h * heating_period + theta_set_c * cooling_period + theta_ac_m * middle_period


def get_air_conditioned_room_absolute_humidity(
        cooling_period: np.ndarray, x_ex: np.ndarray, x_set_c: float) -> np.ndarray:
    """
    calculate air conditioned absolute humidity
    Args:
        cooling_period: cooling schedule, operation day represents True, (8760 times)
        x_ex: outdoor absolute humidity, kg/kgDA (8760 times)
        x_set_c: set absolute humidity for cooling, kg/kgDA (=27 degree C and 60%)

    Returns:
        air conditioned room absolute humidity, kg/kgDA (8760 times)
    """

    return x_set_c * cooling_period + x_ex * np.logical_not(cooling_period)


def get_non_occupant_room_temperature_balanced(
        heating_period: np.ndarray, cooling_period: np.ndarray,
        l_h: np.ndarray, l_cs: np.ndarray,
        q: float, a_nr: float, v_local_nr: np.ndarray,
        v_d_supply: np.ndarray, u_prt: float, a_prt: np.ndarray,
        theta_ac: np.ndarray) -> np.ndarray:
    """
    Args:
        heating_period: heating schedule, operation day represents True, (8760 times)
        cooling_period: cooling schedule, operation day represents True, (8760 times)
        l_h: heating load, MJ/h (12 rooms * 8760 times)
        l_cs: sensible cooling load, MJ/h (12 rooms * 8760 times)
        q: Q value, W/m2K
        a_nr: floor area of non occupant room, m2
        v_local_nr: local ventilation amount in non occupant room, m3/h (8760 times)
        v_d_supply: supply air volume, m3/h (5 rooms * 8760 times)
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2 (5 rooms)
        theta_ac: air conditioned temperature, degree C (8760 times)
    Returns:
        non occupant room temperature, degree C (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    cf = (q - 0.35 * 0.5 * 2.4) * a_nr + c * rho * v_local_nr / 3600 \
        + np.sum(c * rho * v_d_supply / 3600, axis=0) + np.sum(u_prt * a_prt)

    theta_nac_h = theta_ac - np.sum(l_h[5:12], axis=0) / cf * 10 ** 6 / 3600

    theta_nac_c = theta_ac + np.sum(l_cs[5:12], axis=0) / cf * 10 ** 6 / 3600

    middle_period = (heating_period == cooling_period)

    return theta_nac_h * heating_period + theta_nac_c * cooling_period + theta_ac * middle_period


def get_non_occupant_room_absolute_humidity_balanced(
        cooling_period: np.ndarray, l_cl: np.ndarray, v_local_nr: np.ndarray, v_d_supply: np.ndarray,
        x_ac: np.ndarray) -> np.ndarray:
    """
        calculate non occupant room absolute humidity
    Args:
        cooling_period: cooling schedule, operation day represents True, (8760 times)
        l_cl: latent cooling load, MJ/h (12 rooms * 8760 times)
        v_local_nr: local ventilation amount in non occupant room, m3/h (8760 times)
        v_d_supply: supply air volume, m3/h (5 rooms * 8760 times)
        x_ac: air conditioned absolute humidity, kg/kg(DA) (8760 times)
    Returns:
        non occupant room absolute humidity, kg/kgDA (8760 times)
    """

    # air density, kg/m3
    rho = get_air_density()

    # latent heat of evaporation, kJ/kg
    l_wtr = get_evaporation_latent_heat()

    x_d_nac_c = x_ac + np.sum(l_cl[5:12], axis=0) / (l_wtr * rho * (v_local_nr + np.sum(v_d_supply, axis=0)))

    return x_d_nac_c * cooling_period + x_ac * np.logical_not(cooling_period)


def get_heat_transfer_through_partition_balanced(
        u_prt: float, a_prt: np.ndarray, theta_ac: np.ndarray, theta_d_nac: np.ndarray) -> np.ndarray:
    """
    calculate heat transfer through the partition from occupant room into non occupant room
    Args:
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2, (5 rooms)
        theta_ac: air conditioned temperature, degree C (8760 times)
        theta_d_nac: non occupant room temperature, degree C (8760 times)
    Returns:
        heat transfer through the partition, MJ/h (5 rooms * 8760 times)
    """

    # area of the partition, m2
    a_prt = a_prt.reshape(1, 5).T

    return u_prt * a_prt * (theta_ac - theta_d_nac) * 3600 * 10 ** (-6)


def get_occupant_room_load_for_heating_balanced(l_h: np.ndarray, q_d_trs_prt: np.ndarray) -> np.ndarray:
    """
    calculate the heating load of the occupant room
    Args:
        l_h: heating load, MJ/h, (12 rooms * 8760 times)
        q_d_trs_prt: heat transfer from occupant room to non occupant room through partition, (5 rooms * 8760 times)
    Returns:
        heating load of occupant room, MJ/h, (5 rooms * 8760 times)
    """

    l_d_h = np.where(l_h[0:5] > 0.0, l_h[0:5] + q_d_trs_prt, 0.0)

    return np.clip(l_d_h, 0.0, None)


def get_occupant_room_load_for_cooling_balanced(
        l_cs: np.ndarray, l_cl: np.ndarray, q_d_trs_prt: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    calculate the cooling load of the occupant room
    Args:
        l_cs: sensible cooling load, MJ/h, (12 rooms * 8760 times)
        l_cl: latent cooling load, MJ/h, (12 rooms * 8760 times)
        q_d_trs_prt: heat transfer from occupant room to non occupant room through partition, (5 rooms * 8760 times)
    Returns:
        sensible and latent cooling load of occupant room, MJ/h, ((5 rooms *  8760 times), (5 rooms *  8760 times))
    """

    l_d_cs = np.where(l_cs[0:5] > 0.0, l_cs[0:5] - q_d_trs_prt, 0.0)
    l_d_cl = l_cl[0:5]

    return np.clip(l_d_cs, 0.0, None), np.clip(l_d_cl, 0.0, None)

# endregion


# region treated and untreated load

def get_duct_linear_heat_loss_coefficient() -> float:
    """
    get the liner heat loss coefficient (W/mK) of the duct
    Returns:
          liner heat loss coefficient, W/mK
    """
    return 0.49


def get_standard_house_duct_length() -> (np.ndarray, np.ndarray, np.ndarray):
    """
    get duct length of standard house
    Returns:
        duct length of standard house through the inside space of the insulated space, m (5 rooms)
        duct length of standard house through the outside space of the insulated space, m (5 rooms)
        total duct length of standard house, m (5 rooms)
    """

    # internal duct length of the insulated boundary
    internal = np.array([25.6, 8.6, 0.0, 0.0, 0.0])

    # external duct length of the insulated boundary
    external = np.array([0.0, 0.0, 10.2, 11.8, 8.1])

    # total duc length
    total = internal + external

    return internal, external, total


def get_duct_length(l_duct_r: np.ndarray, a_a: float) -> np.ndarray:
    """
    calculate duct length for each room in the estimated house
    Args:
        l_duct_r: duct length for each room in the standard house, m, (5 rooms)
        a_a: total floor area of the estimated house, m2
    Returns:
        duct length for each room in estimated house, m, (5 rooms)
    """

    a_a_r = 120.08

    return l_duct_r * np.sqrt(a_a / a_a_r)


def get_attic_temperature(theta_sat: np.ndarray, theta_ac: np.ndarray) -> np.ndarray:
    """
    calculate attic temperature for heating
    Args:
        theta_sat: SAT temperature, degree C, (8760 times)
        theta_ac: air conditioned temperature, degree C, (8760 times)
    Returns:
        attic temperature for heating, degree C, (8760 times)
    """

    # temperature difference coefficient
    h = 1.0

    return theta_sat * h + theta_ac * (1 - h)


def get_duct_ambient_air_temperature(
        is_duct_insulated: bool, l_duct_in_r: np.ndarray, l_duct_ex_r: np.ndarray,
        theta_ac: np.ndarray, theta_attic: np.ndarray) -> np.ndarray:
    """
    calculate duct ambient air temperature for heating
    Args:
        is_duct_insulated: is the duct insulated ?
        l_duct_in_r: duct length inside the insulated area in the standard house, m, (5 rooms)
        l_duct_ex_r: duct length outside the insulated area in the standard house, m, (5 rooms)
        theta_ac: air conditioned room temperature, degree C, (8760 times)
        theta_attic: attic temperature, degree C, (8760 times)
    Returns:
        duct ambient temperature, degree C, (5 rooms * 8760 times)
    """

    if is_duct_insulated:
        # If the duct insulated, the duct ambient temperatures are equals to the air conditioned temperatures.
#        return np.full((5, 8760), theta_ac)
        return np.tile(theta_ac, (5, 1))
    else:
        # If the duct NOT insulated, the duct ambient temperatures are
        # between the attic temperatures and the air conditioned temperatures.
        l_in = l_duct_in_r.reshape(1, 5).T
        l_ex = l_duct_ex_r.reshape(1, 5).T
        return (l_in * theta_ac + l_ex * theta_attic) / (l_in + l_ex)


def get_heat_source_inlet_air_balanced(theta_d_nac: np.ndarray, x_d_nac: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    calculate the inlet air temperature of heat source
    Args:
        theta_d_nac: non occupant room temperature when balanced, degree C (8760 times)
        x_d_nac: non occupant room absolute humidity when balanced, kg/kgDA (8760 times)
    Returns:
        the inlet air temperature of heat source when balanced, degree C (8760 times)
        the inlet air absolute humidity of heat source when balanced, kg/kgDA (8760 times)
    """

    return theta_d_nac, x_d_nac


def get_heat_source_maximum_heating_output(region: int, q_rtd_h: float) -> np.ndarray:
    """
    calculate maximum heating output
    Args:
        region: region, 1-8
        q_rtd_h: rated heating capacity, W
    Returns:
        maximum heating output, MJ/h
    """

    return appendix.get_maximum_heating_output(region, q_rtd_h)


def get_heat_source_maximum_cooling_output(
        q_rtd_c: float, l_d_cs: np.ndarray, l_d_cl: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    calculate the corrected_latent_cooling_load
    Args:
        q_rtd_c: rated cooling capacity, W
        l_d_cs: sensible cooling load in the occupant rooms, MJ/h, (5 rooms * 8760 times)
        l_d_cl: latent cooling load in the occupant rooms, MJ/h, (5 rooms * 8760 times)
    Returns:
        (a,b)
            a: sensible maximum cooling output, MJ/h, (8760 times)
            b: latent maximum cooling output, MJ/h, (8760 times)
    """

    return appendix.get_maximum_cooling_output(q_rtd_c, l_d_cs, l_d_cl)


def get_theta_hs_out_max_h(
        theta_d_hs_in: np.ndarray, q_hs_max_h: np.ndarray, v_d_supply: np.ndarray) -> np.ndarray:
    """
    calculate maximum temperature of output air of heat source when maximum output of heating
    Args:
        theta_d_hs_in: inlet air temperature of the heat source, degree C (8760 times)
        q_hs_max_h: maximum heating output, MJ/h (8760 times)
        v_d_supply: supply air volume for heating, m3/h (5 rooms * 8760 times)
    Returns:
        maximum temperature of output air of heat source when maximum output of heating, degree C, (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    return np.minimum(theta_d_hs_in + q_hs_max_h / (c * rho * np.sum(v_d_supply, axis=0)) * 10 ** 6, 45.0)


def get_theta_hs_out_min_c(
        theta_d_hs_in: np.ndarray, q_hs_max_cs: np.ndarray, v_d_supply: np.ndarray) -> np.ndarray:
    """
    calculate minimum temperature of output air of heat source when maximum output of cooling
    Args:
        theta_d_hs_in: inlet air temperature of the heat source, degree C (8760 times)
        q_hs_max_cs: maximum sensible cooling output, MJ/h (8760 times)
        v_d_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
    Returns:
        minimum temperature of output air of heat source when maximum output of cooling, degree C (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    return np.maximum(theta_d_hs_in - q_hs_max_cs / (c * rho * np.sum(v_d_supply, axis=0)) * 10 ** 6, 15.0)


def get_x_hs_out_min_c(
        x_d_hs_in: np.ndarray, q_hs_max_cl: np.ndarray, v_d_supply: np.ndarray) -> np.ndarray:
    """
    calculate minimum absolute humidity of output air of heat source when maximum output of cooling
    Args:
        x_d_hs_in: inlet air absolute humidity of the heat source, kg/kgDA (8760 times)
        q_hs_max_cl: maximum latent cooling output, MJ/h (8760 times)
        v_d_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
    Returns:
        minimum absolute humidity of output air of heat source when maximum output of cooling, kg/kgDA (8760 times)
    """

    # air density, kg/m3
    rho = get_air_density()

    # latent heat of evaporation, kJ/kg
    l_wtr = get_evaporation_latent_heat()

    return x_d_hs_in - q_hs_max_cl / (rho * l_wtr * np.sum(v_d_supply, axis=0)) * 10 ** 3


def get_requested_supply_air_temperature_for_heating(
        theta_sur_h: np.ndarray, theta_ac: np.ndarray, l_d_h: np.ndarray, v_d_supply: np.ndarray,
        psi: float, l_duct: np.ndarray) -> np.ndarray:
    """
    calculate the requested supply air temperature for heating
    Args:
        theta_sur_h: ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
        theta_ac: air conditioned room temperature, degree C, (8760 times)
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
        v_d_supply: supply air volume for heating, m3/h (5 rooms * 8760 times)
        psi: linear heat loss coefficient of the duct, W/mK
        l_duct: duct length, m, (5 rooms)
    Returns:
        requested temperature, degree C, (5 rooms * 8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_duct = np.array(l_duct).reshape(1, 5).T

    theta_req_h = theta_sur_h + (theta_ac + l_d_h * 10 ** 6 / (v_d_supply * c * rho) - theta_sur_h) \
        * np.exp(psi * l_duct * 3600 / (v_d_supply * c * rho))

    return np.maximum(theta_req_h, theta_ac)


def get_requested_supply_air_temperature_for_cooling(
        theta_sur_c: np.ndarray, theta_ac: np.ndarray, l_d_cs: np.ndarray, v_d_supply: np.ndarray,
        psi: float, l_duct: np.ndarray) -> np.ndarray:
    """
    calculate the requested supply air temperature for heating
    Args:
        theta_sur_c: ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
        theta_ac: air conditioned room temperature, degree C, (8760 times)
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms *  8760 times)
        v_d_supply: supply air volume for heating, m3/h (5 rooms * 8760 times)
        psi: linear heat loss coefficient of the duct, W/mK
        l_duct: duct length, m
    Returns:
        requested temperature, degree C, (5 rooms * 8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_duct = np.array(l_duct).reshape(1,5).T

    theta_req_c = theta_sur_c - (theta_sur_c - theta_ac + l_d_cs * 10 ** 6 / (v_d_supply * c * rho)) \
        * np.exp(psi * l_duct * 3600 / (v_d_supply * c * rho))

    return np.minimum(theta_req_c, theta_ac)


def get_requested_supply_air_absolute_humidity_for_cooling(
        x_ac: np.ndarray, l_d_cl: np.ndarray, v_d_supply: np.ndarray) -> np.ndarray:
    """
    calculate the requested supply air absolute humidity for cooling
    Args:
        x_ac: air conditioned absolute humidity, kg/kg(DA) (8760 times)
        l_d_cl: latent cooling load in the occupant rooms, MJ/h, (5 rooms * 8760 times)
        v_d_supply: supply air volume for heating, m3/h (5 rooms * 8760 times)
    Returns:
        requested absolute humidity, kg/kgDA (5 rooms * 8760 times)
    """

    # air density, kg/m3
    rho = get_air_density()

    # latent heat of evaporation, kJ/kg
    l_wtr = get_evaporation_latent_heat()

    return x_ac - l_d_cl * 10 ** 3 / (v_d_supply * rho * l_wtr)


def get_decided_outlet_supply_air_temperature_for_heating(
        vav_system: bool, theta_req_h: np.ndarray, v_d_supply: np.ndarray,
        theta_hs_out_max_h: np.ndarray) -> np.ndarray:
    """
    decide the outlet supply air temperature for heating
    Args:
        vav_system: is vav system equipped or not
        theta_req_h: requested supply air temperature of heat source, degree C, (5 rooms * 8760 times)
        v_d_supply: supply air volume without vav adjustment, m3/h (5 rooms * 8760 times)
        theta_hs_out_max_h:
            maximum temperature of output air of heat source when maximum output of heating, degree C, (8760 times)
    Returns:
        decided outlet supply air temperature, degree C, (8760 times)
    """

    if vav_system:
        return np.minimum(np.max(theta_req_h, axis=0), theta_hs_out_max_h)
    else:
        return np.minimum(np.sum(theta_req_h * v_d_supply / v_d_supply.sum(axis=0), axis=0), theta_hs_out_max_h)


def get_decided_outlet_supply_air_temperature_for_cooling(
        vav_system: bool, theta_req_c: np.ndarray, v_d_supply: np.ndarray,
        theta_hs_out_min_c: np.ndarray) -> np.ndarray:
    """
    decide the outlet supply air temperature for cooling
    Args:
        vav_system: is vav system equipped or not
        theta_req_c: requested supply air temperature of heat source, degree C, (5 rooms * 8760 times)
        v_d_supply: supply air volume without vav adjustment, m3/h (5 rooms * 8760 times)
        theta_hs_out_min_c:
                minimum temperature of output air of heat source when maximum output of cooling, degree C (8760 times)
    Returns:
        decided outlet supply air temperature, degree C, (8760 times)
    """

    if vav_system:
        return np.maximum(np.min(theta_req_c, axis=0), theta_hs_out_min_c)
    else:
        return np.maximum(np.sum(theta_req_c * v_d_supply / v_d_supply.sum(axis=0), axis=0), theta_hs_out_min_c)


def get_each_supply_air_volume(
        heating_period: np.ndarray, cooling_period: np.ndarray,
        vav_system: bool, l_d_h: np.ndarray, l_d_cs: np.ndarray,
        theta_hs_out_h: np.ndarray, theta_hs_out_c: np.ndarray, theta_sur: np.ndarray,
        psi: float, l_duct: np.ndarray, theta_ac: np.ndarray,
        v_vent: np.ndarray, v_d_supply: np.ndarray) -> np.ndarray:
    """
    calculate each supply air volume
    Args:
        heating_period: heating schedule, operation day represents True, (8760 times)
        cooling_period: cooling schedule, operation day represents True, (8760 times)
        vav_system: is vav system equipped or not
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms *  8760 times)
        theta_hs_out_h: supply air temperature of heat source, degree C, (5 rooms * 8760 times)
        theta_hs_out_c: supply air temperature of heat source, degree C, (5 rooms * 8760 times)
        theta_sur: ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
        theta_ac: air conditioned temperature, degree C (8760 times)
        v_vent: supply air volume of mechanical ventilation, m3/h, (5 rooms)
        v_d_supply: supply air volume without vav adjustment, m3/h (5 rooms * 8760 times)
    Returns:
        each supply air volume adjusted, m3/h, (5 rooms * 8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_duct = np.array(l_duct).reshape(1, 5).T

    v_vent = v_vent.reshape(1, 5).T

    if vav_system:

        # np.where の条件式はどちらも評価するためゼロ割の警告が発生する。
        # それを避けるため、ゼロ割が発生する場合はゼロ割が発生しないようにダミーの値を設定しておく。
        theta_hs_out_h = np.where(theta_hs_out_h > theta_ac, theta_hs_out_h, theta_hs_out_h + 1)
        theta_hs_out_c = np.where(theta_ac > theta_hs_out_c, theta_hs_out_c, theta_hs_out_c - 1)

        v_h = np.where(theta_hs_out_h > theta_ac,
                       np.clip(
                           (l_d_h * 10 ** 6 + (theta_hs_out_h - theta_sur) * psi * l_duct * 3600)
                           / (c * rho * (theta_hs_out_h - theta_ac)), v_vent, v_d_supply),
                       v_vent)

        v_c = np.where(theta_ac > theta_hs_out_c,
                       np.clip(
                           (l_d_cs * 10 ** 6 + (theta_sur - theta_hs_out_c) * psi * l_duct * 3600)
                           / (c * rho * (theta_ac - theta_hs_out_c)), v_vent, v_d_supply),
                       v_vent)

    else:

        v_h = v_d_supply
        v_c = v_d_supply

    v_supply_h = np.where(np.sum(l_d_h, axis=0) > 0.0, v_h, v_vent)
    v_supply_c = np.where(np.sum(l_d_cs, axis=0) > 0.0, v_c, v_vent)

    middle_period = (heating_period == cooling_period)

    return v_supply_h * heating_period + v_supply_c * cooling_period + v_vent * middle_period


def get_decided_outlet_supply_air_absolute_humidity_for_cooling(
        x_req_c: np.ndarray, v_supply: np.ndarray, x_hs_out_min_c: np.ndarray) -> np.ndarray:
    """
    decide the outlet supply air absolute humidity for cooling
    Args:
        x_req_c: requested supply air absolute humidity of heat source, kg/kgDA (5 rooms * 8760 times)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        x_hs_out_min_c:
            minimum absolute humidity of output air of heat source when maximum output of cooling, kg/kgDA (8760 times)
    Returns:
        decided outlet supply air absolute humidity, kg/kgDA (8760 times)
    """

    return np.maximum(np.sum(x_req_c * v_supply / v_supply.sum(axis=0), axis=0), x_hs_out_min_c)


def get_duct_heat_loss_for_heating(
        theta_sur_h: np.ndarray, theta_hs_out_h: np.ndarray, v_supply: np.ndarray,
        psi: float, l_duct: np.ndarray, l_d_h: np.ndarray) -> np.ndarray:
    """
    calculate the heat loss from the ducts for heating
    Args:
        theta_sur_h: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_h: outlet temperature of heat source, degree C, (8760 times)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
    """

    l_duct = np.array(l_duct).reshape(1, 5).T

    q_duct_h = get_duct_heat_loss_from_upside_temperature(
        theta_sur_h, theta_hs_out_h, v_supply, psi, l_duct)

    return np.where(np.sum(l_d_h, axis=0) > 0.0, q_duct_h, 0.0)


def get_duct_heat_gain_for_cooling(
        theta_sur_c: np.ndarray, theta_hs_out_c: np.ndarray, v_supply: np.ndarray,
        psi: float, l_duct: np.ndarray, l_d_cs: np.ndarray) -> np.ndarray:
    """
    calculate the heat gain to the ducts for cooling
    Args:
        theta_sur_c: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_c: outlet temperature of heat source, degree C, (8760 times)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms *  8760 times)
    """

    l_duct = np.array(l_duct).reshape(1, 5).T

    q_duct_c = - get_duct_heat_loss_from_upside_temperature(
        theta_sur_c, theta_hs_out_c, v_supply, psi, l_duct)

    return np.where(np.sum(l_d_cs, axis=0) > 0.0, q_duct_c, 0.0)


def get_supply_air_temperature_for_heating(
        theta_sur: np.ndarray, theta_hs_out_h: np.ndarray, psi: float, l_duct: np.ndarray,
        v_supply: np.ndarray, theta_ac: np.ndarray, l_d_h: np.ndarray) -> np.ndarray:
    """
    calculate supply air temperatures for heating
    Args:
        theta_sur: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_h: outlet temperature of heat source, degree C, (8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        theta_ac: air conditioned temperature, degree C (8760 times)
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
    Returns:
        supply air temperatures, degree C, (5 rooms * 8760 times)
    """

    l_duct = np.array(l_duct).reshape(1, 5).T

    theta_supply_h = get_downside_temperature_from_upside_temperature(theta_sur, theta_hs_out_h, v_supply, psi, l_duct)

    return np.where(np.sum(l_d_h, axis=0) > 0.0, theta_supply_h, theta_ac)


def get_supply_air_temperature_for_cooling(
        theta_sur: np.ndarray, theta_hs_out_c: np.ndarray, psi: float, l_duct: np.ndarray,
        v_supply: np.ndarray, theta_ac: np.ndarray, l_d_cs: np.ndarray) -> np.ndarray:
    """
    calculate supply air temperatures for cooling
    Args:
        theta_sur: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_c: outlet temperature of heat source, degree C, (8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        theta_ac: air conditioned temperature, degree C (8760 times)
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms * 8760 times)
    Returns:
        supply air temperatures, degree C, (5 rooms * 8760 times)
    """

    l_duct = np.array(l_duct).reshape(1, 5).T

    theta_supply_c = get_downside_temperature_from_upside_temperature(theta_sur, theta_hs_out_c, v_supply, psi, l_duct)

    return np.where(np.sum(l_d_cs, axis=0) > 0.0, theta_supply_c, theta_ac)


def get_supply_air_absolute_humidity_for_cooling(
        x_hs_out_c: np.ndarray, x_ac: np.ndarray, l_d_cl: np.ndarray) -> np.ndarray:
    """
    calculate supply air absolute humidity for cooling
    Args:
        x_hs_out_c: decided outlet supply air absolute humidity, kg/kgDA (8760 times)
        x_ac: air conditioned absolute humidity, kg/kgDA (8760 times)
        l_d_cl: latent cooling load in the occupant rooms, MJ/h, (5 rooms * 8760 times)
    Returns:
        supply air absolute humidity, kg/kgDA (5 rooms * 8760 times)
    """

    return np.where(np.sum(l_d_cl, axis=0) > 0.0, x_hs_out_c, x_ac)


def get_actual_air_conditioned_temperature(
        heating_period: np.ndarray, cooling_period: np.ndarray,
        theta_ac: np.ndarray, v_supply: np.ndarray, theta_supply_h: np.ndarray, theta_supply_c: np.ndarray,
        l_d_h: np.ndarray, l_d_cs: np.ndarray,
        u_prt: float, a_prt: np.ndarray, a_hcz: np.ndarray, q: float) -> np.ndarray:
    """
    calculate the actual air conditioned temperature
    Args:
        heating_period: heating schedule, operation day represents True, (8760 times)
        cooling_period: cooling schedule, operation day represents True, (8760 times)
        theta_ac: air conditioned temperature, degree C, (8760 times)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        theta_supply_h: supply air temperatures, degree C, (5 rooms * 8760 times)
        theta_supply_c: supply air temperatures, degree C, (5 rooms * 8760 times)
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
        l_d_cs: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2, (5 rooms)
        a_hcz: floor area of heating and cooling zones, m2, (12 rooms)
        q: float,
    Returns:
        actual air conditioned temperature, degree C, (5 rooms * 8760 times)
    """

    rho = get_air_density()
    c = get_specific_heat()

    a_prt = a_prt.reshape(1, 5).T
    a_hcz = a_hcz[0:5].reshape(1, 5).T

    middle_period = (heating_period == cooling_period)

    theta_ac_act_h = np.maximum(theta_ac + (c * rho * v_supply * (theta_supply_h - theta_ac) - l_d_h * 10 ** 6)
                                / (c * rho * v_supply + (u_prt * a_prt + q * a_hcz) * 3600), theta_ac)

    theta_ac_act_c = np.minimum(theta_ac - (c * rho * v_supply * (theta_ac - theta_supply_c) - l_d_cs * 10 ** 6)
                                / (c * rho * v_supply + (u_prt * a_prt + q * a_hcz) * 3600), theta_ac)

    return theta_ac_act_h * heating_period + theta_ac_act_c * cooling_period + theta_ac * middle_period


def get_actual_air_conditioned_absolute_humidity(x_ac: np.ndarray) -> np.ndarray:
    """
    calculate actual air conditioned absolute humidity
    Args:
        x_ac: air conditioned absolute humidity, kg/kgDA (8760 times)
    Returns:
        actual air conditioned absolute humidity, kg/kgDA (5 rooms * 8760 times)
    """

    return np.tile(x_ac, (5, 1))


def get_actual_treated_heating_load(
        heating_period: np.ndarray,
        theta_supply_h: np.ndarray, theta_ac_act_h: np.ndarray, v_supply: np.ndarray) -> np.ndarray:
    """
    Args:
        heating_period: heating period
        theta_supply_h: supply air temperatures, degree C, (5 rooms * 8760 times)
        theta_ac_act_h: air conditioned temperature for heating, degree C, (5 rooms * 8760 times)
        v_supply: supply air volume for heating, m3/h (5 rooms * 8760 times)
    Returns:
        actual treated load for heating, MJ/h, (5 rooms * 8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_d_act_h = (theta_supply_h - theta_ac_act_h) * c * rho * v_supply * 10 ** (-6)

    return l_d_act_h * heating_period


def get_actual_treated_sensible_cooling_load(
        cooling_period: np.ndarray,
        theta_supply_c: np.ndarray, theta_ac_act_c: np.ndarray, v_supply: np.ndarray) -> np.ndarray:
    """
    Args:
        cooling_period: cooling period
        theta_supply_c: supply air temperatures, degree C, (5 rooms * 8760 times)
        theta_ac_act_c: air conditioned temperature for cooling, degree C, (5 rooms * 8760 times)
        v_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
    Returns:
        actual treated sensible load for cooling, MJ/h, (5 rooms * 8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_d_act_cs = (theta_ac_act_c - theta_supply_c) * c * rho * v_supply * 10 ** (-6)

    return l_d_act_cs * cooling_period


def get_actual_treated_latent_cooling_load(
        cooling_period: np.ndarray,
        x_supply_c: np.ndarray, x_ac_act_c: np.ndarray, v_supply: np.ndarray) -> np.ndarray:
    """
    calculate actual treated latent cooling load
    Args:
        cooling_period: cooling period
        x_supply_c: supply air absolute humidity, kg/kgDA (5 rooms * 8760 times)
        x_ac_act_c: air conditioned absolute humidity for cooling, kg/kgDA (5 rooms * 8760 times)
        v_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
    Returns:
        actual treated latent load for cooling, MJ/h (5 rooms * 8760 times)
    """

    # latent heat of evaporation, kJ/kg
    l_wtr = get_evaporation_latent_heat()
    
    rho = get_air_density()

    l_d_act_cl = (x_ac_act_c - x_supply_c) * l_wtr * rho * v_supply * 10 ** (-3)

    return l_d_act_cl * cooling_period


def get_untreated_load(
        l_d_act_h: np.ndarray, l_d_h: np.ndarray,
        l_d_act_cs: np.ndarray, l_d_cs: np.ndarray,
        l_d_act_cl: np.ndarray, l_d_cl: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Args:
        l_d_act_h: actual treated heating load, MJ/h, (5 rooms * 8760 times)
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
        l_d_act_cs: actual treated sensible cooling load, MJ/h, (5 rooms * 8760 times)
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms * 8760 times)
        l_d_act_cl: actual treated latent cooling load, MJ/h, (5 rooms * 8760 times)
        l_d_cl: latent cooling load of occupant room, MJ/h, (5 rooms * 8760 times)
    Returns:
        untreated heating load, MJ/h, (5 rooms * 8760 times)
        untreated sensible cooling load, MJ/h, (5 rooms * 8760 times)
        untreated latent cooling load, MJ/h, (5 rooms * 8760 times)
    """

    # untreated load, MJ/h
    q_ut_h = np.clip(l_d_h - l_d_act_h, 0.0, None)
    q_ut_cs = np.clip(l_d_cs - l_d_act_cs, 0.0, None)
    q_ut_cl = np.clip(l_d_cl - l_d_act_cl, 0.0, None)

    return q_ut_h, q_ut_cs, q_ut_cl


def get_actual_non_occupant_room_temperature(
        theta_d_nac: np.ndarray, theta_ac: np.ndarray, theta_ac_act: np.ndarray,
        v_supply: np.ndarray, v_d_supply: np.ndarray, v_local_nr,
        u_prt: float, a_prt: np.ndarray, q: float, a_nr: float) -> np.ndarray:
    """
    calculate actual non occupant room temperature
    Args:
        theta_d_nac: non occupant room temperature when balanced, degree C (8760 times)
        theta_ac: air conditioned temperature, degree C (8760 times)
        theta_ac_act: air conditioned temperature, degree C, (5 rooms * 8760 times)
        v_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
        v_d_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
        v_local_nr: local ventilation amount in non occupant room, m3/h (8760 times)
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2, (5 rooms)
        q: Q value, W/m2K
        a_nr: floor area of non occupant room, m2
    Returns:
        actual non occupant room temperature, degree C (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    k_prt = c * rho * v_supply / 3600 + u_prt * a_prt.reshape(1, 5).T
    k_d_prt = c * rho * v_d_supply / 3600 + u_prt * a_prt.reshape(1, 5).T
    k_evp = (q - 0.35 * 0.5 * 2.4) * a_nr + c * rho * v_local_nr / 3600

    theta_nac = theta_d_nac\
        + (- np.sum(k_d_prt, axis=0) * (theta_ac - theta_d_nac)
           + np.sum(k_prt * (theta_ac_act - theta_d_nac), axis=0)) / (k_evp + np.sum(k_prt, axis=0))
    return theta_nac


def get_actual_non_occupant_room_absolute_humidity(x_d_nac: np.ndarray) -> np.ndarray:
    """
    calculate actual non occupant room absolute humidity, kg/kgDA
    Args:
        x_d_nac: non occupant room absolute humidity, kg/kgDA (8760 times)
    Returns:
        actual non occupant room absolute humidity, kg/kgDA (8760 times)
    """

    return x_d_nac

# endregion


def get_actual_non_occupant_room_load_for_heating(
        theta_ac_act_h: np.ndarray, theta_nac_h: np.ndarray, v_supply: np.ndarray,
        l_d_h: np.ndarray) -> np.ndarray:
    """
    calculate actual non occupant room heating load
    Args:
        theta_ac_act_h: air conditioned temperature for heating, degree C, (5 rooms * 8760 times)
        theta_nac_h: non occupant room temperature, degree C (8760 times)
        v_supply: supply air volume, m3/h
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
    Returns:
        actual non occupant room heating load, MJ/h, (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_d_act_nac_h = np.sum((theta_ac_act_h - theta_nac_h) * c * rho * v_supply * 10 ** (-6), axis=0)

    return np.where(np.sum(l_d_h, axis=0) > 0.0, l_d_act_nac_h, 0.0)


def get_actual_non_occupant_room_load_for_cooling(
        theta_ac_act_c: np.ndarray, theta_nac_c: np.ndarray, v_supply: np.ndarray,
        l_d_cs: np.ndarray) -> np.ndarray:
    """
    calculate actual non occupant room sensible cooling load
    Args:
        theta_ac_act_c: air conditioned temperature for cooling, degree C, (5 rooms * 8760 times)
        theta_nac_c: non occupant room temperature, degree C (8760 times)
        v_supply: supply air volume, m3/h
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms *  8760 times)
    Returns:
        actual non occupant room sensible cooling load, MJ/h, (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_d_act_nac_cs = np.sum((theta_nac_c - theta_ac_act_c) * c * rho * v_supply * 10 ** (-6), axis=0)

    return np.where(np.sum(l_d_cs, axis=0) > 0.0, l_d_act_nac_cs, 0.0)


def get_actual_heat_loss_through_partition_for_heating(
        u_prt: float, a_prt: np.ndarray, theta_ac_act_h: np.ndarray, theta_nac_h: np.ndarray,
        l_d_h: np.ndarray) -> np.ndarray:
    """
    calculate actual heat loss through the partition
    Args:
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2, (5 rooms)
        theta_ac_act_h: air conditioned temperature for heating, degree C, (5 rooms * 8760 times)
        theta_nac_h: non occupant room temperature, degree C (8760 times)
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
    Returns:
        heat loss through the partition, MJ/h (5 rooms * 8760 times)
    """

    # area of the partition, m2
    a_prt = a_prt.reshape(1, 5).T

    q_trs_prt_h = u_prt * a_prt * (theta_ac_act_h - theta_nac_h) * 3600 * 10 ** (-6)

    return np.where(np.sum(l_d_h, axis=0) > 0.0, q_trs_prt_h, 0.0)


def get_actual_heat_gain_through_partition_for_cooling(
        u_prt: float, a_prt: np.ndarray, theta_ac_act_c: np.ndarray, theta_nac_c: np.ndarray,
        l_d_cs: np.ndarray) -> np.ndarray:
    """
    Args:
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2
        theta_ac_act_c: air conditioned temperature for heating, degree C, (5 rooms * 8760 times)
        theta_nac_c: non occupant room temperature, degree C (8760 times)
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms *  8760 times)
    Returns:
        heat gain through the partition, MJ/h (5 rooms * 8760 times)
    """

    # area of the partition, m2
    a_prt = a_prt.reshape(1, 5).T

    q_trs_prt_c = u_prt * a_prt * (theta_nac_c - theta_ac_act_c) * 3600 * 10 ** (-6)

    return np.where(np.sum(l_d_cs, axis=0) > 0.0, q_trs_prt_c, 0.0)


def get_heat_source_inlet_air_temperature_for_heating(theta_nac_h: np.ndarray) -> np.ndarray:
    """
    get heat source inlet air temperature for heating
    Args:
        theta_nac_h: non occupant room temperature, degree C (8760 times)
    Returns:
        heat source inlet air temperature, degree C (8760 times)
    """

    return theta_nac_h


def get_heat_source_inlet_air_temperature_for_cooling(theta_nac_c: np.ndarray) -> np.ndarray:
    """
    get heat source inlet air temperature for cooling
    Args:
        theta_nac_c: non occupant room temperature, degree C (8760 times)
    Returns:
        heat source inlet air temperature, degree C (8760 times)
    """

    return theta_nac_c


def get_heat_source_heating_output(
        theta_hs_out_h: np.ndarray, theta_hs_in_h: np.ndarray, v_supply: np.ndarray,
        l_d_h: np.ndarray) -> np.ndarray:
    """
    calculate heat source heating output
    Args:
        theta_hs_out_h: supply air temperature, degree C, (5 rooms * 8760 times)
        theta_hs_in_h: inlet air temperature of the heat source for heating, degree C (8760 times)
        v_supply: supply air volume for heating, m3/h (5 rooms * 8760 times)
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
    Returns:
        heating output, MJ/h, (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    q_hs_h = np.maximum((theta_hs_out_h - theta_hs_in_h) * c * rho * np.sum(v_supply, axis=0) * 10 ** (-6), 0.0)

    return np.where(np.sum(l_d_h, axis=0) > 0.0, q_hs_h, 0.0)


def get_heat_source_cooling_output(
        theta_hs_in_c: np.ndarray, theta_hs_out_c: np.ndarray, v_supply: np.ndarray, l_cl: np.ndarray,
        l_d_cs: np.ndarray) -> np.ndarray:
    """
    Args:
        theta_hs_in_c: inlet air temperature of the heat source for cooling, degree C (8760 times)
        theta_hs_out_c: supply air temperature, degree C (8760 times)
        v_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
        l_cl: latent cooling load, MJ/h, (12 rooms * 8760 times)
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms *  8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    q_hs_cs = np.maximum((theta_hs_in_c - theta_hs_out_c) * c * rho * np.sum(v_supply, axis=0) * 10 ** (-6), 0.0)

    q_hs_cl = np.sum(l_cl[0:5], axis=0)

    return np.where(np.sum(l_d_cs, axis=0) > 0.0, q_hs_cs, 0.0), q_hs_cl


# region duct heat balance

def get_upside_temperature_from_load(
        t_sur: float, load: float, v: float, t_ac: float, psi: float, length: float) -> float:
    """
    Calculate the upside air temperature of the duct from the heating load in the room and air supply volume
    Args:
        t_sur: the ambient temperature around the duct, degree C
        load: heating load. positive is heating load and negative is cooling load, MJ/h
        v: supply air volume, m3/h
        t_ac: air conditioned room temperature, degree C
        psi: linear heat loss coefficient, W/mK
        length: duct length, m
    Returns:
        Upside air temperature, degree C
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    return t_sur + (load * 10 ** 6 / (c * rho * v) + t_ac - t_sur) * np.exp(psi * length * 3600 / (c * rho * v))


def get_duct_heat_loss_from_load(
        t_sur: float, load: float, v: float, t_ac: float, psi: float, length: float) -> float:
    """
    Calculate the heat loss from the duct from the heating load in the room and air supply volume
    Args:
        t_sur: the ambient temperature around the duct, degree C
        load: heating load. positive is heating load and negative is cooling load, MJ/h
        v: supply air volume, m3/h
        t_ac: air conditioned room temperature, degree C
        psi: linear heat loss coefficient, W/mK
        length: duct length, m
    Returns:
        heat loss from duct, MJ/h
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    return (load * 10 ** 6 / (c * rho * v) + t_ac - t_sur) * (np.exp(psi * length * 3600 / (c * rho * v)) - 1) \
           * c * rho * v * 10 ** (-6)


def get_downside_temperature_from_load(load: float, v: float, t_ac: float) -> float:
    """
    Calculate the downside air temperature of the duct from the heating load in the room and air supply volume
    Args:
        load: heating load. positive is heating load and negative is cooling load, MJ/h
        v: supply air volume, m3/h
        t_ac: air conditioned room temperature, degree C
    Returns:
        downside air temperature, degree C
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    return load * 10 ** 6 / (c * rho * v) + t_ac


def get_load_from_upside_temperature(
        t_sur: float, t_up: float, v: float, t_ac: float, psi: float, length: float) -> float:
    """
    Calculate the upside air temperature of the duct from the heating load in the room and air supply volume
    Args:
        t_sur: the ambient temperature around the duct, degree C
        t_up: upside air temperature, degree C
        v: supply air volume, m3/h
        t_ac: air conditioned room temperature, degree C
        psi: linear heat loss coefficient, W/mK
        length: duct length, m
    Returns:
        heat load, MJ/h
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    return (t_sur - t_ac + (t_up - t_sur) * np.exp(- psi * length * 3600 / (c * rho * v))) * c * rho * v * 10 ** (-6)


def get_duct_heat_loss_from_upside_temperature(
        t_sur: float, t_up: float, v: float, psi: float, length: float) -> float:
    """
    Calculate the upside air temperature of the duct from the heating load in the room and air supply volume
    Args:
        t_sur: the ambient temperature around the duct, degree C
        t_up: upside air temperature, degree C
        v: supply air volume, m3/h
        psi: linear heat loss coefficient, W/mK
        length: duct length, m
    Returns:
        duct heat loss, MJ/h
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    return (t_up - t_sur) * (1 - np.exp(- psi * length * 3600 / (c * rho * v))) * c * rho * v * 10 ** (-6)


def get_downside_temperature_from_upside_temperature(
        t_sur: float, t_up: float, v: float, psi: float, length: float) -> float:
    """
    Calculate the upside air temperature of the duct from the heating load in the room and air supply volume
    Args:
        t_sur: the ambient temperature around the duct, degree C
        t_up: upside air temperature, degree C
        v: supply air volume, m3/h
        psi: linear heat loss coefficient, W/mK
        length: duct length, m
    Returns:
        downside_temperature, degree C
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    return t_sur + (t_up - t_sur) * np.exp(- psi * length * 3600 / (c * rho * v))

# endregion


# endregion


def get_main_value(
        region: int,
        a_mr: float, a_or: float, a_a: float, r_env: float,
        insulation: str, solar_gain: str,
        default_heat_source_spec: bool,
        v_hs_rtd_h: float, v_hs_rtd_c: float,
        is_duct_insulated: bool, vav_system: bool,
        q_rtd_h: float =None, q_rtd_c: float=None):
    """
    Args:
        region: region, 1-8
        a_mr: main occupant floor area, m2
        a_or: other occupant floor area, m2
        a_a: total floor area, m2
        r_env: ratio of the envelope total area to the total floor area, -
        insulation: insulation level. specify the level as string following below:
            's55': Showa 55 era level
            'h4': Heisei 4 era level
            'h11': Heisei 11 era level
            'h11more': more than Heisei 11 era level
        solar_gain: solar gain level. specify the level as string following below.
            'small': small level
            'middle': middle level
            'large': large level
        default_heat_source_spec: does use the default value for rated heating and cooling capacity ?
        v_hs_rtd_h: rated supply air volume for heating, m3/h
        v_hs_rtd_c: rated supply air volume for cooling, m3/h
        is_duct_insulated: is the duct inside the insulated area or not
        vav_system: is VAV system applied ?
        q_rtd_h: rated heating capacity, W
        q_rtd_c: rated cooling capacity, W
    """

    # region system spec

    # set default value for heating and cooling capacity, W
    if default_heat_source_spec:
        q_rtd_h, q_rtd_c = get_rated_capacity(region, a_a)

    # endregion

    # region house spec

    # floor area of non occupant room, m2
    a_nr = get_non_occupant_room_floor_area(a_mr, a_or, a_a, r_env)

    # referenced floor area, m2, (12 rooms)
    a_hcz_r = get_referenced_floor_area()

    # floor area, m2, (12 rooms)
    a_hcz = get_floor_area(a_mr, a_or, a_a, r_env)

    # the partition area looking from each occupant rooms to the non occupant room, m2, (5 rooms)
    a_prt = get_partition_area(a_hcz, a_mr, a_or, a_nr, r_env)

    # heat loss coefficient of the partition wall, W/m2K
    u_prt = get_heat_loss_coefficient_of_partition()

    # Q value, W/m2K
    # mu_h value, mu_c value (W/m2)/(W/m2)
    q, mu_h, mu_c = get_envelope_spec(region, insulation, solar_gain)

    # mechanical ventilation, m3/h, (5 rooms)
    v_vent = get_mechanical_ventilation(a_hcz_r, a_hcz)

    # endregion

    # region general property

    # air density, kg/m3
    rho = get_air_density()

    # air specific heat, J/kg K
    c = get_specific_heat()

    # latent heat of evaporation, kJ/kg
    l_wtr = get_evaporation_latent_heat()

    # calender
    calender = get_calender()

    # endregion

    # region external conditions

    # outdoor temperature, degree C (8760 times)
    theta_ex = get_outdoor_temperature(region=region)

    # outdoor absolute humidity, kg/kg(DA) (8760 times)
    x_ex = get_absolute_humidity(region)

    # horizontal solar radiation, W/m2K (8760 times)
    j = get_horizontal_solar(region)

    # SAT temperature, degree C, (8760 times)
    theta_sat = get_sat_temperature(region)

    # endregion

    # region occupant usage

    # heating schedule (8760 times), cooling schedule (8760 times)
    heating_period, cooling_period = get_heating_and_cooling_schedule(region)

    # number of people (8760 times)
    n_p, _, _, _ = get_n_p(a_mr, a_or, a_nr, calender)

    # heat generation, W (8760 times)
    q_gen, _, _, _ = get_q_gen(a_mr, a_or, a_nr, calender)

    # moisture generation, g/h (8760 times)
    w_gen, _, _, _ = get_w_gen(a_mr, a_or, a_nr, calender)

    # local ventilation amount, m3/h (8760 times)
    v_local, v_local_mr, v_local_or, v_local_nr = get_v_local(calender)

    # set temperature for heating, degree C, set temperature for cooling, degree C
    theta_set_h, theta_set_c = get_theta_set()

    # set absolute humidity for cooling, kg/kgDA (when 28 degree C and 60 % )
    x_set_c = get_x_set()

    # endregion

    # region circulating air flow

    # heating and cooling output for supply air estimation, MJ/h
    q_d_hs_h = get_heating_output_for_supply_air_estimation(
        a_a, q, mu_h, v_vent, theta_ex, j, heating_period, n_p, q_gen, v_local, theta_set_h)
    q_d_hs_c = get_cooling_output_for_supply_air_estimation(
        a_a, q, mu_c, v_vent, theta_ex, x_ex, j, cooling_period, n_p, q_gen, w_gen, v_local, theta_set_c, x_set_c)

    # minimum supply air volume of the system for heating and cooling, (m3/h, m3/h)
    v_hs_min = get_minimum_air_volume(v_vent)

    # rated heating and cooling output of the heat source, (MJ/h, MJ/h)
    q_hs_rtd_h, q_hs_rtd_c = get_rated_output(q_rtd_h, q_rtd_c)

    # supply air volume of heat source, m3/h
    v_d_hs_supply = get_heat_source_supply_air_volume(
        heating_period, cooling_period, q_d_hs_h, q_d_hs_c, q_hs_rtd_h, q_hs_rtd_c, v_hs_min, v_hs_rtd_h, v_hs_rtd_c)

    # the ratio of the supply air volume valance for each 5 rooms
    r_supply_des = get_supply_air_volume_valance(a_hcz)

    # supply air volume without vav adjustment, m3/h (5 rooms * 8760 times)
    v_d_supply = get_each_supply_air_volume_not_vav_adjust(r_supply_des, v_d_hs_supply, v_vent)

    # endregion

    # region load

    # heating load, and sensible and latent cooling load, MJ/h ((8760times), (8760 times), (8760 times))
    l_h, l_cs, l_cl = get_load(region, insulation, solar_gain, a_mr, a_or, a_a, r_env)

    # heating and cooling room temperature, degree C (8760 times)
    theta_ac = get_air_conditioned_room_temperature(
        heating_period, cooling_period, theta_ex, theta_set_h, theta_set_c)

    # room absolute humidity, kg/kgDA (8760 times)
    x_ac = get_air_conditioned_room_absolute_humidity(cooling_period, x_ex, x_set_c)

    # non occupant room temperature balanced, degree C, (8760 times)
    theta_d_nac = get_non_occupant_room_temperature_balanced(
        heating_period, cooling_period, l_h, l_cs, q, a_nr, v_local_nr, v_d_supply, u_prt, a_prt, theta_ac)

    # non occupant room absolute humidity, kg/kgDA (8760 times)
    x_d_nac = get_non_occupant_room_absolute_humidity_balanced(
        cooling_period, l_cl, v_local_nr, v_d_supply, x_ac)

    # heat transfer through partition from occupant room to non occupant room balanced, MJ/h, (5 rooms * 8760 times)
    q_d_trs_prt = get_heat_transfer_through_partition_balanced(u_prt, a_prt, theta_ac, theta_d_nac)

    # heating and sensible cooling load in the occupant rooms, MJ/h, (5 rooms * 8760 times)
    l_d_h = get_occupant_room_load_for_heating_balanced(l_h, q_d_trs_prt)
    l_d_cs, l_d_cl = get_occupant_room_load_for_cooling_balanced(l_cs, l_cl, q_d_trs_prt)

    # endregion

    # treated and untreated load

    # duct liner heat loss coefficient, W/mK
    psi = get_duct_linear_heat_loss_coefficient()

    # duct length in the standard house, m, ((5 rooms), (5 rooms), (5 rooms))
    l_duct_in_r, l_duct_ex_r, l_duct_r = get_standard_house_duct_length()

    # duct length for each room, m, (5 rooms)
    l_duct = get_duct_length(l_duct_r=l_duct_r, a_a=a_a)

    # attic temperature, degree C, (8760 times)
    theta_attic = get_attic_temperature(theta_sat, theta_ac)

    # duct ambient temperature, degree C, (5 rooms * 8760 times)
    theta_sur = get_duct_ambient_air_temperature(is_duct_insulated, l_duct_in_r, l_duct_ex_r, theta_ac, theta_attic)

    # inlet air temperature of heat source,degree C, (8760 times)
    theta_d_hs_in, x_d_hs_in = get_heat_source_inlet_air_balanced(theta_d_nac, x_d_nac)

    # maximum heating and cooling output, MJ/h (8760 times)
    q_hs_max_h = get_heat_source_maximum_heating_output(region, q_rtd_h)
    q_hs_max_cs, q_hs_max_cl = get_heat_source_maximum_cooling_output(q_rtd_c, l_d_cs, l_d_cl)

    # maximum and minimum temperature and absolute humidity when maximum output of heat sourace
    theta_hs_out_max_h = get_theta_hs_out_max_h(theta_d_hs_in, q_hs_max_h, v_d_supply)
    theta_hs_out_min_c = get_theta_hs_out_min_c(theta_d_hs_in, q_hs_max_cs, v_d_supply)
    x_hs_out_min_c = get_x_hs_out_min_c(x_d_hs_in, q_hs_max_cl, v_d_supply)

    # requested supply air temperature, degree C, (5 rooms * 8760 times)
    theta_req_h = get_requested_supply_air_temperature_for_heating(
        theta_sur, theta_ac, l_d_h, v_d_supply, psi, l_duct)
    theta_req_c = get_requested_supply_air_temperature_for_cooling(
        theta_sur, theta_ac, l_d_cs, v_d_supply, psi, l_duct)
    x_req_c = get_requested_supply_air_absolute_humidity_for_cooling(x_ac, l_d_cl, v_d_supply)

    # outlet temperature of heat source, degree C, (8760 times)
    theta_hs_out_h = get_decided_outlet_supply_air_temperature_for_heating(
        vav_system, theta_req_h, v_d_supply, theta_hs_out_max_h)
    theta_hs_out_c = get_decided_outlet_supply_air_temperature_for_cooling(
        vav_system, theta_req_c, v_d_supply, theta_hs_out_min_c)

    # supply air volume for each room for heating, m3/h, (5 rooms * 8760 times)
    v_supply = get_each_supply_air_volume(
        heating_period, cooling_period, vav_system, l_d_h, l_d_cs, theta_hs_out_h, theta_hs_out_c, theta_sur,
        psi, l_duct, theta_ac, v_vent, v_d_supply)

    # outlet absolute humidity of heat source, kg/kgDA (8760 times)
    x_hs_out_c = get_decided_outlet_supply_air_absolute_humidity_for_cooling(x_req_c, v_supply, x_hs_out_min_c)

    # heat loss from ducts, heat gain to ducts, MJ/h, (5 rooms * 8760 times), reference
    q_loss_duct_h = get_duct_heat_loss_for_heating(theta_sur, theta_hs_out_h, v_supply, psi, l_duct, l_d_h)
    q_gain_duct_c = get_duct_heat_gain_for_cooling(theta_sur, theta_hs_out_c, v_supply, psi, l_duct, l_d_cs)

    # supply air temperature, degree C (5 rooms * 8760 times)
    theta_supply_h = get_supply_air_temperature_for_heating(
        theta_sur, theta_hs_out_h, psi, l_duct, v_supply, theta_ac, l_d_h)
    # supply air temperature, degree C (5 rooms * 8760 times)
    theta_supply_c = get_supply_air_temperature_for_cooling(
        theta_sur, theta_hs_out_c, psi, l_duct, v_supply, theta_ac, l_d_cs)
    # supply air absolute humidity, kg/kgDA (5 rooms * 8760 times)
    x_supply_c = get_supply_air_absolute_humidity_for_cooling(x_hs_out_c, x_ac, l_d_cl)

    # actual air conditioned temperature, degree C, (5 rooms * 8760 times)
    theta_ac_act = get_actual_air_conditioned_temperature(
        heating_period, cooling_period, theta_ac, v_supply, theta_supply_h, theta_supply_c,
        l_d_h, l_d_cs, u_prt, a_prt, a_hcz, q)

    # actual air conditioned absolute humidity, kg/kgDA (5 rooms * 8760 times)
    x_ac_act = get_actual_air_conditioned_absolute_humidity(x_ac)

    # actual treated load for heating, MJ/h, (5 rooms * 8760 times)
    l_d_act_h = get_actual_treated_heating_load(heating_period, theta_supply_h, theta_ac_act, v_supply)
    l_d_act_cs = get_actual_treated_sensible_cooling_load(cooling_period, theta_supply_c, theta_ac_act, v_supply)
    l_d_act_cl = get_actual_treated_latent_cooling_load(cooling_period, x_supply_c, x_ac_act, v_supply)

    # untreated load, MJ/h, (5 rooms * 8760 times, 5 rooms * 8760 times, 5 rooms * 8760 times)
    q_ut_h, q_ut_cs, q_ut_cl = get_untreated_load(
        l_d_act_h, l_d_h, l_d_act_cs, l_d_cs, l_d_act_cl, l_d_cl)

    # actual non occupant room temperature, degree C, (8760 times)
    theta_nac = get_actual_non_occupant_room_temperature(
        theta_d_nac, theta_ac, theta_ac_act, v_supply, v_d_supply, v_local_nr, u_prt, a_prt, q, a_nr)
    # actual non occupant room absolute humidity, kg/kgDA, (8760 times)
    x_nac = get_actual_non_occupant_room_absolute_humidity(x_d_nac)

    # ----------------------------

    # actual non occupant room load, MJ/h, (8760 times)
    l_d_act_nac_h = get_actual_non_occupant_room_load_for_heating(theta_ac_act, theta_nac, v_supply, l_d_h)
    l_d_act_nac_cs = get_actual_non_occupant_room_load_for_cooling(theta_ac_act, theta_nac, v_supply, l_d_cs)

    # actual heat loss or gain through partitions, MJ/h, (5 rooms * 8760 times)
    q_trs_prt_h = get_actual_heat_loss_through_partition_for_heating(u_prt, a_prt, theta_ac_act, theta_nac, l_d_h)
    q_trs_prt_c = get_actual_heat_gain_through_partition_for_cooling(u_prt, a_prt, theta_ac_act, theta_nac, l_d_cs)

    # inlet air temperature of heat source,degree C, (8760 times)
    theta_hs_in_h = get_heat_source_inlet_air_temperature_for_heating(theta_nac)
    theta_hs_in_c = get_heat_source_inlet_air_temperature_for_cooling(theta_nac)

    # output of heat source, MJ/h, (8760 times)
    q_hs_h = get_heat_source_heating_output(theta_hs_out_h, theta_hs_in_h, v_supply, l_d_h)
    q_hs_cs, q_hs_cl = get_heat_source_cooling_output(theta_hs_in_c, theta_hs_out_c, v_supply, l_cl, l_d_cs)

    return {
        'constant_value': {
            'air_density': rho,  # kg/m3
            'air_specific_heat': c,  # J/kgK
            'duct_length_room1': l_duct[0],  # m
            'duct_length_room2': l_duct[1],  # m
            'duct_length_room3': l_duct[2],  # m
            'duct_length_room4': l_duct[3],  # m
            'duct_length_room5': l_duct[4],  # m
            'supply_air_valance_room1': r_supply_des[0],
            'supply_air_valance_room2': r_supply_des[1],
            'supply_air_valance_room3': r_supply_des[2],
            'supply_air_valance_room4': r_supply_des[3],
            'supply_air_valance_room5': r_supply_des[4],
            'mechanical_ventilation_volume_room1': v_vent[0],  # m3/h
            'mechanical_ventilation_volume_room2': v_vent[1],  # m3/h
            'mechanical_ventilation_volume_room3': v_vent[2],  # m3/h
            'mechanical_ventilation_volume_room4': v_vent[3],  # m3/h
            'mechanical_ventilation_volume_room5': v_vent[4],  # m3/h
            'minimum_supply_air_volume_of_heat_source': v_hs_min,  # m3/h
            'partition_area_room1': a_prt[0],  # m2
            'partition_area_room2': a_prt[1],  # m2
            'partition_area_room3': a_prt[2],  # m2
            'partition_area_room4': a_prt[3],  # m2
            'partition_area_room5': a_prt[4],  # m2
            'rated_capacity_heating': q_hs_rtd_h,  # MJ/h
            'rated_capacity_cooling': q_hs_rtd_c,  # MJ/h
        },
        'time_value': {
            'heating_load_room1': l_h[0],  # MJ/h
            'heating_load_room2': l_h[1],  # MJ/h
            'heating_load_room3': l_h[2],  # MJ/h
            'heating_load_room4': l_h[3],  # MJ/h
            'heating_load_room5': l_h[4],  # MJ/h
            'sensible_cooling_load_room1': l_cs[0],  # MJ/h
            'sensible_cooling_load_room2': l_cs[1],  # MJ/h
            'sensible_cooling_load_room3': l_cs[2],  # MJ/h
            'sensible_cooling_load_room4': l_cs[3],  # MJ/h
            'sensible_cooling_load_room5': l_cs[4],  # MJ/h
            'latent_cooling_load_room1': l_cl[0],  # MJ/h
            'latent_cooling_load_room2': l_cl[1],  # MJ/h
            'latent_cooling_load_room3': l_cl[2],  # MJ/h
            'latent_cooling_load_room4': l_cl[3],  # MJ/h
            'latent_cooling_load_room5': l_cl[4],  # MJ/h
            'old_heating_load_sum_of_12_rooms': np.sum(l_h, axis=0),  # MJ/h
            'old_sensible_cooling_load_sum_of_12_rooms': np.sum(l_cs, axis=0),  # MJ/h
            'old_latent_cooling_load_sum_of_12_rooms': np.sum(l_cl, axis=0),  # MJ/h
            'air_conditioned_temperature': theta_ac,  # degree C
            'sat_temperature': theta_sat,  # degree C
            'attic_temperature': theta_attic,  # degree C
            'duct_ambient_temperature_room1': theta_sur[0],  # degree C
            'duct_ambient_temperature_room2': theta_sur[1],  # degree C
            'duct_ambient_temperature_room3': theta_sur[2],  # degree C
            'duct_ambient_temperature_room4': theta_sur[3],  # degree C
            'duct_ambient_temperature_room5': theta_sur[4],  # degree C
            'output_of_heat_source_for_supply_air_volume_estimation_heating': q_d_hs_h,  # MJ/h
            'output_of_heat_source_for_supply_air_volume_estimation_cooling': q_d_hs_c,  # MJ/h
            'supply_air_volume_of_heat_source': v_d_hs_supply,  # MJ/h
            'designed_supply_air_volume_room1': v_d_supply[0],  # MJ/h
            'designed_supply_air_volume_room2': v_d_supply[1],  # MJ/h
            'designed_supply_air_volume_room3': v_d_supply[2],  # MJ/h
            'designed_supply_air_volume_room4': v_d_supply[3],  # MJ/h
            'designed_supply_air_volume_room5': v_d_supply[4],  # MJ/h
            'non_occupant_room_temperature': theta_d_nac,  # degree C
            'heat_loss_through_partition_heating_room1': q_d_trs_prt[0],  # MJ/h
            'heat_loss_through_partition_heating_room2': q_d_trs_prt[1],  # MJ/h
            'heat_loss_through_partition_heating_room3': q_d_trs_prt[2],  # MJ/h
            'heat_loss_through_partition_heating_room4': q_d_trs_prt[3],  # MJ/h
            'heat_loss_through_partition_heating_room5': q_d_trs_prt[4],  # MJ/h
            'maximum_output_heating': q_hs_max_h,  # MJ/h
            'maximum_output_sensible_cooling': q_hs_max_cs,  # MJ/h
            'maximum_output_latent_cooling': q_hs_max_cl,  # MJ/h
            'untreated_heating_load_room1': q_ut_h[0],  # MJ/h
            'untreated_heating_load_room2': q_ut_h[1],  # MJ/h
            'untreated_heating_load_room3': q_ut_h[2],  # MJ/h
            'untreated_heating_load_room4': q_ut_h[3],  # MJ/h
            'untreated_heating_load_room5': q_ut_h[4],  # MJ/h
            'untreated_sensible_cooling_load_room1': q_ut_cs[0],  # MJ/h
            'untreated_sensible_cooling_load_room2': q_ut_cs[1],  # MJ/h
            'untreated_sensible_cooling_load_room3': q_ut_cs[2],  # MJ/h
            'untreated_sensible_cooling_load_room4': q_ut_cs[3],  # MJ/h
            'untreated_sensible_cooling_load_room5': q_ut_cs[4],  # MJ/h
            'untreated_latent_cooling_load_room1': q_ut_cl[0],  # MJ/h
            'untreated_latent_cooling_load_room2': q_ut_cl[1],  # MJ/h
            'untreated_latent_cooling_load_room3': q_ut_cl[2],  # MJ/h
            'untreated_latent_cooling_load_room4': q_ut_cl[3],  # MJ/h
            'untreated_latent_cooling_load_room5': q_ut_cl[4],  # MJ/h
            'duct_upside_supply_air_temperature_heating_room1': theta_req_h[0],  # degree C
            'duct_upside_supply_air_temperature_heating_room2': theta_req_h[1],  # degree C
            'duct_upside_supply_air_temperature_heating_room3': theta_req_h[2],  # degree C
            'duct_upside_supply_air_temperature_heating_room4': theta_req_h[3],  # degree C
            'duct_upside_supply_air_temperature_heating_room5': theta_req_h[4],  # degree C
            'duct_upside_supply_air_temperature_cooling_room1': theta_req_c[0],  # degree C
            'duct_upside_supply_air_temperature_cooling_room2': theta_req_c[1],  # degree C
            'duct_upside_supply_air_temperature_cooling_room3': theta_req_c[2],  # degree C
            'duct_upside_supply_air_temperature_cooling_room4': theta_req_c[3],  # degree C
            'duct_upside_supply_air_temperature_cooling_room5': theta_req_c[4],  # degree C
            'outlet_temperature_of_heat_source_heating': theta_hs_out_h,  # degree C
            'outlet_temperature_of_heat_source_cooling': theta_hs_out_c,  # degree C
            'supply_air_volume_room1': v_supply[0],  # degree C
            'supply_air_volume_room2': v_supply[1],  # degree C
            'supply_air_volume_room3': v_supply[2],  # degree C
            'supply_air_volume_room4': v_supply[3],  # degree C
            'supply_air_volume_room5': v_supply[4],  # degree C
            'duct_heat_loss_heating_room1': q_loss_duct_h[0],  # MJ/h
            'duct_heat_loss_heating_room2': q_loss_duct_h[1],  # MJ/h
            'duct_heat_loss_heating_room3': q_loss_duct_h[2],  # MJ/h
            'duct_heat_loss_heating_room4': q_loss_duct_h[3],  # MJ/h
            'duct_heat_loss_heating_room5': q_loss_duct_h[4],  # MJ/h
            'duct_heat_gain_cooling_room1': q_gain_duct_c[0],  # MJ/h
            'duct_heat_gain_cooling_room2': q_gain_duct_c[1],  # MJ/h
            'duct_heat_gain_cooling_room3': q_gain_duct_c[2],  # MJ/h
            'duct_heat_gain_cooling_room4': q_gain_duct_c[3],  # MJ/h
            'duct_heat_gain_cooling_room5': q_gain_duct_c[4],  # MJ/h
            'supply_air_temperature_heating_room1': theta_supply_h[0],  # degree C
            'supply_air_temperature_heating_room2': theta_supply_h[1],  # degree C
            'supply_air_temperature_heating_room3': theta_supply_h[2],  # degree C
            'supply_air_temperature_heating_room4': theta_supply_h[3],  # degree C
            'supply_air_temperature_heating_room5': theta_supply_h[4],  # degree C
            'supply_air_temperature_cooling_room1': theta_supply_c[0],  # degree C
            'supply_air_temperature_cooling_room2': theta_supply_c[1],  # degree C
            'supply_air_temperature_cooling_room3': theta_supply_c[2],  # degree C
            'supply_air_temperature_cooling_room4': theta_supply_c[3],  # degree C
            'supply_air_temperature_cooling_room5': theta_supply_c[4],  # degree C
            'actual_air_conditioned_temperature_room1': theta_ac_act[0],  # degree C
            'actual_air_conditioned_temperature_room2': theta_ac_act[1],  # degree C
            'actual_air_conditioned_temperature_room3': theta_ac_act[2],  # degree C
            'actual_air_conditioned_temperature_room4': theta_ac_act[3],  # degree C
            'actual_air_conditioned_temperature_room5': theta_ac_act[4],  # degree C
            'actual_treated_heating_load_room1': l_d_act_h[0],  # MJ/h
            'actual_treated_heating_load_room2': l_d_act_h[1],  # MJ/h
            'actual_treated_heating_load_room3': l_d_act_h[2],  # MJ/h
            'actual_treated_heating_load_room4': l_d_act_h[3],  # MJ/h
            'actual_treated_heating_load_room5': l_d_act_h[4],  # MJ/h
            'actual_treated_sensible_cooling_load_room1': l_d_act_cs[0],  # MJ/h
            'actual_treated_sensible_cooling_load_room2': l_d_act_cs[1],  # MJ/h
            'actual_treated_sensible_cooling_load_room3': l_d_act_cs[2],  # MJ/h
            'actual_treated_sensible_cooling_load_room4': l_d_act_cs[3],  # MJ/h
            'actual_treated_sensible_cooling_load_room5': l_d_act_cs[4],  # MJ/h
            'actual_treated_latent_cooling_load_room1': l_d_act_cl[0],  # MJ/h
            'actual_treated_latent_cooling_load_room2': l_d_act_cl[1],  # MJ/h
            'actual_treated_latent_cooling_load_room3': l_d_act_cl[2],  # MJ/h
            'actual_treated_latent_cooling_load_room4': l_d_act_cl[3],  # MJ/h
            'actual_treated_latent_cooling_load_room5': l_d_act_cl[4],  # MJ/h
            'actual_non_occupant_room_temperature': theta_nac,  # degree C
            'actual_non_occupant_room_load_heating': l_d_act_nac_h,  # MJ/h
            'actual_non_occupant_room_load_cooling': l_d_act_nac_cs,  # MJ/h
            'actual_heat_loss_through_partitions_heating_room1': q_trs_prt_h[0],  # MJ/h
            'actual_heat_loss_through_partitions_heating_room2': q_trs_prt_h[1],  # MJ/h
            'actual_heat_loss_through_partitions_heating_room3': q_trs_prt_h[2],  # MJ/h
            'actual_heat_loss_through_partitions_heating_room4': q_trs_prt_h[3],  # MJ/h
            'actual_heat_loss_through_partitions_heating_room5': q_trs_prt_h[4],  # MJ/h
            'actual_heat_gain_through_partitions_heating_room1': q_trs_prt_c[0],  # MJ/h
            'actual_heat_gain_through_partitions_heating_room2': q_trs_prt_c[1],  # MJ/h
            'actual_heat_gain_through_partitions_heating_room3': q_trs_prt_c[2],  # MJ/h
            'actual_heat_gain_through_partitions_heating_room4': q_trs_prt_c[3],  # MJ/h
            'actual_heat_gain_through_partitions_heating_room5': q_trs_prt_c[4],  # MJ/h
            'output_of_heat_source_heating': q_hs_h,  # MJ/h
            'output_of_heat_source_sensible_cooling': q_hs_cs,  # MJ/h
            'output_of_heat_source_latent_cooling': q_hs_cl,  # MJ/h
        },
    }
