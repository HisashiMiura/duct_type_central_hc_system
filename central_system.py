﻿from typing import List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from scipy import optimize

import read_conditions
import envelope
import read_load
import appendix


# region functions


# region system spec

def get_system_spec(region: int, a_a: float, system_spec: dict)\
        -> (str, bool, bool, float, float, float, float, float, float, float,
            float, float, float, float, float, float, float, float, float):
    """
    Args:
        region: region
        a_a: total floor area, m2
        system_spec: system spec dictionary
    Returns:
        input method = 'default', 'rated', 'rated_and_middle'
        is the duct inside the insulated area or not
        is VAV system applied
        rated heating capacity, W
        rated cooling capacity, W
        rated supply air volume for heating, m3/h
        rated supply air volume for cooling, m3/h
        rated power for heating, W
        rated power for cooling, W
        rated fan power for heating, W
        rated fan power for cooling, W
        middle heating capacity, W
        middle cooling capacity, W
        middle supply air volume for heating, m3/h
        middle supply air volume for cooling, m3/h
        middle power for heating, W
        middle power for cooling, W
        middle fan power for heating, W
        middle fan power for cooling, W
    """

    is_duct_insulated = system_spec['is_duct_insulated']
    vav_system = system_spec['vav_system']
    input_method = system_spec['input']
    ventilation_included = system_spec['ventilation_included']

    # set rated value
    if input_method == 'default':
        q_rtd_h, q_rtd_c = appendix.get_default_rated_capacity(region, a_a)
        p_rtd_h, p_rtd_c = appendix.get_default_rated_power(q_rtd_h, q_rtd_c)
        v_hs_rtd_h, v_hs_rtd_c = appendix.get_default_rated_supply_air_volume(q_rtd_h, q_rtd_c)
        p_fan_rtd_h, p_fan_rtd_c = appendix.get_default_rated_fan_power(v_hs_rtd_h, v_hs_rtd_c)
    elif input_method == 'rated' or input_method == 'rated_and_middle':
        q_rtd_h, q_rtd_c = system_spec['cap_rtd_h'], system_spec['cap_rtd_c']
        p_rtd_h, p_rtd_c = system_spec['p_rtd_h'], system_spec['p_rtd_c']
        v_hs_rtd_h, v_hs_rtd_c = system_spec['v_hs_rtd_h'], system_spec['v_hs_rtd_c']
        p_fan_rtd_h, p_fan_rtd_c = system_spec['p_fan_rtd_h'], system_spec['p_fan_rtd_c']
    else:
        raise ValueError

    if input_method == 'default' or input_method == 'rated':
        q_mid_h, q_mid_c = None, None
        v_hs_mid_h, v_hs_mid_c = None, None
        p_mid_h, p_mid_c = None, None
        p_fan_mid_h, p_fan_mid_c = None, None
    elif input_method == 'rated_and_middle':
        q_mid_h, q_mid_c = system_spec['q_mid_h'], system_spec['q_mid_c']
        v_hs_mid_h, v_hs_mid_c = system_spec['v_hs_mid_h'], system_spec['v_hs_mid_c']
        p_mid_h, p_mid_c = system_spec['p_mid_h'], system_spec['p_mid_c']
        p_fan_mid_h, p_fan_mid_c = system_spec['p_fan_mid_h'], system_spec['p_fan_mid_c']
    else:
        raise ValueError

    return input_method, is_duct_insulated, vav_system, ventilation_included,\
        q_rtd_h, q_rtd_c, v_hs_rtd_h, v_hs_rtd_c, p_rtd_h, p_rtd_c, p_fan_rtd_h, p_fan_rtd_c, \
        q_mid_h, q_mid_c, v_hs_mid_h, v_hs_mid_c, p_mid_h, p_mid_c, p_fan_mid_h, p_fan_mid_c

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

def get_heating_and_cooling_schedule(region: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    get the heating and cooling schedule
    operation represents True as boolean type
    Args:
        region: region, 1-8
    Returns:
        heating schedule, operation day represents True, (8760 times)
        cooling schedule, operation day represents True, (8760 times)
        heating and cooling schedule, operation day represents 'h', 'c' or 'm' (8760 times)
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

    hc_period = np.where(heating_period, 'h', np.where(cooling_period, 'c', 'm'))

    return np.repeat(hc_period, 24)


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
        hc_period: np.ndarray, n_p: np.ndarray, q_gen: np.ndarray, v_local: np.ndarray,
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
        hc_period: heating and cooling schedule (8760 times)
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
        0.0) * (hc_period == 'h')

    return q_d_hs_h


def get_cooling_output_for_supply_air_estimation(
        a_a: float, q: float, mu_c: float, v_vent: np.ndarray,
        theta_ex: np.ndarray, x_ex: np.ndarray, j: np.ndarray,
        hc_period: np.ndarray, n_p: np.ndarray, q_gen: np.ndarray, w_gen: np.ndarray, v_local: np.ndarray,
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
        hc_period: heating and cooling schedule (8760 times)
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
        0.0) * (hc_period == 'c')

    q_d_hs_cl = np.maximum(
        (((v_local + sum(v_vent)) * rho * (x_ex - x_set_c) * 10 ** 3 + w_gen) * l_wtr
         + n_p * 40.0 * 3600) * 10 ** (-6), 0.0) * (hc_period == 'c')

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
        hc_period: np.ndarray, ventilation_included: bool,
        q_d_hs_h: np.ndarray, q_d_hs_c: np.ndarray, q_hs_rtd_h: float, q_hs_rtd_c: float,
        v_hs_min: float, v_hs_rtd_h: float, v_hs_rtd_c: float) -> np.ndarray:
    """
    calculate the supply air volume
    Args:
        hc_period: heating and cooling schedule (8760 times)
        ventilation_included: is ventilation included ?
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
        if ventilation_included:
            if q < 0.0:
                return v_hs_min
            elif q < q_hs_rtd:
                return (v_hs_rtd - v_hs_min) / q_hs_rtd * q + v_hs_min
            else:
                return v_hs_rtd
        else:
            if q < 0.0:
                return v_hs_min
            elif q < q_hs_rtd:
                return v_hs_rtd / q_hs_rtd * q + v_hs_min
            else:
                return v_hs_rtd + v_hs_min

    # supply air volume of heat source for heating and cooling, m3/h
    v_d_hs_supply_h = np.vectorize(get_v)(q_d_hs_h, q_hs_rtd_h, v_hs_rtd_h)
    v_d_hs_supply_c = np.vectorize(get_v)(q_d_hs_c, q_hs_rtd_c, v_hs_rtd_c)

    return v_d_hs_supply_h * (hc_period == 'h') + v_d_hs_supply_c * (hc_period == 'c')


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
        hc_period: np.ndarray,
        theta_ex: np.ndarray, theta_set_h: float, theta_set_c: float) -> np.ndarray:
    """
    calculate air conditioned room temperature
    Args:
        hc_period: heating and cooling schedule, operation day represents True, (8760 times)
        theta_ex: outdoor temperature, degree C, (8760 times)
        theta_set_h: set temperature for heating, degree C
        theta_set_c: set temperature for cooling, degree C
    Returns:
        air conditioned room temperature, degree C, (8760 times)
    """

    theta_ac_m = np.clip(theta_ex, theta_set_h, theta_set_c)

    return theta_set_h * (hc_period == 'h') + theta_set_c * (hc_period == 'c') + theta_ac_m * (hc_period == 'm')


def get_air_conditioned_room_absolute_humidity(
        hc_period: np.ndarray, x_ex: np.ndarray, x_set_c: float) -> np.ndarray:
    """
    calculate air conditioned absolute humidity
    Args:
        hc_period: heating and cooling schedule (8760 times)
        x_ex: outdoor absolute humidity, kg/kgDA (8760 times)
        x_set_c: set absolute humidity for cooling, kg/kgDA (=27 degree C and 60%)

    Returns:
        air conditioned room absolute humidity, kg/kgDA (8760 times)
    """

    return x_set_c * (hc_period == 'c') + x_ex * (hc_period != 'c')


def get_non_occupant_room_temperature_balanced(
        hc_period: np.ndarray,
        l_h: np.ndarray, l_cs: np.ndarray,
        q: float, a_nr: float, v_local_nr: np.ndarray,
        v_d_supply: np.ndarray, u_prt: float, a_prt: np.ndarray,
        theta_ac: np.ndarray) -> np.ndarray:
    """
    Args:
        hc_period: heating and cooling schedule, (8760 times)
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

    return theta_nac_h * (hc_period == 'h') + theta_nac_c * (hc_period == 'c') + theta_ac * (hc_period == 'm')


def get_non_occupant_room_absolute_humidity_balanced(
        hc_period: np.ndarray, l_cl: np.ndarray, v_local_nr: np.ndarray, v_d_supply: np.ndarray,
        x_ac: np.ndarray) -> np.ndarray:
    """
        calculate non occupant room absolute humidity
    Args:
        hc_period: heating and cooling schedule (8760 times)
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

    return x_d_nac_c * (hc_period == 'c') + x_ac * (hc_period != 'c')


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

def get_operation(l_d_h, l_d_cs):
    """
    judge operation
    Args:
        l_d_h: heating load of occupant room, MJ/h, (5 rooms * 8760 times)
        l_d_cs: sensible cooling load of occupant room, MJ/h, (5 rooms *  8760 times)

    Returns:
        operation (8760 times) 'h' = heating operation, 'c' = cooling operation, 'n' = operation stop
    """

    return np.where(np.sum(l_d_h, axis=0) > 0.0, 'h', np.where(np.sum(l_d_cs, axis=0) > 0.0, 'c', 'n'))


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
        hc_period: np.ndarray,
        vav_system: bool, l_d_h: np.ndarray, l_d_cs: np.ndarray,
        theta_hs_out_h: np.ndarray, theta_hs_out_c: np.ndarray, theta_sur: np.ndarray,
        psi: float, l_duct: np.ndarray, theta_ac: np.ndarray,
        v_vent: np.ndarray, v_d_supply: np.ndarray, operation: np.ndarray) -> np.ndarray:
    """
    calculate each supply air volume
    Args:
        hc_period: heating and cooling schedule, (8760 times)
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
        operation: operation (8760 times)
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

    v_supply_h = np.where(operation == 'h', v_h, v_vent)
    v_supply_c = np.where(operation == 'c', v_c, v_vent)

    return v_supply_h * (hc_period == 'h') + v_supply_c * (hc_period == 'c') + v_vent * (hc_period == 'm')


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


def get_duct_heat_loss_and_gain(
        theta_sur: np.ndarray, theta_hs_out_h: np.ndarray, theta_hs_out_c: np.ndarray, v_supply: np.ndarray,
        psi: float, l_duct: np.ndarray, operation: np.ndarray) -> np.ndarray:
    """
    calculate the heat loss and gain of the ducts
    Args:
        theta_sur: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_h: outlet temperature of heat source, degree C, (8760 times)
        theta_hs_out_c: outlet temperature of heat source, degree C, (8760 times)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
        operation: operation (8760 times)
    """

    l_duct = np.array(l_duct).reshape(1, 5).T

    q_duct_h = np.where(operation == 'h',
                        get_duct_heat_loss_from_upside_temperature(theta_sur, theta_hs_out_h, v_supply, psi, l_duct),
                        0.0)

    q_duct_c = np.where(operation == 'c',
                        - get_duct_heat_loss_from_upside_temperature(theta_sur, theta_hs_out_c, v_supply, psi, l_duct),
                        0.0)

    return q_duct_h, q_duct_c


def get_supply_air_temperature_for_heating(
        theta_sur: np.ndarray, theta_hs_out_h: np.ndarray, psi: float, l_duct: np.ndarray,
        v_supply: np.ndarray, theta_ac: np.ndarray, operation: np.ndarray) -> np.ndarray:
    """
    calculate supply air temperatures for heating
    Args:
        theta_sur: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_h: outlet temperature of heat source, degree C, (8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        theta_ac: air conditioned temperature, degree C (8760 times)
        operation: operation (8760 times)
    Returns:
        supply air temperatures, degree C, (5 rooms * 8760 times)
    """

    l_duct = np.array(l_duct).reshape(1, 5).T

    theta_supply_h = get_downside_temperature_from_upside_temperature(theta_sur, theta_hs_out_h, v_supply, psi, l_duct)

    return np.where(operation == 'h', theta_supply_h, theta_ac)


def get_supply_air_temperature_for_cooling(
        theta_sur: np.ndarray, theta_hs_out_c: np.ndarray, psi: float, l_duct: np.ndarray,
        v_supply: np.ndarray, theta_ac: np.ndarray, operation: np.ndarray) -> np.ndarray:
    """
    calculate supply air temperatures for cooling
    Args:
        theta_sur: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_c: outlet temperature of heat source, degree C, (8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        theta_ac: air conditioned temperature, degree C (8760 times)
        operation: operation (8760 times)
    Returns:
        supply air temperatures, degree C, (5 rooms * 8760 times)
    """

    l_duct = np.array(l_duct).reshape(1, 5).T

    theta_supply_c = get_downside_temperature_from_upside_temperature(theta_sur, theta_hs_out_c, v_supply, psi, l_duct)

    return np.where(operation == 'c', theta_supply_c, theta_ac)


def get_supply_air_absolute_humidity_for_cooling(
        x_hs_out_c: np.ndarray, x_ac: np.ndarray, operation: np.ndarray) -> np.ndarray:
    """
    calculate supply air absolute humidity for cooling
    Args:
        x_hs_out_c: decided outlet supply air absolute humidity, kg/kgDA (8760 times)
        x_ac: air conditioned absolute humidity, kg/kgDA (8760 times)
        operation: operation (8760 times)
    Returns:
        supply air absolute humidity, kg/kgDA (5 rooms * 8760 times)
    """

    return np.where(operation == 'c', x_hs_out_c, x_ac)


def get_actual_air_conditioned_temperature(
        hc_period: np.ndarray,
        theta_ac: np.ndarray, v_supply: np.ndarray, theta_supply_h: np.ndarray, theta_supply_c: np.ndarray,
        l_d_h: np.ndarray, l_d_cs: np.ndarray,
        u_prt: float, a_prt: np.ndarray, a_hcz: np.ndarray, q: float) -> np.ndarray:
    """
    calculate the actual air conditioned temperature
    Args:
        hc_period: heating and cooling schedule, (8760 times)
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

    theta_ac_act_h = np.maximum(theta_ac + (c * rho * v_supply * (theta_supply_h - theta_ac) - l_d_h * 10 ** 6)
                                / (c * rho * v_supply + (u_prt * a_prt + q * a_hcz) * 3600), theta_ac)

    theta_ac_act_c = np.minimum(theta_ac - (c * rho * v_supply * (theta_ac - theta_supply_c) - l_d_cs * 10 ** 6)
                                / (c * rho * v_supply + (u_prt * a_prt + q * a_hcz) * 3600), theta_ac)

    return theta_ac_act_h * (hc_period == 'h') + theta_ac_act_c * (hc_period == 'c') + theta_ac * (hc_period == 'm')


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
        hc_period: np.ndarray,
        theta_supply_h: np.ndarray, theta_ac_act_h: np.ndarray, v_supply: np.ndarray) -> np.ndarray:
    """
    Args:
        hc_period: heating and cooling period (8760 times)
        theta_supply_h: supply air temperatures, degree C, (5 rooms * 8760 times)
        theta_ac_act_h: air conditioned temperature for heating, degree C, (5 rooms * 8760 times)
        v_supply: supply air volume for heating, m3/h (5 rooms * 8760 times)
    Returns:
        actual treated load for heating, MJ/h, (5 rooms * 8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_d_act_h = (theta_supply_h - theta_ac_act_h) * c * rho * v_supply * 10 ** (-6)

    return l_d_act_h * (hc_period == 'h')


def get_actual_treated_sensible_cooling_load(
        hc_period: np.ndarray,
        theta_supply_c: np.ndarray, theta_ac_act_c: np.ndarray, v_supply: np.ndarray) -> np.ndarray:
    """
    Args:
        hc_period: heating and cooling period (8760 times)
        theta_supply_c: supply air temperatures, degree C, (5 rooms * 8760 times)
        theta_ac_act_c: air conditioned temperature for cooling, degree C, (5 rooms * 8760 times)
        v_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
    Returns:
        actual treated sensible load for cooling, MJ/h, (5 rooms * 8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_d_act_cs = (theta_ac_act_c - theta_supply_c) * c * rho * v_supply * 10 ** (-6)

    return l_d_act_cs * (hc_period == 'c')


def get_actual_treated_latent_cooling_load(
        hc_period: np.ndarray,
        x_supply_c: np.ndarray, x_ac_act_c: np.ndarray, v_supply: np.ndarray) -> np.ndarray:
    """
    calculate actual treated latent cooling load
    Args:
        hc_period: cooling period
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

    return l_d_act_cl * (hc_period == 'c')


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


def get_actual_non_occupant_room_heating_load(
        theta_ac_act: np.ndarray, theta_nac: np.ndarray, v_supply: np.ndarray,
        operation: np.ndarray) -> np.ndarray:
    """
    calculate actual non occupant room heating load
    Args:
        theta_ac_act: air conditioned temperature for heating, degree C, (5 rooms * 8760 times)
        theta_nac: non occupant room temperature, degree C (8760 times)
        v_supply: supply air volume, m3/h
        operation: operation (8760 times)
    Returns:
        actual non occupant room heating load, MJ/h, (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_d_act_nac_h = np.sum((theta_ac_act - theta_nac) * c * rho * v_supply * 10 ** (-6), axis=0)

    return np.where(operation == 'h', l_d_act_nac_h, 0.0)


def get_actual_non_occupant_room_sensible_cooling_load(
        theta_ac_act: np.ndarray, theta_nac: np.ndarray, v_supply: np.ndarray,
        operation: np.ndarray) -> np.ndarray:
    """
    calculate actual non occupant room sensible cooling load
    Args:
        theta_ac_act: air conditioned temperature for cooling, degree C, (5 rooms * 8760 times)
        theta_nac: non occupant room temperature, degree C (8760 times)
        v_supply: supply air volume, m3/h
        operation: operation (8760 times)
    Returns:
        actual non occupant room sensible cooling load, MJ/h, (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    l_d_act_nac_cs = np.sum((theta_nac - theta_ac_act) * c * rho * v_supply * 10 ** (-6), axis=0)

    return np.where(operation == 'c', l_d_act_nac_cs, 0.0)


def get_actual_non_occupant_room_latent_cooling_load(
        x_ac_act_c: np.ndarray, x_nac: np.ndarray, v_supply: np.ndarray, operation: np.ndarray) -> np.ndarray:
    """
    calculate actual non occupant room latent cooling load
    Args:
        x_ac_act_c: air onditioned absolute humidity for cooling, kg/kgDA (5 rooms * 8760 times)
        x_nac: non occupant room absolute humidity, kg/kgDA (8760 times)
        v_supply: supply air volume, m3/h (5 rooms * 8760 times)
        operation: operation (8760 times)
    Returns:
        actual non occupant room latent cooling load, MJ/h (8760 times)
    """

    rho = get_air_density()
    l_wtr = get_evaporation_latent_heat()

    l_d_act_nac_cl = np.sum((x_nac - x_ac_act_c) * rho * l_wtr * v_supply * 10 ** (-3), axis=0)

    return np.where(operation == 'c', l_d_act_nac_cl, 0.0)


def get_actual_heat_loss_through_partition_for_heating(
        u_prt: float, a_prt: np.ndarray, theta_ac_act_h: np.ndarray, theta_nac_h: np.ndarray,
        operation: np.ndarray) -> np.ndarray:
    """
    calculate actual heat loss through the partition
    Args:
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2, (5 rooms)
        theta_ac_act_h: air conditioned temperature for heating, degree C, (5 rooms * 8760 times)
        theta_nac_h: non occupant room temperature, degree C (8760 times)
        operation: operation (8760 times)
    Returns:
        heat loss through the partition, MJ/h (5 rooms * 8760 times)
    """

    # area of the partition, m2
    a_prt = a_prt.reshape(1, 5).T

    q_trs_prt_h = u_prt * a_prt * (theta_ac_act_h - theta_nac_h) * 3600 * 10 ** (-6)

    return np.where(operation == 'h', q_trs_prt_h, 0.0)


def get_actual_heat_gain_through_partition_for_cooling(
        u_prt: float, a_prt: np.ndarray, theta_ac_act_c: np.ndarray, theta_nac_c: np.ndarray,
        operation: np.ndarray) -> np.ndarray:
    """
    Args:
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2
        theta_ac_act_c: air conditioned temperature for heating, degree C, (5 rooms * 8760 times)
        theta_nac_c: non occupant room temperature, degree C (8760 times)
        operation: operation (8760 times)
    Returns:
        heat gain through the partition, MJ/h (5 rooms * 8760 times)
    """

    # area of the partition, m2
    a_prt = a_prt.reshape(1, 5).T

    q_trs_prt_c = u_prt * a_prt * (theta_nac_c - theta_ac_act_c) * 3600 * 10 ** (-6)

    return np.where(operation == 'c', q_trs_prt_c, 0.0)


def get_heat_source_inlet_air_temperature(theta_nac: np.ndarray) -> np.ndarray:
    """
    get heat source inlet air temperature
    Args:
        theta_nac: non occupant room temperature, degree C (8760 times)
    Returns:
        heat source inlet air temperature, degree C (8760 times)
    """

    return theta_nac


def get_heat_source_inlet_air_absolute_humidity(x_nac: np.ndarray) -> np.ndarray:
    """
    get heat source inlet air absolute humidity
    Args:
        x_nac: non occupant room absolute humidity, kg/kgDA (8760 times)
    Returns:
        heat source inlet air absolute humidity, kg/kgDA (8760 times)
    """

    return x_nac


def get_heat_source_heating_output(
        theta_hs_out_h: np.ndarray, theta_hs_in: np.ndarray, v_supply: np.ndarray, operation: np.ndarray) -> np.ndarray:
    """
    calculate heat source heating output
    Args:
        theta_hs_out_h: supply air temperature, degree C, (5 rooms * 8760 times)
        theta_hs_in: inlet air temperature of the heat source, degree C (8760 times)
        v_supply: supply air volume for heating, m3/h (5 rooms * 8760 times)
        operation: operation (8760 times)
    Returns:
        heating output, MJ/h, (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()

    q_hs_h = np.maximum((theta_hs_out_h - theta_hs_in) * c * rho * np.sum(v_supply, axis=0) * 10 ** (-6), 0.0)

    return np.where(operation == 'h', q_hs_h, 0.0)


def get_heat_source_cooling_output(
        theta_hs_in: np.ndarray, x_hs_in: np.ndarray, theta_hs_out_c: np.ndarray, x_hs_out_c: np.ndarray,
        v_supply: np.ndarray, operation: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Args:
        theta_hs_in: inlet air temperature of the heat source for cooling, degree C (8760 times)
        x_hs_in: inlet air absolute humidity of the heat source, kg/kgDA (8760 times)
        theta_hs_out_c: supply air temperature, degree C (8760 times)
        x_hs_out_c: supply air absolute humidity, kg/kgDA (8760 times)
        v_supply: supply air volume for cooling, m3/h (5 rooms * 8760 times)
        operation: operation (8760 times)
    Returns:
        cooling output, MJ/h (8760 times)
        sensible cooling output, MJ/h (8760 times)
        latent cooling output, MJ/h (8760 times)
    """

    c = get_specific_heat()
    rho = get_air_density()
    l_wtr = get_evaporation_latent_heat()

    q_hs_cs = np.where(operation == 'c',
                       np.maximum(
                           (theta_hs_in - theta_hs_out_c) * c * rho * np.sum(v_supply, axis=0) * 10 ** (-6), 0.0),
                       0.0)

    q_hs_cl = np.where(operation == 'c',
                       np.maximum((x_hs_in - x_hs_out_c) * rho * l_wtr * np.sum(v_supply, axis=0) * 10 ** (-3), 0.0),
                       0.0)

    q_hs_c = q_hs_cs + q_hs_cl

    return q_hs_c, q_hs_cs, q_hs_cl

# endregion


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


# region theoretical efficiency


def get_a_f_hex() -> float:
    """
    Returns:
        effective area for heat exchange of front projected area of heat exchanger of the internal unit, m2
    """

    return 0.23559


def get_a_e_hex() -> float:
    """
    Returns:
        effective area for heat exchange of the surface area of heat exchanger or the internal unit, m2
    """

    return 6.396


def get_alpha_c_hex_h(v_hs: float) -> float:
    """
    calculate sensible heat transfer coefficient on the surface of the heat exchanger of the internal unit
    Args:
        v_hs: air supply volume of heat source, m3/h
    Returns:
        sensible heat transfer coefficient on the surface of the heat exchanger of the internal unit, W/m2K
    """

    # effective area for heat exchange of front projected area of heat exchanger of the internal unit, m2
    a_f_hex = get_a_f_hex()

    v = v_hs / 3600 / a_f_hex

    return (- 0.0017 * v ** 2 + 0.044 * v + 0.0271) * 1000


def get_alpha_c_hex_c(v_hs: float, x_hs_in_c: float) -> (float, float):
    """
    Args:
        v_hs: air volume of heat source, m3/h
        x_hs_in_c: inlet air absolute humidity of the heat source for cooling, kg/kgDA
    Returns:
        sensible heat transfer coefficient on the surface of the heat exchanger of the internal unit, kW/m2K
        latent heat transfer coefficient on the surface of the heat exchanger of the internal unit, kg/m2s
    """

    # effective area for heat exchange of front projected area of heat exchanger of the internal unit, m2
    a_f_hex = get_a_f_hex()

    v_hs = np.clip(v_hs, 400.0, None)

    alpha_d_hex_c = 0.050 * np.log(v_hs / 3600 / a_f_hex) + 0.073

    # specific heat of the vapour at constant pressure, kJ/kg K
    c_pw = 1846

    # specific heat of the dry air  at constant pressure, kJ/kg K
    c_p_air = 1006

    alpha_c_hex_c = alpha_d_hex_c * (c_p_air + c_pw * x_hs_in_c)

    return alpha_c_hex_c, alpha_d_hex_c


def get_theta_surf_hex_test_h(theta_hs_in_h: float, q_hs_h: float, v_hs: float) -> float:
    """
    calculate surface temperature of heat exchanger for heating
    Args:
        theta_hs_in_h: inlet air temperature of heat source for heating, degree C
        q_hs_h: refrigerant cycle capacity for heating, W
        v_hs: air supply volume of heat source, m3/h
    Returns:
        surface temperature of the heat exchanger of the internal unit for heating, degree C
    """

    c = get_specific_heat()
    rho = get_air_density()

    theta_hs_out_h = theta_hs_in_h + q_hs_h / (c * rho * v_hs) * 3600

    # sensible heat transfer coefficient on the surface of the heat exchanger of the internal unit, W/m2K
    alpha_c_hex_h = get_alpha_c_hex_h(v_hs)

    # effective area for heat exchange of the surface area of heat exchanger or the internal unit, m2
    a_e_hex = get_a_e_hex()

    return (theta_hs_in_h + theta_hs_out_h) / 2 + q_hs_h / (a_e_hex * alpha_c_hex_h)


def get_q_hs_test_c(
        theta_hs_in_c: float, x_hs_in_c: float, theta_surf_hex_c: float, v_hs: float) -> (float, float, float):
    """
    Args:
        theta_hs_in_c: inlet air temperature of heat source for cooling, degree C
        x_hs_in_c: inlet air absolute humidity of heat source for cooling, kg/kgDA
        theta_surf_hex_c: surface temperature of the heat exchanger of the internal unit for cooling, degree C
        v_hs: air supply volume of heat source, m3/h
    Returns:
        cooling capacity, W
        sensible cooling capacity, W
        latent cooling capacity, W
    """

    c = get_specific_heat()
    rho = get_air_density()

    # latent heat of evaporation, kJ/kg
    l_wtr = get_evaporation_latent_heat()

    x_surf_hex_c = get_saturated_absolute_humidity(theta_surf_hex_c)

    # effective area for heat exchange of the surface area of heat exchanger or the internal unit, m2
    a_e_hex = get_a_e_hex()

    alpha_c_hex_c, alpha_d_hex_c = get_alpha_c_hex_c(v_hs, x_hs_in_c)

    q_hs_cs = (theta_hs_in_c - theta_surf_hex_c) / (3600/2/c/rho/v_hs + 1/a_e_hex/alpha_c_hex_c)

    if x_hs_in_c > x_surf_hex_c:
        q_hs_cl = (x_hs_in_c - x_surf_hex_c) / (3600/2/l_wtr/rho/v_hs/10**3 + 1/l_wtr/a_e_hex/alpha_d_hex_c/10**3)
    else:
        q_hs_cl = 0.0

    q_hs = q_hs_cs + q_hs_cl

    return q_hs, q_hs_cs, q_hs_cl


def get_theta_surf_hex_test_c(theta_hs_in_c: float, x_hs_in_c: float, v_hs: float, q_hs_c: float) -> float:
    """
    Args:
        theta_hs_in_c: inlet air temperature of heat source for cooling, degree C
        x_hs_in_c: inlet air absolute humidity of heat source for cooling, kg/kgDA
        v_hs: air supply volume of heat source, m3/h
        q_hs_c: refrigerant cycle capacity for cooling, W
    Returns:
        surface temperature of the heat exchanger of the internal unit for cooling, degree C
    """

    def f(theta_surf_hex_c):
        return get_q_hs_test_c(theta_hs_in_c, x_hs_in_c, theta_surf_hex_c, v_hs)[0] - q_hs_c

    return float(optimize.fsolve(f, 0.0)[0])


def get_theta_surf_hex_h(theta_hs_in_h: float, theta_hs_out_h: float, v_hs: float) -> float:
    """
    calculate surface temperature of heat exchanger for heating
    Args:
        theta_hs_in_h: inlet air temperature of heat source for heating, degree C
        theta_hs_out_h: outlet air temperature of heat source for heating, degree C
        v_hs: air supply volume of heat source, m3/h
    Returns:
        surface temperature of the heat exchanger of the internal unit for heating, degree C
    """

    c = get_specific_heat()
    rho = get_air_density()

    # sensible heating capacity of heat source for heating, W
    q_hs_h = (theta_hs_out_h - theta_hs_in_h) * c * rho * v_hs / 3600

    # sensible heat transfer coefficient on the surface of the heat exchanger of the internal unit, W/m2K
    alpha_c_hex_h = get_alpha_c_hex_h(v_hs)

    # effective area for heat exchange of the surface area of heat exchanger or the internal unit, m2
    a_e_hex = get_a_e_hex()

    return (theta_hs_in_h + theta_hs_out_h) / 2 + q_hs_h / (a_e_hex * alpha_c_hex_h)


def get_theta_surf_hex_c(theta_hs_in_c: float, x_hs_in_c: float, theta_hs_out_c: float, v_hs: float) -> float:
    """
    calculate surface temperature of heat exchanger for cooling
    Args:
        theta_hs_in_c: inlet air temperature of heat source for cooling, degree C
        x_hs_in_c: inlet air absolute humidity of heat source for cooling, kg/kgDA
        theta_hs_out_c: outlet air temperature of heat source for cooling, degree C
        v_hs: air supply volume of heat source, m3/h
    Returns:
        surface temperature of the heat exchanger of the internal unit for cooling, degree C
    """

    c = get_specific_heat()
    rho = get_air_density()

    # sensible cooling capacity of heat source for heating, W
    q_hs_c = (theta_hs_in_c - theta_hs_out_c) * c * rho * v_hs / 3600

    # sensible heat transfer coefficient on the surface of the heat exchanger of the internal unit, W/m2K
    alpha_c_hex_c = get_alpha_c_hex_c(v_hs, x_hs_in_c)[0]

    # effective area for heat exchange of the surface area of heat exchanger or the internal unit, m2
    a_e_hex = get_a_e_hex()

    return (theta_hs_in_c + theta_hs_out_c) / 2 - q_hs_c / (a_e_hex * alpha_c_hex_c)


def get_refrigerant_temperature_heating(theta_ex: float, theta_surf_hex_h: float) -> (float, float, float, float):
    """
    calculate temperatures of evaporator, condenser, super heat and sub cool for heating
    Args:
        theta_ex: outdoor temperature, degree C
        theta_surf_hex_h: surface temperature of the heat exchanger in the internal unit, degree C
    Returns:
        evaporator temperature, degree C
        condenser temperature, degree C
        super heat temperature, degree C
        sub cool temperature, degree C
    """

    theta_ref_cnd_h = np.clip(theta_surf_hex_h, None, 65.0)
#    theta_ref_evp_h = np.clip(theta_ex - (0.1 * theta_ref_cnd_h + 2.95), -50.0, None)
    theta_ref_evp_h = np.clip(theta_ex - (0.1 * theta_ref_cnd_h + 2.95), -50.0, theta_ref_cnd_h - 5.0)
    theta_ref_sc_h = 0.245 * theta_ref_cnd_h - 1.72
    theta_ref_sh_h = 4.49 - 0.036 * theta_ref_cnd_h

    return theta_ref_evp_h, theta_ref_cnd_h, theta_ref_sh_h, theta_ref_sc_h


def get_refrigerant_temperature_cooling(theta_ex: float, theta_surf_hex_c: float) -> (float, float, float, float):
    """
    calculate temperatures of evaporator, condenser, super heat and sub cool for cooling
    Args:
        theta_ex: outdoor temperature, degree C
        theta_surf_hex_c: surface temperature of the heat exchanger in the internal unit, degree C
    Returns:
        evaporator temperature, degree C
        condenser temperature, degree C
        super heat temperature, degree C
        sub cool temperature, degree C
    """

    theta_ref_evp_c = np.clip(theta_surf_hex_c, -50.0, None)
#    theta_ref_cnd_c = np.clip(np.clip(theta_ex + 27.4 - 1.35 * theta_ref_evp_c, theta_ex, None), None, 65.0)
    theta_ref_cnd_c = np.clip(np.clip(theta_ex + 27.4 - 1.35 * theta_ref_evp_c, theta_ex, None), theta_ref_evp_c + 5.0, 65.0)
    theta_ref_sc_c = np.clip(0.772 * theta_ref_cnd_c - 25.6, 0.0, None)
    theta_ref_sh_c = np.clip(0.194 * theta_ref_cnd_c - 3.86, 0.0, None)

    return theta_ref_evp_c, theta_ref_cnd_c, theta_ref_sh_c, theta_ref_sc_c


def get_heat_pump_theoretical_efficiency_heating(
        theta_ref_evp: float, theta_ref_cnd: float, theta_ref_sh: float, theta_ref_sc: float) -> float:
    """
    calculate theoretical efficiency
    Args:
        theta_ref_evp: temperature of evaporator, degree C
        theta_ref_cnd: temperature of condenser, degree C
        theta_ref_sh:  temperature of super heat, degree C
        theta_ref_sc: temperature of sub cool, degree C
    Returns:
        theoretical heating efficiency of heat pump
    """

    e_th_h, _ = get_heat_pump_theoretical_efficiency(theta_ref_evp, theta_ref_cnd, theta_ref_sh, theta_ref_sc)

    return e_th_h


def get_heat_pump_theoretical_efficiency_cooling(
        theta_ref_evp: float, theta_ref_cnd: float, theta_ref_sh: float, theta_ref_sc: float) -> float:
    """
    calculate theoretical efficiency
    Args:
        theta_ref_evp: temperature of evaporator, degree C
        theta_ref_cnd: temperature of condenser, degree C
        theta_ref_sh:  temperature of super heat, degree C
        theta_ref_sc: temperature of sub cool, degree C
    Returns:
        theoretical cooling efficiency of heat pump
    """

    _, e_th_c = get_heat_pump_theoretical_efficiency(theta_ref_evp, theta_ref_cnd, theta_ref_sh, theta_ref_sc)

    return e_th_c


def get_heat_pump_theoretical_efficiency(
        theta_ref_evp: float, theta_ref_cnd: float, theta_ref_sh: float, theta_ref_sc: float) -> (float, float):
    """
    calculate theoretical efficiency
    Args:
        theta_ref_evp: temperature of evaporator, degree C
        theta_ref_cnd: temperature of condenser, degree C
        theta_ref_sh:  temperature of super heat, degree C
        theta_ref_sc: temperature of sub cool, degree C
    Returns:
        theoretical heating efficiency of heat pump
        theoretical cooling efficiency of heat pump
    """

    # pressure of evaporator, MPa
    p_ref_evp = get_f_p_sgas(theta_ref_evp)

    # pressure of condenser, MPa
    p_ref_cnd = get_f_p_sgas(theta_ref_cnd)

    # temperature of condenser outlet, degree C
    theta_ref_cnd_out = theta_ref_cnd - theta_ref_sc

    # specific enthalpy of condenser outlet, kJ/kg
    h_ref_cnd_out = get_f_h_liq(p_ref_cnd, theta_ref_cnd_out)

    # temperature of compressor inlet, degree C
    theta_ref_comp_in = theta_ref_evp + theta_ref_sh

    # pressure of compressor inlet, MPa
    p_ref_comp_in = p_ref_evp

    # specific enthalpy of compressor inlet, kJ/kg
    h_ref_comp_in = get_f_h_gas_comp_in(p_ref_comp_in, theta_ref_comp_in)

    # specific entropy of compressor inlet, kJ/kg K
    s_ref_comp_in = get_f_s_gas(p_ref_comp_in, h_ref_comp_in)

    # specific entropy of compressor outlet, kJ/kg K
    s_ref_comp_out = s_ref_comp_in

    # pressure of compressor outlet, MPa
    p_ref_comp_out = p_ref_cnd

    # specific enthalpy of compressor outlet, kJ/kg
    h_ref_comp_out = get_f_h_gas_comp_out(p_ref_comp_out, s_ref_comp_out)

    # theoretical heating efficiency of heat pump
    e_ref_h_th = (h_ref_comp_out - h_ref_cnd_out)/(h_ref_comp_out - h_ref_comp_in)

    # theoretical cooling efficiency of heat pump
    e_ref_c_th = e_ref_h_th - 1

    # maximum e_ref_h_th value is 25.0
    # maximum e_ref_c_th value is 24.0
#    e_ref_h_th = np.clip(e_ref_h_th, 0.0, 25.0)
#    e_ref_c_th = np.clip(e_ref_c_th, 0.0, 24.0)

    return e_ref_h_th, e_ref_c_th


def get_saturated_absolute_humidity(theta: float) -> float:
    """
    Args:
        theta: temperature, degree C
    Returns:
        saturated absolute humidity, kg/kgDA
    """

    p_vs = get_saturated_vapour_pressure_by_temperature(theta)

    return 0.622 * p_vs / (101325 - p_vs)


def get_vapour_pressure_by_absolute_humidity(x: float) -> float:
    """
    calculate vapour pressure
    Args:
        x: absolute humidity, kg/kgDA
    Returns:
        vapour pressure, Pa
    """

    # convert unit from kg/kgDA to g/kgDA
    x = x * 1000

    # vapour pressure, Pa
    p_v = 101325 * x / (622 + x)

    return p_v


def get_saturated_vapour_pressure_by_temperature(theta: float) -> float:
    """
    calculate relative humidity
    Args:
        theta: temperature, degree C
    Returns:
        saturated vapour pressure, Pa
    """

    # absolute temperature, K
    t = theta + 273.16

    a1 = -6096.9385
    a2 = 21.2409642
    a3 = -0.02711193
    a4 = 0.00001673952
    a5 = 2.433502
    b1 = -6024.5282
    b2 = 29.32707
    b3 = 0.010613863
    b4 = -0.000013198825
    b5 = -0.49382577

    # saturated vapour pressure, Pa
    k = np.where(theta > 0.0,
                 a1 / t + a2 + a3 * t + a4 * t ** 2 + a5 * np.log(t),
                 b1 / t + b2 + b3 * t + b4 * t ** 2 + b5 * np.log(t))
    p_vs = np.e ** k

    return p_vs


def get_e_th_h_test(
        theta_hs_in_h: float, q_hs_h: float, v_hs: float, theta_ex: float) -> (float, float, float, float, float):
    """
    Args:
        theta_hs_in_h: inlet air temperature, degree C
        q_hs_h: sensible heating capacity, W
        v_hs: supply air volume, m3/h
        theta_ex: external temperature, degree C
    Returns:
        theoretical heat pump efficiency
        evaporator temperature, degree C
        condenser temperature, degree C
        super heat, degree C
        sub cool, degree C
    """

    # surface temperature, degree C
    theta_surf_hex_test_h = get_theta_surf_hex_test_h(theta_hs_in_h, q_hs_h, v_hs)

    # evaporator temperature, condenser temperature, super heat temperature, suc cool temperature, degree C
    theta_ref_evp_h, theta_ref_cnd_h, theta_ref_sh_h, theta_ref_sc_h \
        = get_refrigerant_temperature_heating(theta_ex, theta_surf_hex_test_h)

    e_th_h_test = get_heat_pump_theoretical_efficiency_heating(
        theta_ref_evp_h, theta_ref_cnd_h, theta_ref_sh_h, theta_ref_sc_h)

    return e_th_h_test, theta_ref_evp_h, theta_ref_cnd_h, theta_ref_sh_h, theta_ref_sc_h


def get_e_th_c_test(
        theta_hs_in_c: float, x_hs_in_c: float, q_hs_c: float, v_hs: float, theta_ex: float
                    ) -> (float, float, float, float, float):
    """
    Args:
        theta_hs_in_c: inlet air temperature, degree C
        x_hs_in_c: inlet air absolute humidity, kg/kgDA
        q_hs_c: sensible and latent cooling capacity, W
        v_hs: supply air volume, m3/h
        theta_ex: external temperature, degree C
    Returns:
        theoretical heat pump efficiency
        evaporator temperature, degree C
        condenser temperature, degree C
        super heat, degree C
        sub cool, degree C
    """

    # surface temperature, degree C
    theta_surf_hex_test_c = get_theta_surf_hex_test_c(theta_hs_in_c, x_hs_in_c, v_hs, q_hs_c)

    # evaporator temperature, condenser temperature, super heat temperature, suc cool temperature, degree C
    theta_ref_evp_c, theta_ref_cnd_c, theta_ref_sh_c, theta_ref_sc_c = get_refrigerant_temperature_cooling(
        theta_ex, theta_surf_hex_test_c)

    # theoretical efficiency
    e_th_c_test = get_heat_pump_theoretical_efficiency_cooling(
        theta_ref_evp_c, theta_ref_cnd_c, theta_ref_sh_c, theta_ref_sc_c)

    return e_th_c_test, theta_ref_evp_c, theta_ref_cnd_c, theta_ref_sh_c, theta_ref_sc_c


# region heat pump cycle

def get_f_p_sgas(theta: float) -> float:
    """
    calculate saturated vapor pressure
    Args:
        theta: saturated vapor temperature, degree C
    Return:
        saturated vapor pressure, MPa
    """

    return 2.75857926950901 * 10 ** (-17) * theta ** 8 \
        + 1.49382057911753 * 10 ** (-15) * theta ** 7 \
        + 6.52001687267015 * 10 ** (-14) * theta ** 6 \
        + 9.14153034999975 * 10 ** (-12) * theta ** 5 \
        + 3.18314616500361 * 10 ** (-9) * theta ** 4 \
        + 1.60703566663019 * 10 ** (-6) * theta ** 3 \
        + 3.06278984019513 * 10 ** (-4) * theta ** 2 \
        + 2.54461992992037 * 10 ** (-2) * theta \
        + 7.98086455154775 * 10 ** (-1)


def get_f_h_gas_comp_in(p: float, theta: float) -> float:
    """
    calculate specific enthalpy of the heated vapour at the condition of the compressor inlet
    Args:
        p: pressure of the over heated vapour at the condition of the compressor inlet, MPa
        theta: temperature of the over heated vapour at the condition of the compressor inlet, degree C
    Return:
        specific enthalpy of the over heated vapour, kJ/kg
    """

    return -1.00110355 * 10 ** (-1) * p ** 3 \
        - 1.184450639 * 10 * p ** 2 \
        - 2.052740252 * 10 ** 2 * p \
        + 3.20391 * 10 ** (-6) * (theta + 273.15) ** 3 \
        - 2.24685 * 10 ** (-3) * (theta + 273.15) ** 2 \
        + 1.279436909 * (theta + 273.15) \
        + 3.1271238 * 10 ** (-2) * p ** 2 * (theta + 273.15) \
        - 1.415359 * 10 ** (-3) * p * (theta + 273.15) ** 2 \
        + 1.05553912 * p * (theta + 273.15)+1.949505039 * 10 ** 2


def get_f_h_gas_comp_out(p: float, s: float) -> float:
    """
    calculate specific enthalpy of the over heated vapour at the condition of the compressor outlet
    Args:
        p: pressure of the over heated vapour at the condition of the compressor outlet, MPa
        s: specific entropy of the over heated vapour at the condition of the compressor outlet, kJ/kg K
    Returns:
        specific enthalpy of the over heated vapour at the condition of the copressor outlet, kJ/kg
    """

    return - 1.869892835947070 * 10 ** (-1) * p ** 4 \
        + 8.223224182177200 * 10 ** (-1) * p ** 3 \
        + 4.124595239531860 * p ** 2 \
        - 8.346302788803210 * 10 * p \
        - 1.016388214044490 * 10 ** 2 * s ** 4 \
        + 8.652428629143880 * 10 ** 2 * s ** 3 \
        - 2.574830800631310 * 10 ** 3 * s ** 2 \
        + 3.462049327009730 * 10 ** 3 * s \
        + 9.209837906396910 * 10 ** (-1) * p ** 3 * s \
        - 5.163305566700450 * 10 ** (-1) * p ** 2 * s ** 2 \
        + 4.076727767130210 * p * s ** 3 \
        - 8.967168786520070 * p ** 2 * s \
        - 2.062021416757910 * 10 * p * s ** 2 \
        + 9.510257675728610 * 10 * p * s \
        - 1.476914346214130 * 10 ** 3


def get_f_s_gas(p: float, h: float) -> float:
    """
    calculate specific entropy of the over heated vapour
    Args:
        p: pressure of the over heated vapour, MPa
        h: specific enthalpy of the over heated vapour, kJ/kg
    Returns:
        specific entropy, kJ/kg K
    """
    return 5.823109493752840 * 10 ** (-2) * p ** 4 \
        - 3.309666523931270 * 10 ** (-1) * p ** 3 \
        + 7.700179914440890 * 10 ** (-1) * p ** 2 \
        - 1.311726004718660 * p \
        + 1.521486605815750 * 10 ** (-9) * h ** 4 \
        - 2.703698863404160 * 10 ** (-6) * h ** 3 \
        + 1.793443775071770 * 10 ** (-3) * h ** 2 \
        - 5.227303746767450 * 10 ** (-1) * h \
        + 1.100368875131490 * 10 ** (-4) * p ** 3 * h \
        + 5.076769807083600 * 10 ** (-7) * p ** 2 * h ** 2 \
        + 1.202580329499520 * 10 ** (-8) * p * h ** 3 \
        - 7.278049214744230 * 10 ** (-4) * p ** 2 * h \
        - 1.449198550965620 * 10 ** (-5) * p * h ** 2 \
        + 5.716086851760640 * 10 ** (-3) * p * h \
        + 5.818448621582900 * 10


def get_f_h_liq(p: float, theta: float) -> float:
    """
    calculate specific enthalpy of the over cooled liquid, kJ/kg
    Args:
        p: pressure of the over cooled liquid, MPa
        theta: temperature of the over cooled liquid, degree C
    Returns:
        specific enthalpy of the over cooled liquid, kJ/kg
    """

    return 1.7902915 * 10 ** (-2) * p ** 3 \
        + 7.96830322 * 10 ** (-1) * p ** 2 \
        + 5.985874958 * 10 * p \
        + 0 * (theta + 273.15) ** 3 \
        + 9.86677 * 10 ** (-4) * (theta + 273.15) ** 2 \
        + 9.8051677 * 10 ** (-1) * (theta + 273.15) \
        - 3.58645 * 10 ** (-3) * p ** 2 * (theta + 273.15) \
        + 8.23122 * 10 ** (-4) * p * (theta + 273.15) ** 2 \
        - 4.42639115 * 10 ** (-1) * p * (theta + 273.15) \
        - 1.415490404 * 10 ** 2

# endregion

# endregion


# region energy


def get_comp_eta_rtd_h(q_hs_rtd_h: float, v_hs_rtd_h: float, p_hs_rtd_h: float, p_fan_rtd_h: float)\
        -> (float, float, float, float, float):
    """
    Args:
        q_hs_rtd_h: rated heating capacity, W
        v_hs_rtd_h: rated air supply volume for heating, m3/h
        p_hs_rtd_h: rated power for heating, W
        p_fan_rtd_h: rated fan power for heating, W
    Returns:
        rated compression efficiency of compressor for heating
        rated compressor efficiency for heating
        rated theoretical efficiency of heat pump cycle for heating
        evaporator temperature in rated condition for heating, degree C
        condenser temperature in rated condition for heating, degree C
    """

    e_comp_th_rtd_h, theta_ref_evp_h, theta_ref_cnd_h, theta_ref_sh_h, theta_ref_sc_h = get_e_th_h_test(
        theta_hs_in_h=20.0, q_hs_h=q_hs_rtd_h, v_hs=v_hs_rtd_h, theta_ex=7.0)
    e_comp_rtd_h = q_hs_rtd_h / (p_hs_rtd_h - p_fan_rtd_h)
    eta_comp_rtd_h = e_comp_rtd_h / e_comp_th_rtd_h
    eta_comp_rtd_h = np.clip(eta_comp_rtd_h, 0.0, 1.0)
    return eta_comp_rtd_h, e_comp_rtd_h, e_comp_th_rtd_h, theta_ref_evp_h, theta_ref_cnd_h


def get_comp_eta_mid_h(
        input_method: str, eta_comp_rtd_h: float, q_hs_mid_h: Optional[float],
        v_hs_mid_h: Optional[float], p_hs_mid_h: Optional[float], p_fan_mid_h: Optional[float])\
        -> (float, Union[float, str], Union[float, str], Union[float, str], Union[float, str]):
    """
    Args:
        input_method: 'default', 'rated' or 'rated_and_middle'
        eta_comp_rtd_h: rated compression efficiency of compressor for heating
        q_hs_mid_h: middle heating capacity [option], W
        v_hs_mid_h: middle air supply volume for heating [option], m3/h
        p_hs_mid_h: middle power for heating [option], W
        p_fan_mid_h: middle fan power for heating [option], W
    Returns:
        middle compression efficiency of compressor for heating
        middle compressor efficiency for heating [option]
        middle theoretical efficiency of heat  pump cycle for heating [option]
        evaporator temperature in middle condition for heating [option], degree C
        condenser temperature in middle condition for heating [option], degree C
    """

    if input_method == 'rated_and_middle':
        e_comp_th_mid_h, theta_ref_evp_h, theta_ref_cnd_h, theta_ref_sh_h, theta_ref_sc_h = get_e_th_h_test(
            theta_hs_in_h=20.0, q_hs_h=q_hs_mid_h, v_hs=v_hs_mid_h, theta_ex=7.0)
        e_comp_mid_h = q_hs_mid_h / (p_hs_mid_h - p_fan_mid_h)
        eta_comp_mid_h = e_comp_mid_h / e_comp_th_mid_h
        eta_comp_mid_h = np.clip(eta_comp_mid_h, 0.0, 1.0)
        return eta_comp_mid_h, e_comp_mid_h, e_comp_th_mid_h, theta_ref_evp_h, theta_ref_cnd_h
    else:
        return eta_comp_rtd_h * 0.95, None, None, None, None


def get_q_hs_mid_h(input_method: str, q_hs_rtd_h: float, q_hs_mid_h: Optional[float]) -> float:
    """
    Args:
        input_method: 'default', 'rated' or 'rated_and_middle'
        q_hs_rtd_h: rated heating capacity, W
        q_hs_mid_h: middle heating capacity [option], W
    Returns:
        middle heating capacity, W
    """

    if input_method == 'rated_and_middle':
        return q_hs_mid_h
    else:
        return q_hs_rtd_h * 0.5


def get_comp_eta_min_h(eta_comp_rtd_h: float) -> float:
    """
    Args:
        eta_comp_rtd_h: rated compression efficiency of compressor for heating
    Returns:
        minimum compression efficiency of compressor for heating
    """

    return eta_comp_rtd_h * 0.65


def get_q_hs_min_h(q_hs_rtd_h: float) -> float:
    """
    Args:
        q_hs_rtd_h: rated heating capacity, W
    Returns:
        minimum heating capacity, W
    """

    return q_hs_rtd_h * 0.35


def get_comp_eta_rtd_c(q_hs_rtd_c: float, v_hs_rtd_c: float, p_hs_rtd_c: float, p_fan_rtd_c: float)\
        -> (float, float, float, float):
    """
    Args:
        q_hs_rtd_c: rated cooling capacity, W
        v_hs_rtd_c: rated air supply volume for cooling, m3/h
        p_hs_rtd_c: rated power for cooling, W
        p_fan_rtd_c: rated fan power for cooling, W
    Returns:
        rated compression efficiency of compressor for cooling
        rated compressor efficiency for cooling
        rated theoretical efficiency of heat pump cycle for cooling
        evaporator temperature in rated condition for cooling, degree C
        condenser temperature in rated condition for cooling, degree C
    """

    e_comp_th_rtd_c, theta_ref_evp_c, theta_ref_cnd_c, theta_ref_sh_c, theta_ref_sc_c = get_e_th_c_test(
        theta_hs_in_c=27.0, x_hs_in_c=0.010376, q_hs_c=q_hs_rtd_c, v_hs=v_hs_rtd_c, theta_ex=35.0)
    e_comp_rtd_c = q_hs_rtd_c / (p_hs_rtd_c - p_fan_rtd_c)
    eta_comp_rtd_c = e_comp_rtd_c / e_comp_th_rtd_c
    eta_comp_rtd_c = np.clip(eta_comp_rtd_c, 0.0, 1.0)
    return eta_comp_rtd_c, e_comp_rtd_c, e_comp_th_rtd_c, theta_ref_evp_c, theta_ref_cnd_c


def get_comp_eta_mid_c(
        input_method: str, eta_comp_rtd_c: float, q_hs_mid_c: Optional[float],
        v_hs_mid_c: Optional[float], p_hs_mid_c: Optional[float], p_fan_mid_c: Optional[float])\
        -> (float, Union[float, str], Union[float, str], Union[float, str], Union[float, str]):
    """
    Args:
        input_method: 'default', 'rated' or 'rated_and_middle'
        eta_comp_rtd_c: rated compression efficiency of compressor for cooling
        q_hs_mid_c: middle cooling capacity [option], W
        v_hs_mid_c: middle air supply volume for cooling [option], m3/h
        p_hs_mid_c: middle power for cooling [option], W
        p_fan_mid_c: middle fan power for cooling [option], W
    Returns:
        middle compression efficiency of compressor for cooling
        middle compressor efficiency for cooling [option]
        middle theoretical efficiency of heat  pump cycle for cooling [option]
        evaporator temperature in middle condition for cooling [option], degree C
        condenser temperature in middle condition for cooling [option], degree C
    """

    if input_method == 'rated_and_middle':
        e_comp_th_mid_c, theta_ref_evp_c, theta_ref_cnd_c, theta_ref_sh_c, theta_ref_sc_c = get_e_th_c_test(
            theta_hs_in_c=27.0, x_hs_in_c=0.010376, q_hs_c=q_hs_mid_c, v_hs=v_hs_mid_c, theta_ex=35.0)
        e_comp_mid_c = q_hs_mid_c / (p_hs_mid_c - p_fan_mid_c)
        eta_comp_mid_c = e_comp_mid_c / e_comp_th_mid_c
        eta_comp_mid_c = np.clip(eta_comp_mid_c, 0.0, 1.0)
        return eta_comp_mid_c, e_comp_mid_c, e_comp_th_mid_c, theta_ref_evp_c, theta_ref_cnd_c
    else:
        return eta_comp_rtd_c * 0.95, None, None, None, None


def get_q_hs_mid_c(input_method: str, q_hs_rtd_c: float, q_hs_mid_c: Optional[float]) -> float:
    """
    Args:
        input_method: 'default', 'rated' or 'rated_and_middle'
        q_hs_rtd_c: rated cooling capacity, W
        q_hs_mid_c: middle cooling capacity [option], W
    Returns:
        middle cooling capacity, W
    """

    if input_method == 'rated_and_middle':
        return q_hs_mid_c
    else:
        return q_hs_rtd_c * 0.5


def get_comp_eta_min_c(eta_comp_rtd_c: float) -> float:
    """
    Args:
        eta_comp_rtd_c: rated compression efficiency of compressor for cooling
    Returns:
        minimum compression efficiency of compressor for cooling
    """

    return eta_comp_rtd_c * 0.65


def get_q_hs_min_c(q_hs_rtd_c: float) -> float:
    """
    Args:
        q_hs_rtd_c: rated cooling capacity, W
    Returns:
        minimum cooling capacity, W
    """

    return q_hs_rtd_c * 0.35


def get_heat_source_heating_capacity(
        theta_ex: np.ndarray, h_ex: np.ndarray, operation: np.ndarray,
        theta_hs_out_h: np.ndarray, theta_hs_in: np.ndarray, v_supply: np.ndarray):
    """
    Args:
        theta_ex: outdoor temperature (8760 times), degree C
        h_ex: outdoor relative humidity (8760 times), kg/kgDA
        operation: operation (8760 times) = 'h', 'c' or 'm'
        theta_hs_out_h: supply air temperature (8760 times), degree C
        theta_hs_in: inlet air temperature of the heat source (8760 times), degree C
        v_supply: supply air volume for heating (5 rooms * 8760 times), m3/h
    Returns:
        heating capacity (8760 times), W
    """

    # coefficient for defrosting, (8760 times)
    c_df_h = np.where((theta_ex < 5.0) & (h_ex >= 80.0), 0.77, 1.0)

    c = get_specific_heat()
    rho = get_air_density()

    q_hs_cap_h = np.maximum(
        (theta_hs_out_h - theta_hs_in) * c * rho * np.sum(v_supply, axis=0) / 3600 / c_df_h, 0.0)

    return np.where(operation == 'h', q_hs_cap_h, 0.0)


def get_heat_source_cooling_capacity(
        theta_hs_in: np.ndarray, x_hs_in: np.ndarray, theta_hs_out_c: np.ndarray, x_hs_out_c: np.ndarray,
        v_supply: np.ndarray, operation: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Args:
        theta_hs_in: inlet air temperature of the heat source for cooling (8760 times), degree C
        x_hs_in: inlet air absolute humidity of the heat source (8760 times), kg/kgDA
        theta_hs_out_c: supply air temperature (8760 times), degree C
        x_hs_out_c: supply air absolute humidity (8760 times), kg/kgDA
        v_supply: supply air volume for cooling (5 rooms * 8760 times), m3/h
        operation: operation (8760 times)
    Returns:
        cooling capacity (8760 times), W
        sensible cooling capacity (8760 times), W
        latent cooling capacity (8760 times), W
    """

    c = get_specific_heat()
    rho = get_air_density()
    l_wtr = get_evaporation_latent_heat()

    q_hs_cap_cs = np.maximum((theta_hs_in - theta_hs_out_c) * c * rho * np.sum(v_supply, axis=0) / 3600, 0.0)
    q_hs_cap_cs = np.where(operation == 'c', q_hs_cap_cs, 0.0)

    q_hs_cap_cl = np.maximum((x_hs_in - x_hs_out_c) * rho * l_wtr * np.sum(v_supply, axis=0) * 10**3 / 3600, 0.0)
    q_hs_cap_cl = np.where(operation == 'c', q_hs_cap_cl, 0.0)

    q_hs_cap_c = q_hs_cap_cs + q_hs_cap_cl

    return q_hs_cap_c, q_hs_cap_cs, q_hs_cap_cl


def get_e_th_h(
        operation: np.ndarray,
        theta_hs_in_h: np.ndarray, theta_hs_out_h: np.ndarray, v_hs: np.ndarray, theta_ex: np.ndarray)\
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Args:
        operation: operation mode (8760 times)
        theta_hs_in_h: inlet air temperature (8760 times), degree C
        theta_hs_out_h: outlet air temperature (8760 times), degree C
        v_hs: supply air volume (8760 times), m3/h
        theta_ex: external temperature (8760 times), degree C
    Returns:
        theoretical heat pump efficiency (8760 times)
        evaporator temperature (8760 times), degree C
        condenser temperature (8760 times), degree C
    """

    # surface temperature, degree C
    theta_surf_hex_h = get_theta_surf_hex_h(theta_hs_in_h, theta_hs_out_h, v_hs)

    # evaporator temperature, condenser temperature, super heat temperature, suc cool temperature, degree C
    theta_ref_evp_h, theta_ref_cnd_h, theta_ref_sh_h, theta_ref_sc_h = get_refrigerant_temperature_heating(
        theta_ex, theta_surf_hex_h)

    e_th_h = get_heat_pump_theoretical_efficiency_heating(
        theta_ref_evp_h, theta_ref_cnd_h, theta_ref_sh_h, theta_ref_sc_h)

    e_th_h = np.where(operation == 'h', e_th_h, 0.0)
    theta_ref_evp_h = np.where(operation == 'h', theta_ref_evp_h, 0.0)
    theta_ref_cnd_h = np.where(operation == 'h', theta_ref_cnd_h, 0.0)

    return e_th_h, theta_ref_evp_h, theta_ref_cnd_h


def get_e_th_c(
        operation: np.ndarray, theta_hs_in_c: np.ndarray, x_hs_in_c: np.ndarray,
        theta_hs_out_c: np.ndarray, v_hs: np.ndarray, theta_ex: np.ndarray)\
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Args:
        operation: operation mode (8760 times)
        theta_hs_in_c: inlet air temperature (8760 times), degree C
        x_hs_in_c: inlet air absolute humidity (8760 times), kg/kgDA
        theta_hs_out_c: outlet air temperature (8760 times), degree C
        v_hs: supply air volume (8760 times), m3/h
        theta_ex: external temperature (8760 times), degree C
    Returns:
        theoretical heat pump efficiency (8760 times)
        evaporator temperature (8760 times), degree C
        condenser temperature (8760 times), degree C
    """

    # surface temperature, degree C
    theta_surf_hex_c = get_theta_surf_hex_c(theta_hs_in_c, x_hs_in_c, theta_hs_out_c, v_hs)

    # evaporator temperature, condenser temperature, super heat temperature, suc cool temperature, degree C
    theta_ref_evp_c, theta_ref_cnd_c, theta_ref_sh_c, theta_ref_sc_c = get_refrigerant_temperature_cooling(
        theta_ex, theta_surf_hex_c)

    e_th_c = get_heat_pump_theoretical_efficiency_cooling(
        theta_ref_evp_c, theta_ref_cnd_c, theta_ref_sh_c, theta_ref_sc_c)

    e_th_c = np.where(operation == 'c', e_th_c, 0.0)
    theta_ref_evp_c = np.where(operation == 'c', theta_ref_evp_c, 0.0)
    theta_ref_cnd_c = np.where(operation == 'c', theta_ref_cnd_c, 0.0)

    return e_th_c, theta_ref_evp_c, theta_ref_cnd_c


def get_eta_comp_h(q_hs_h: np.ndarray, eta_comp_min_h: float, eta_comp_mid_h: float, eta_comp_rtd_h: float,
                   q_hs_min_h: float, q_hs_mid_h: float, q_hs_rtd_h: float) -> np.ndarray:
    """
    Args:
        q_hs_h: average heating capacity (8760 times), W
        eta_comp_min_h: compression efficiency of compressor at the minimum heating condition
        eta_comp_mid_h: compression efficiency of compressor at the middle heating condition
        eta_comp_rtd_h: compression efficiency of compressor at the rated heating condition
        q_hs_min_h: minimum heating capacity, W
        q_hs_mid_h: middle heating capacity, W
        q_hs_rtd_h: rated heating capacity, W
    Returns:
        compression efficiency of compressor for cooling

    """

    def f(q):
        if q <= q_hs_min_h:
            return eta_comp_min_h - (q_hs_min_h - q) * eta_comp_min_h / q_hs_min_h
        elif q <= q_hs_mid_h:
            return eta_comp_mid_h - (q_hs_mid_h - q) * (eta_comp_mid_h - eta_comp_min_h) / (q_hs_mid_h - q_hs_min_h)
        elif q <= q_hs_rtd_h:
            return eta_comp_rtd_h - (q_hs_rtd_h - q) * (eta_comp_rtd_h - eta_comp_mid_h) / (q_hs_rtd_h - q_hs_mid_h)
        else:
            return max(0.4, eta_comp_rtd_h - (q - q_hs_rtd_h) * eta_comp_rtd_h / q_hs_rtd_h)

    return np.vectorize(f)(q_hs_h)


def get_eta_comp_c(q_hs_c: np.ndarray, eta_comp_min_c: float, eta_comp_mid_c: float, eta_comp_rtd_c: float,
                   q_hs_min_c: float, q_hs_mid_c: float, q_hs_rtd_c: float) -> np.ndarray:
    """
    Args:
        q_hs_c: average cooling capacity (8760 times), W
        eta_comp_min_c: compression efficiency of compressor at the minimum cooling condition
        eta_comp_mid_c: compression efficiency of compressor at the middle cooling condition
        eta_comp_rtd_c: compression efficiency of compressor at the rated cooling condition
        q_hs_min_c: minimum cooling capacity, W
        q_hs_mid_c: middle cooling capacity, W
        q_hs_rtd_c: rated cooling capacity, W
    Returns:
        compression efficiency of compressor for cooling
    """

    def f(q):
        if q <= q_hs_min_c:
            return eta_comp_min_c - (q_hs_min_c - q) * eta_comp_min_c / q_hs_min_c
        elif q <= q_hs_mid_c:
            return eta_comp_mid_c - (q_hs_mid_c - q) * (eta_comp_mid_c - eta_comp_min_c) / (q_hs_mid_c - q_hs_min_c)
        elif q <= q_hs_rtd_c:
            return eta_comp_rtd_c - (q_hs_rtd_c - q) * (eta_comp_rtd_c - eta_comp_mid_c) / (q_hs_rtd_c - q_hs_mid_c)
        else:
            return max(0.4, eta_comp_rtd_c - (q - q_hs_rtd_c) * eta_comp_rtd_c / q_hs_rtd_c)

    return np.vectorize(f)(q_hs_c)


def get_e_comp_h(e_comp_th_h: np.ndarray, eta_comp_h: np.ndarray) -> np.ndarray:
    """
    Args:
        e_comp_th_h: theoretical heat pump efficiency for heating (8760 times)
        eta_comp_h: compression efficiency of compressor for heating (8760 times)
    Returns:
        compressor efficiency for heating (8760 times)
    """

    return e_comp_th_h * eta_comp_h


def get_e_comp_c(e_comp_th_c: np.ndarray, eta_comp_c: np.ndarray) -> np.ndarray:
    """
    Args:
        e_comp_th_c: theoretical heat pump efficiency for cooling (8760 times)
        eta_comp_c: compression efficiency of compressor for cooling (8760 times)
    Returns:
        compressor efficiency for heating (8760 times)
    """

    return e_comp_th_c * eta_comp_c


def get_e_e_comp_h(q_hs_cap_h: np.ndarray, e_comp_h: np.ndarray) -> np.ndarray:
    """
    Args:
        q_hs_cap_h: heating capacity (8760 times), W
        e_comp_h: compression efficiency of compressor for heating (8760 times)
    Returns:
        Energy consumption of compressor for heating (8760 times), kWh/h
    """

    def f(q, e):
        if q > 0.0:
            return q / e * 10 ** (-3)
        else:
            return 0.0

    return np.vectorize(f)(q_hs_cap_h, e_comp_h)


def get_e_e_comp_c(q_hs_cap_c: np.ndarray, e_comp_c: np.ndarray) -> np.ndarray:
    """
    Args:
        q_hs_cap_c: cooling capacity (8760 times), W
        e_comp_c: compression efficiency of compressor for cooling (8760 times)
    Returns:
        Energy consumption of compressor for cooling, kWh/h
    """

    def f(q, e):
        if q > 0.0:
            return q / e * 10 ** (-3)
        else:
            return 0.0

    return np.vectorize(f)(q_hs_cap_c, e_comp_c)


def get_e_e_fan_h(
        operation: np.ndarray,
        ventilation_included: bool, p_fan_rtd_h: float, v_vent: np.ndarray, v_supply: np.ndarray, v_fan_rtd_h: float
) -> np.ndarray:
    """
    Args:
        operation: operation
        ventilation_included: is ventilation included ?
        p_fan_rtd_h: rated fan power for heating, W
        v_vent: mechanical ventilation amount (5 rooms), m3/h
        v_supply: air supply volume(5 rooms * 8760 times), m3/h
        v_fan_rtd_h: air supply volume of heat source in the rated heating capacity operation, m3/h
    Returns:
        the amount added to the heating system of the fan power, kWh/h
    """

    vent = np.sum(v_vent)

    f_sfp = 0.4 * 0.36

    if ventilation_included:
        e_e_fan_h = (p_fan_rtd_h - f_sfp * vent) * (np.sum(v_supply, axis=0) - vent) / (v_fan_rtd_h - vent) * 10 ** (-3)
    else:
        e_e_fan_h = p_fan_rtd_h * (np.sum(v_supply, axis=0) - vent) / v_fan_rtd_h * 10 ** (-3)

    return np.where(operation == 'h', e_e_fan_h, 0.0)


def get_e_e_fan_c(
        operation: np.ndarray,
        ventilation_included: bool, p_fan_rtd_c: float, v_vent: np.ndarray, v_supply: np.ndarray, v_fan_rtd_c: float
) -> np.ndarray:
    """
    Args:
        operation: operation
        ventilation_included: is ventilation inculded ?
        p_fan_rtd_c: rated fan power for cooling, W
        v_vent: mechanical ventilation amount (5 rooms * 8760 times), m3/h
        v_supply: air supply volume(5 rooms * 8760 times), m3/h
        v_fan_rtd_c: air supply volume of heat source in the rated cooling capacity operation, m3/h
    Returns:
        the amount added to the cooling system of the fan power, kWh/h
    """

    vent = np.sum(v_vent)

    f_sfp = 0.4 * 0.36

    if ventilation_included:
        e_e_fan_c = (p_fan_rtd_c - f_sfp * vent) * (np.sum(v_supply, axis=0) - vent) / (v_fan_rtd_c - vent) * 10 ** (-3)
    else:
        e_e_fan_c = p_fan_rtd_c * (np.sum(v_supply, axis=0) - vent) / v_fan_rtd_c * 10 ** (-3)

    return np.where(operation == 'c', e_e_fan_c, 0.0)


def get_e_e_h(e_e_comp_h: np.ndarray, e_e_fan_h: np.ndarray) -> np.ndarray:
    """
    Args:
        e_e_comp_h: energy consumption of compressor for heating, kWh/h (8760 times)
        e_e_fan_h: energy consumption of fan for heating, kWh/h (8760 times)
    Returns:
        energy consumption for heating, kWh/h (8760 times)
    """

    return e_e_comp_h + e_e_fan_h


def get_e_e_c(e_e_comp_c: np.ndarray, e_e_fan_c: np.ndarray) -> np.ndarray:
    """
    Args:
        e_e_comp_c: energy consumption of compressor for heating, kWh/h (8760 times)
        e_e_fan_c: energy consumption of fan for cooling, kWh/h (8760 times)
    Returns:
        energy consumption for cooling, kWh/h (8760 times)
    """

    return e_e_comp_c + e_e_fan_c


def get_e(
        region: int, e_e_h: np.ndarray, e_e_c: np.ndarray,
        q_ut_h: np.ndarray, q_ut_cs: np.ndarray, q_ut_cl: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Args:
        region: region
        e_e_h: energy consumption for heating (8760 times), kWh/h
        e_e_c: energy consumption for cooling (8760 times), kWh/h
        q_ut_h: untreated heating load of heating system, MJ/h
        q_ut_cs: untreated sensible cooling load of cooling system, MJ/h
        q_ut_cl: untreated latent cooling load of cooling system, MJ/h
    Returns:
        designed primary energy consumption of heating system, MJ/h
        designed primary energy consumption equivalent to untreated load of heating system, MJ/h
        designed primary energy consumption of cooling system, MJ/h
        designed primary energy consumption equivalent to untreated load of cooling system, MJ/h
    """

    # conversion coefficient, kJ/kWh
    f_prim = 9760

    # equivalent conversion coefficient
    alpha_ut_h = {
        1: 1.61,
        2: 1.46,
        3: 1.32,
        4: 1.30,
        5: 1.20,
        6: 1.09,
        7: 1.12,
        8: None
    }[region]

    alpha_ut_c = 1.0

    e_h_t = e_e_h * f_prim * 10 ** (-3)
    e_h_ut = np.sum(q_ut_h, axis=0) * alpha_ut_h

    e_c_t = e_e_c * f_prim * 10 ** (-3)
    e_c_ut = np.sum(q_ut_cs + q_ut_cl, axis=0) * alpha_ut_c

    return e_h_t, e_h_ut, e_c_t, e_c_ut


# endregion

# endregion


class OutputData:

    def __init__(self):

        self.l_duct = None
        self.r_supply_des = None
        self.v_vent = None
        self.v_hs_min = None
        self.a_prt = None
        self.q_hs_rtd_h = None
        self.q_hs_rtd_c = None

        self.theta_ex = None
        self.x_ex = None
        self.h_ex = None
        self.j = None
        self.theta_sat = None
        self.q_d_hs_h = None
        self.q_d_hs_c = None
        self.v_d_hs_supply = None
        self.v_d_supply = None
        self.l_h = None
        self.l_cs = None
        self.l_cl = None
        self.theta_ac = None
        self.x_ac = None
        self.theta_d_nac = None
        self.x_d_nac = None
        self.q_d_trs_prt = None
        self.l_d_h = None
        self.l_d_cs = None
        self.l_d_cl = None
        self.theta_attic = None
        self.theta_sur = None
        self.theta_d_hs_in = None
        self.x_d_hs_in = None
        self.q_hs_max_h = None
        self.q_hs_max_cs = None
        self.q_hs_max_cl = None
        self.theta_hs_out_max_h = None
        self.theta_hs_out_min_c = None
        self.x_hs_out_min_c = None
        self.theta_req_h = None
        self.theta_req_c = None
        self.x_req_c = None
        self.theta_hs_out_h = None
        self.theta_hs_out_c = None
        self.v_supply = None
        self.x_hs_out_c = None
        self.q_loss_duct_h = None
        self.q_gain_duct_c = None
        self.theta_supply_h = None
        self.theta_supply_c = None
        self.x_supply_c = None
        self.theta_ac_act = None
        self.x_ac_act = None
        self.l_d_act_h = None
        self.l_d_act_cs = None
        self.l_d_act_cl = None
        self.q_ut_h = None
        self.q_ut_cs = None
        self.q_ut_cl = None
        self.theta_nac = None
        self.x_nac = None
        self.l_d_act_nac_h = None
        self.l_d_act_nac_cs = None
        self.l_d_act_nac_cl = None
        self.q_trs_prt_h = None
        self.q_trs_prt_c = None
        self.theta_hs_in = None
        self.x_hs_in = None
        self.q_hs_h = None
        self.q_hs_c = None
        self.q_hs_cs = None
        self.q_hs_cl = None

    def get_constant_value_dict(self):

        return {
            'duct_length_room1': self.l_duct[0],  # m
            'duct_length_room2': self.l_duct[1],  # m
            'duct_length_room3': self.l_duct[2],  # m
            'duct_length_room4': self.l_duct[3],  # m
            'duct_length_room5': self.l_duct[4],  # m
            'supply_air_valance_room1': self.r_supply_des[0],
            'supply_air_valance_room2': self.r_supply_des[1],
            'supply_air_valance_room3': self.r_supply_des[2],
            'supply_air_valance_room4': self.r_supply_des[3],
            'supply_air_valance_room5': self.r_supply_des[4],
            'mechanical_ventilation_volume_room1': self.v_vent[0],  # m3/h
            'mechanical_ventilation_volume_room2': self.v_vent[1],  # m3/h
            'mechanical_ventilation_volume_room3': self.v_vent[2],  # m3/h
            'mechanical_ventilation_volume_room4': self.v_vent[3],  # m3/h
            'mechanical_ventilation_volume_room5': self.v_vent[4],  # m3/h
            'minimum_supply_air_volume_of_heat_source': self.v_hs_min,  # m3/h
            'partition_area_room1': self.a_prt[0],  # m2
            'partition_area_room2': self.a_prt[1],  # m2
            'partition_area_room3': self.a_prt[2],  # m2
            'partition_area_room4': self.a_prt[3],  # m2
            'partition_area_room5': self.a_prt[4],  # m2
            'rated_capacity_heating': self.q_hs_rtd_h,  # MJ/h
            'rated_capacity_cooling': self.q_hs_rtd_c,  # MJ/h
        }

    def get_time_value_dict(self):

        return {
            'outdoor temperature': self.theta_ex,  # degree C
            'outdoor absolute humidity': self.x_ex,  # kg/kgDA
            'outdoor relative humidity': self.h_ex,  # %
            'horizontal solar radiation': self.j,  # W/m2K
            'SAT temperature': self.theta_sat,  # degree C
            'output_of_heat_source_for_supply_air_volume_estimation_heating': self.q_d_hs_h,  # MJ/h
            'output_of_heat_source_for_supply_air_volume_estimation_cooling': self.q_d_hs_c,  # MJ/h
            'supply_air_volume_of_heat_source': self.v_d_hs_supply,  # MJ/h
            'designed_supply_air_volume_room1': self.v_d_supply[0],  # MJ/h
            'designed_supply_air_volume_room2': self.v_d_supply[1],  # MJ/h
            'designed_supply_air_volume_room3': self.v_d_supply[2],  # MJ/h
            'designed_supply_air_volume_room4': self.v_d_supply[3],  # MJ/h
            'designed_supply_air_volume_room5': self.v_d_supply[4],  # MJ/h
            'old_heating_load_room1': self.l_h[0],  # MJ/h
            'old_heating_load_room2': self.l_h[1],  # MJ/h
            'old_heating_load_room3': self.l_h[2],  # MJ/h
            'old_heating_load_room4': self.l_h[3],  # MJ/h
            'old_heating_load_room5': self.l_h[4],  # MJ/h
            'old_heating_load_sum_of_occupant_rooms': np.sum(self.l_h[0:5], axis=0),  # MJ/h
            'old_sensible_cooling_load_room1': self.l_cs[0],  # MJ/h
            'old_sensible_cooling_load_room2': self.l_cs[1],  # MJ/h
            'old_sensible_cooling_load_room3': self.l_cs[2],  # MJ/h
            'old_sensible_cooling_load_room4': self.l_cs[3],  # MJ/h
            'old_sensible_cooling_load_room5': self.l_cs[4],  # MJ/h
            'old_sensible_cooling_load_sum_of_occupant_rooms': np.sum(self.l_cs[0:5], axis=0),  # MJ/h
            'old_latent_cooling_load_room1': self.l_cl[0],  # MJ/h
            'old_latent_cooling_load_room2': self.l_cl[1],  # MJ/h
            'old_latent_cooling_load_room3': self.l_cl[2],  # MJ/h
            'old_latent_cooling_load_room4': self.l_cl[3],  # MJ/h
            'old_latent_cooling_load_room5': self.l_cl[4],  # MJ/h
            'old_latent_cooling_load_sum_of_occupant_rooms': np.sum(self.l_cl[0:5], axis=0),  # MJ/h
            'old_heating_load_sum_of_non_occupant_rooms': np.sum(self.l_h[5:12], axis=0),  # MJ/h
            'old_sensible_cooling_load_sum_of_non_occupant_rooms': np.sum(self.l_cs[5:12], axis=0),  # MJ/h
            'old_latent_cooling_load_sum_of_non_occupant_rooms': np.sum(self.l_cl[5:12], axis=0),  # MJ/h
            'old_heating_load_sum_of_12_rooms': np.sum(self.l_h, axis=0),  # MJ/h
            'old_sensible_cooling_load_sum_of_12_rooms': np.sum(self.l_cs, axis=0),  # MJ/h
            'old_latent_cooling_load_sum_of_12_rooms': np.sum(self.l_cl, axis=0),  # MJ/h
            'air_conditioned_temperature': self.theta_ac,  # degree C
            'air_conditioned_absolute_humidity': self.x_ac,  # kg/kgDA
            'non_occupant_room_temperature': self.theta_d_nac,  # degree C
            'non_occupant_room_absolute_humidity': self.x_d_nac,  # kg/kgDA
            'heat_loss_through_partition_heating_room1': self.q_d_trs_prt[0],  # MJ/h
            'heat_loss_through_partition_heating_room2': self.q_d_trs_prt[1],  # MJ/h
            'heat_loss_through_partition_heating_room3': self.q_d_trs_prt[2],  # MJ/h
            'heat_loss_through_partition_heating_room4': self.q_d_trs_prt[3],  # MJ/h
            'heat_loss_through_partition_heating_room5': self.q_d_trs_prt[4],  # MJ/h
            'heating_load_room1': self.l_d_h[0],  # MJ/h
            'heating_load_room2': self.l_d_h[1],  # MJ/h
            'heating_load_room3': self.l_d_h[2],  # MJ/h
            'heating_load_room4': self.l_d_h[3],  # MJ/h
            'heating_load_room5': self.l_d_h[4],  # MJ/h
            'sensible_cooling_load_room1': self.l_d_cs[0],  # MJ/h
            'sensible_cooling_load_room2': self.l_d_cs[1],  # MJ/h
            'sensible_cooling_load_room3': self.l_d_cs[2],  # MJ/h
            'sensible_cooling_load_room4': self.l_d_cs[3],  # MJ/h
            'sensible_cooling_load_room5': self.l_d_cs[4],  # MJ/h
            'latent_cooling_load_room1': self.l_d_cl[0],  # MJ/h
            'latent_cooling_load_room2': self.l_d_cl[1],  # MJ/h
            'latent_cooling_load_room3': self.l_d_cl[2],  # MJ/h
            'latent_cooling_load_room4': self.l_d_cl[3],  # MJ/h
            'latent_cooling_load_room5': self.l_d_cl[4],  # MJ/h
            'attic_temperature': self.theta_attic,  # degree C
            'duct_ambient_temperature_room1': self.theta_sur[0],  # degree C
            'duct_ambient_temperature_room2': self.theta_sur[1],  # degree C
            'duct_ambient_temperature_room3': self.theta_sur[2],  # degree C
            'duct_ambient_temperature_room4': self.theta_sur[3],  # degree C
            'duct_ambient_temperature_room5': self.theta_sur[4],  # degree C
            'inlet air temperature of heat source': self.theta_d_hs_in,  # degree C
            'inlet air absolute humidity of heat source': self.x_d_hs_in,  # kg/kgDA
            'maximum_output_heating': self.q_hs_max_h,  # MJ/h
            'maximum_output_sensible_cooling': self.q_hs_max_cs,  # MJ/h
            'maximum_output_latent_cooling': self.q_hs_max_cl,  # MJ/h
            'maximum temperature when maximum output of heat source': self.theta_hs_out_max_h,  # degree C
            'minimum temperature when maximum output of heat source': self.theta_hs_out_min_c,  # degree C
            'minimum absolute humidity when maximum output of heat source': self.x_hs_out_min_c,  # kg/kgDA
            'requested_supply_air_temperature_heating_room1': self.theta_req_h[0],  # degree C
            'requested_supply_air_temperature_heating_room2': self.theta_req_h[1],  # degree C
            'requested_supply_air_temperature_heating_room3': self.theta_req_h[2],  # degree C
            'requested_supply_air_temperature_heating_room4': self.theta_req_h[3],  # degree C
            'requested_supply_air_temperature_heating_room5': self.theta_req_h[4],  # degree C
            'requested_supply_air_temperature_cooling_room1': self.theta_req_c[0],  # degree C
            'requested_supply_air_temperature_cooling_room2': self.theta_req_c[1],  # degree C
            'requested_supply_air_temperature_cooling_room3': self.theta_req_c[2],  # degree C
            'requested_supply_air_temperature_cooling_room4': self.theta_req_c[3],  # degree C
            'requested_supply_air_temperature_cooling_room5': self.theta_req_c[4],  # degree C
            'requested_supply_air_absolute_humidity_cooling_room1': self.x_req_c[0],  # kg/kgDA
            'requested_supply_air_absolute_humidity_cooling_room2': self.x_req_c[1],  # kg/kgDA
            'requested_supply_air_absolute_humidity_cooling_room3': self.x_req_c[2],  # kg/kgDA
            'requested_supply_air_absolute_humidity_cooling_room4': self.x_req_c[3],  # kg/kgDA
            'requested_supply_air_absolute_humidity_cooling_room5': self.x_req_c[4],  # kg/kgDA
            'outlet_temperature_of_heat_source_heating': self.theta_hs_out_h,  # degree C
            'outlet_temperature_of_heat_source_cooling': self.theta_hs_out_c,  # degree C
            'supply_air_volume_room1': self.v_supply[0],  # degree C
            'supply_air_volume_room2': self.v_supply[1],  # degree C
            'supply_air_volume_room3': self.v_supply[2],  # degree C
            'supply_air_volume_room4': self.v_supply[3],  # degree C
            'supply_air_volume_room5': self.v_supply[4],  # degree C
            'outlet_absolute_humidity_of_heat_source_cooling': self.x_hs_out_c,  # kg/kgDA
            'duct_heat_loss_heating_room1': self.q_loss_duct_h[0],  # MJ/h
            'duct_heat_loss_heating_room2': self.q_loss_duct_h[1],  # MJ/h
            'duct_heat_loss_heating_room3': self.q_loss_duct_h[2],  # MJ/h
            'duct_heat_loss_heating_room4': self.q_loss_duct_h[3],  # MJ/h
            'duct_heat_loss_heating_room5': self.q_loss_duct_h[4],  # MJ/h
            'duct_heat_gain_cooling_room1': self.q_gain_duct_c[0],  # MJ/h
            'duct_heat_gain_cooling_room2': self.q_gain_duct_c[1],  # MJ/h
            'duct_heat_gain_cooling_room3': self.q_gain_duct_c[2],  # MJ/h
            'duct_heat_gain_cooling_room4': self.q_gain_duct_c[3],  # MJ/h
            'duct_heat_gain_cooling_room5': self.q_gain_duct_c[4],  # MJ/h
            'supply_air_temperature_heating_room1': self.theta_supply_h[0],  # degree C
            'supply_air_temperature_heating_room2': self.theta_supply_h[1],  # degree C
            'supply_air_temperature_heating_room3': self.theta_supply_h[2],  # degree C
            'supply_air_temperature_heating_room4': self.theta_supply_h[3],  # degree C
            'supply_air_temperature_heating_room5': self.theta_supply_h[4],  # degree C
            'supply_air_temperature_cooling_room1': self.theta_supply_c[0],  # degree C
            'supply_air_temperature_cooling_room2': self.theta_supply_c[1],  # degree C
            'supply_air_temperature_cooling_room3': self.theta_supply_c[2],  # degree C
            'supply_air_temperature_cooling_room4': self.theta_supply_c[3],  # degree C
            'supply_air_temperature_cooling_room5': self.theta_supply_c[4],  # degree C
            'supply_air_absolute_humidity_cooling_room1': self.x_supply_c[0],  # kg/kgDA
            'supply_air_absolute_humidity_cooling_room2': self.x_supply_c[1],  # kg/kgDA
            'supply_air_absolute_humidity_cooling_room3': self.x_supply_c[2],  # kg/kgDA
            'supply_air_absolute_humidity_cooling_room4': self.x_supply_c[3],  # kg/kgDA
            'supply_air_absolute_humidity_cooling_room5': self.x_supply_c[4],  # kg/kgDA
            'actual_air_conditioned_temperature_room1': self.theta_ac_act[0],  # degree C
            'actual_air_conditioned_temperature_room2': self.theta_ac_act[1],  # degree C
            'actual_air_conditioned_temperature_room3': self.theta_ac_act[2],  # degree C
            'actual_air_conditioned_temperature_room4': self.theta_ac_act[3],  # degree C
            'actual_air_conditioned_temperature_room5': self.theta_ac_act[4],  # degree C
            'actual_air_conditioned_absolute_humidity_room1': self.x_ac_act[0],  # kg/kgDA
            'actual_air_conditioned_absolute_humidity_room2': self.x_ac_act[1],  # kg/kgDA
            'actual_air_conditioned_absolute_humidity_room3': self.x_ac_act[2],  # kg/kgDA
            'actual_air_conditioned_absolute_humidity_room4': self.x_ac_act[3],  # kg/kgDA
            'actual_air_conditioned_absolute_humidity_room5': self.x_ac_act[4],  # kg/kgDA
            'actual_treated_heating_load_room1': self.l_d_act_h[0],  # MJ/h
            'actual_treated_heating_load_room2': self.l_d_act_h[1],  # MJ/h
            'actual_treated_heating_load_room3': self.l_d_act_h[2],  # MJ/h
            'actual_treated_heating_load_room4': self.l_d_act_h[3],  # MJ/h
            'actual_treated_heating_load_room5': self.l_d_act_h[4],  # MJ/h
            'actual_treated_heating_load_all': np.sum(self.l_d_act_h, axis=0),  # MJ/h
            'actual_treated_sensible_cooling_load_room1': self.l_d_act_cs[0],  # MJ/h
            'actual_treated_sensible_cooling_load_room2': self.l_d_act_cs[1],  # MJ/h
            'actual_treated_sensible_cooling_load_room3': self.l_d_act_cs[2],  # MJ/h
            'actual_treated_sensible_cooling_load_room4': self.l_d_act_cs[3],  # MJ/h
            'actual_treated_sensible_cooling_load_room5': self.l_d_act_cs[4],  # MJ/h
            'actual_treated_sensible_cooling_load_all': np.sum(self.l_d_act_cs, axis=0),  # MJ/h
            'actual_treated_latent_cooling_load_room1': self.l_d_act_cl[0],  # MJ/h
            'actual_treated_latent_cooling_load_room2': self.l_d_act_cl[1],  # MJ/h
            'actual_treated_latent_cooling_load_room3': self.l_d_act_cl[2],  # MJ/h
            'actual_treated_latent_cooling_load_room4': self.l_d_act_cl[3],  # MJ/h
            'actual_treated_latent_cooling_load_room5': self.l_d_act_cl[4],  # MJ/h
            'actual_treated_latent_cooling_load_all': np.sum(self.l_d_act_cl, axis=0),  # MJ/h
            'untreated_heating_load_room1': self.q_ut_h[0],  # MJ/h
            'untreated_heating_load_room2': self.q_ut_h[1],  # MJ/h
            'untreated_heating_load_room3': self.q_ut_h[2],  # MJ/h
            'untreated_heating_load_room4': self.q_ut_h[3],  # MJ/h
            'untreated_heating_load_room5': self.q_ut_h[4],  # MJ/h
            'untreated_heating_load_all': np.sum(self.q_ut_h, axis=0),  # MJ/h
            'untreated_sensible_cooling_load_room1': self.q_ut_cs[0],  # MJ/h
            'untreated_sensible_cooling_load_room2': self.q_ut_cs[1],  # MJ/h
            'untreated_sensible_cooling_load_room3': self.q_ut_cs[2],  # MJ/h
            'untreated_sensible_cooling_load_room4': self.q_ut_cs[3],  # MJ/h
            'untreated_sensible_cooling_load_room5': self.q_ut_cs[4],  # MJ/h
            'untreated_sensible_cooling_load_all': np.sum(self.q_ut_cs, axis=0),  # MJ/h
            'untreated_latent_cooling_load_room1': self.q_ut_cl[0],  # MJ/h
            'untreated_latent_cooling_load_room2': self.q_ut_cl[1],  # MJ/h
            'untreated_latent_cooling_load_room3': self.q_ut_cl[2],  # MJ/h
            'untreated_latent_cooling_load_room4': self.q_ut_cl[3],  # MJ/h
            'untreated_latent_cooling_load_room5': self.q_ut_cl[4],  # MJ/h
            'untreated_latent_cooling_load_all': np.sum(self.q_ut_cl, axis=0),  # MJ/h
            'actual_non_occupant_room_temperature': self.theta_nac,  # degree C
            'actual_non_occupant_room_absolute_humidity': self.x_nac,  # kg/kgDA
            'actual_non_occupant_room_heating_load': self.l_d_act_nac_h,  # MJ/h
            'actual_non_occupant_room_sensible_cooling_load': self.l_d_act_nac_cs,  # MJ/h
            'actual_non_occupant_room_latent_cooling_load': self.l_d_act_nac_cl,  # MJ/h
            'actual_heat_loss_through_partitions_heating_room1': self.q_trs_prt_h[0],  # MJ/h
            'actual_heat_loss_through_partitions_heating_room2': self.q_trs_prt_h[1],  # MJ/h
            'actual_heat_loss_through_partitions_heating_room3': self.q_trs_prt_h[2],  # MJ/h
            'actual_heat_loss_through_partitions_heating_room4': self.q_trs_prt_h[3],  # MJ/h
            'actual_heat_loss_through_partitions_heating_room5': self.q_trs_prt_h[4],  # MJ/h
            'actual_heat_loss_through_partitions_heating_all': np.sum(self.q_trs_prt_h, axis=0),  # MJ/h
            'actual_heat_gain_through_partitions_cooling_room1': self.q_trs_prt_c[0],  # MJ/h
            'actual_heat_gain_through_partitions_cooling_room2': self.q_trs_prt_c[1],  # MJ/h
            'actual_heat_gain_through_partitions_cooling_room3': self.q_trs_prt_c[2],  # MJ/h
            'actual_heat_gain_through_partitions_cooling_room4': self.q_trs_prt_c[3],  # MJ/h
            'actual_heat_gain_through_partitions_cooling_room5': self.q_trs_prt_c[4],  # MJ/h
            'actual_heat_gain_through_partitions_cooling_all': np.sum(self.q_trs_prt_c, axis=0),  # MJ/h
            'actual_inlet_air_temperature_of_heat_source': self.theta_hs_in,  # degree C
            'actual_inlet_absolute_humidity_of_heat_source': self.x_hs_in,  # kg/kgDA
            'output_of_heat_source_heating': self.q_hs_h,  # MJ/h
            'output_of_heat_source_cooling': self.q_hs_c,  # MJ/h
            'output_of_heat_source_sensible_cooling': self.q_hs_cs,  # MJ/h
            'output_of_heat_source_latent_cooling': self.q_hs_cl,  # MJ/h
            'heat_source_heating_capacity': self.q_hs_cap_h,
            'heat_source_cooling_capacity': self.q_hs_cap_c,
            'heat_source_sensible_cooling_capacity': self.q_hs_cap_cs,
            'heat_source_latent_cooling_capacity': self.q_hs_cap_cl,
            'theoretical_compressor_efficiency_heating': self.e_comp_th_h,
            'evaporator_temperature_heating': self.theta_ref_evp_h,
            'condenser_temperature_heating': self.theta_ref_cnd_h,
            'theoretical_compressor_efficiency_cooling': self.e_comp_th_c,
            'evaporator_temperature_cooling': self.theta_ref_evp_c,
            'condenser_temperature_cooling': self.theta_ref_cnd_c,
            'compression efficiency heating': self.eta_comp_h,
            'compression efficiency cooling': self.eta_comp_c,
            'compressor efficiency heating': self.e_comp_h,
            'compressor efficiency cooling': self.e_comp_c,
            'compressor power heating': self.e_e_comp_h,
            'compressor power cooling': self.e_e_comp_c,
            'fan power heating': self.e_e_fan_h,
            'fan power cooling': self.e_e_fan_c,
            'power heating': self.e_e_h,
            'power cooling': self.e_e_c,
            'primary energy for heating treated': self.e_h_t,
            'primary energy for heating untreated': self.e_h_ut,
            'primary energy for cooling treated': self.e_c_t,
            'primary energy for cooling untreated': self.e_c_ut
        }


def get_main_value(
        region: int,
        a_mr: float, a_or: float, a_a: float, r_env: float,
        insulation: str, solar_gain: str,
        system_spec: dict) -> OutputData:
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
        system_spec: system spec
    """

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

    # region system spec

    input_method, is_duct_insulated, vav_system, ventilation_included, \
        q_rtd_h, q_rtd_c, v_hs_rtd_h, v_hs_rtd_c, p_rtd_h, p_rtd_c, p_fan_rtd_h, p_fan_rtd_c, \
        q_mid_h, q_mid_c, v_hs_mid_h, v_hs_mid_c, p_mid_h, p_mid_c, p_fan_mid_h, p_fan_mid_c \
        = get_system_spec(region, a_a, system_spec)

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

    # outdoor relative humidity, % (8760 times)
    h_ex = get_relative_humidity(theta_ex, x_ex)

    # horizontal solar radiation, W/m2K (8760 times)
    j = get_horizontal_solar(region)

    # SAT temperature, degree C, (8760 times)
    theta_sat = get_sat_temperature(region)

    # endregion

    # region occupant usage

    # heating and cooling schedule (8760 times)
    hc_period = get_heating_and_cooling_schedule(region)

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
        a_a, q, mu_h, v_vent, theta_ex, j, hc_period, n_p, q_gen, v_local, theta_set_h)
    q_d_hs_c = get_cooling_output_for_supply_air_estimation(
        a_a, q, mu_c, v_vent, theta_ex, x_ex, j, hc_period, n_p, q_gen, w_gen, v_local, theta_set_c, x_set_c)

    # minimum supply air volume of the system for heating and cooling, (m3/h, m3/h)
    v_hs_min = get_minimum_air_volume(v_vent)

    # rated heating and cooling output of the heat source, (MJ/h, MJ/h)
    q_hs_rtd_h, q_hs_rtd_c = get_rated_output(q_rtd_h, q_rtd_c)

    # supply air volume of heat source, m3/h
    v_d_hs_supply = get_heat_source_supply_air_volume(
        hc_period, ventilation_included, q_d_hs_h, q_d_hs_c, q_hs_rtd_h, q_hs_rtd_c, v_hs_min, v_hs_rtd_h, v_hs_rtd_c)

    # the ratio of the supply air volume valance for each 5 rooms
    r_supply_des = get_supply_air_volume_valance(a_hcz)

    # supply air volume without vav adjustment, m3/h (5 rooms * 8760 times)
    v_d_supply = get_each_supply_air_volume_not_vav_adjust(r_supply_des, v_d_hs_supply, v_vent)

    # endregion

    # region load

    # heating load, and sensible and latent cooling load, MJ/h ((8760times), (8760 times), (8760 times))
    l_h, l_cs, l_cl = get_load(region, insulation, solar_gain, a_mr, a_or, a_a, r_env)

    # heating and cooling room temperature, degree C (8760 times)
    theta_ac = get_air_conditioned_room_temperature(hc_period, theta_ex, theta_set_h, theta_set_c)

    # room absolute humidity, kg/kgDA (8760 times)
    x_ac = get_air_conditioned_room_absolute_humidity(hc_period, x_ex, x_set_c)

    # non occupant room temperature balanced, degree C, (8760 times)
    theta_d_nac = get_non_occupant_room_temperature_balanced(
        hc_period, l_h, l_cs, q, a_nr, v_local_nr, v_d_supply, u_prt, a_prt, theta_ac)

    # non occupant room absolute humidity, kg/kgDA (8760 times)
    x_d_nac = get_non_occupant_room_absolute_humidity_balanced(
        hc_period, l_cl, v_local_nr, v_d_supply, x_ac)

    # heat transfer through partition from occupant room to non occupant room balanced, MJ/h, (5 rooms * 8760 times)
    q_d_trs_prt = get_heat_transfer_through_partition_balanced(u_prt, a_prt, theta_ac, theta_d_nac)

    # heating and sensible cooling load in the occupant rooms, MJ/h, (5 rooms * 8760 times)
    l_d_h = get_occupant_room_load_for_heating_balanced(l_h, q_d_trs_prt)
    l_d_cs, l_d_cl = get_occupant_room_load_for_cooling_balanced(l_cs, l_cl, q_d_trs_prt)

    # endregion

    # region treated and untreated load

    # operation (8760 times) h = heating operation, c = cooling operation, n = non operation
    operation = get_operation(l_d_h, l_d_cs)

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

    # maximum and minimum temperature and absolute humidity when maximum output of heat source
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
        hc_period, vav_system, l_d_h, l_d_cs, theta_hs_out_h, theta_hs_out_c, theta_sur,
        psi, l_duct, theta_ac, v_vent, v_d_supply, operation)

    # outlet absolute humidity of heat source, kg/kgDA (8760 times)
    x_hs_out_c = get_decided_outlet_supply_air_absolute_humidity_for_cooling(x_req_c, v_supply, x_hs_out_min_c)

    # heat loss from ducts, heat gain to ducts, MJ/h, (5 rooms * 8760 times), reference
    q_loss_duct_h, q_gain_duct_c = get_duct_heat_loss_and_gain(
        theta_sur, theta_hs_out_h, theta_hs_out_c, v_supply, psi, l_duct, operation)

    # supply air temperature, degree C (5 rooms * 8760 times)
    theta_supply_h = get_supply_air_temperature_for_heating(
        theta_sur, theta_hs_out_h, psi, l_duct, v_supply, theta_ac, operation)
    # supply air temperature, degree C (5 rooms * 8760 times)
    theta_supply_c = get_supply_air_temperature_for_cooling(
        theta_sur, theta_hs_out_c, psi, l_duct, v_supply, theta_ac, operation)
    # supply air absolute humidity, kg/kgDA (5 rooms * 8760 times)
    x_supply_c = get_supply_air_absolute_humidity_for_cooling(x_hs_out_c, x_ac, operation)

    # actual air conditioned temperature, degree C, (5 rooms * 8760 times)
    theta_ac_act = get_actual_air_conditioned_temperature(
        hc_period, theta_ac, v_supply, theta_supply_h, theta_supply_c,
        l_d_h, l_d_cs, u_prt, a_prt, a_hcz, q)

    # actual air conditioned absolute humidity, kg/kgDA (5 rooms * 8760 times)
    x_ac_act = get_actual_air_conditioned_absolute_humidity(x_ac)

    # actual treated load for heating, MJ/h, (5 rooms * 8760 times)
    l_d_act_h = get_actual_treated_heating_load(hc_period, theta_supply_h, theta_ac_act, v_supply)
    l_d_act_cs = get_actual_treated_sensible_cooling_load(hc_period, theta_supply_c, theta_ac_act, v_supply)
    l_d_act_cl = get_actual_treated_latent_cooling_load(hc_period, x_supply_c, x_ac_act, v_supply)

    # untreated load, MJ/h, (5 rooms * 8760 times, 5 rooms * 8760 times, 5 rooms * 8760 times)
    q_ut_h, q_ut_cs, q_ut_cl = get_untreated_load(
        l_d_act_h, l_d_h, l_d_act_cs, l_d_cs, l_d_act_cl, l_d_cl)

    # actual non occupant room temperature, degree C, (8760 times)
    theta_nac = get_actual_non_occupant_room_temperature(
        theta_d_nac, theta_ac, theta_ac_act, v_supply, v_d_supply, v_local_nr, u_prt, a_prt, q, a_nr)
    # actual non occupant room absolute humidity, kg/kgDA, (8760 times)
    x_nac = get_actual_non_occupant_room_absolute_humidity(x_d_nac)

    # actual non occupant room load, MJ/h, (8760 times)
    l_d_act_nac_h = get_actual_non_occupant_room_heating_load(theta_ac_act, theta_nac, v_supply, operation)
    l_d_act_nac_cs = get_actual_non_occupant_room_sensible_cooling_load(theta_ac_act, theta_nac, v_supply, operation)
    l_d_act_nac_cl = get_actual_non_occupant_room_latent_cooling_load(x_ac_act, x_nac, v_supply, operation)

    # actual heat loss or gain through partitions, MJ/h, (5 rooms * 8760 times)
    q_trs_prt_h = get_actual_heat_loss_through_partition_for_heating(u_prt, a_prt, theta_ac_act, theta_nac, operation)
    q_trs_prt_c = get_actual_heat_gain_through_partition_for_cooling(u_prt, a_prt, theta_ac_act, theta_nac, operation)

    # inlet air temperature of heat source,degree C, (8760 times)
    theta_hs_in = get_heat_source_inlet_air_temperature(theta_nac)
    # inlet air absolute humidity of heat source, kg/kgDA (8760 times)
    x_hs_in = get_heat_source_inlet_air_absolute_humidity(x_nac)

    # output of heat source, MJ/h, (8760 times)
    q_hs_h = get_heat_source_heating_output(theta_hs_out_h, theta_hs_in, v_supply, operation)
    q_hs_c, q_hs_cs, q_hs_cl = get_heat_source_cooling_output(
        theta_hs_in, x_hs_in, theta_hs_out_c, x_hs_out_c, v_supply, operation)

    # endregion

    # region energy

    # heating test
    #  rated
    eta_comp_rtd_h, e_comp_rtd_h, e_comp_th_rtd_h, _, _ = get_comp_eta_rtd_h(
        q_rtd_h, v_hs_rtd_h, p_rtd_h, p_fan_rtd_h)
    # middle
    eta_comp_mid_h, e_comp_mid_h, e_comp_th_mid_h, _, _ = get_comp_eta_mid_h(
        input_method, eta_comp_rtd_h, q_mid_h, v_hs_mid_h, p_mid_h, p_fan_mid_h)
    q_mid_h = get_q_hs_mid_h(input_method, q_rtd_h, q_mid_h)
    #  minimum
    eta_comp_min_h = get_comp_eta_min_h(eta_comp_rtd_h)
    q_min_h = get_q_hs_min_h(q_rtd_h)

    # cooling test
    #  rated
    eta_comp_rtd_c, e_comp_rtd_c, e_comp_th_rtd_c, _, _ = get_comp_eta_rtd_c(
        q_rtd_c, v_hs_rtd_c, p_rtd_c, p_fan_rtd_c)
    #  middle
    eta_comp_mid_c, e_comp_mid_c, e_comp_th_mid_c, _, _ = get_comp_eta_mid_c(
        input_method, eta_comp_rtd_c, q_mid_c, v_hs_mid_c, p_mid_c, p_fan_mid_c)
    q_mid_c = get_q_hs_mid_c(input_method, q_rtd_c, q_mid_c)
    #  minimum
    eta_comp_min_c = get_comp_eta_min_c(eta_comp_rtd_c)
    q_min_c = get_q_hs_min_c(q_rtd_c)

    # capacity
    q_hs_cap_h = get_heat_source_heating_capacity(theta_ex, h_ex, operation, theta_hs_out_h, theta_hs_in, v_supply)
    q_hs_cap_c, q_hs_cap_cs, q_hs_cap_cl = get_heat_source_cooling_capacity(
        theta_hs_in, x_hs_in, theta_hs_out_c, x_hs_out_c, v_supply, operation)

    # theoretical efficiency
    e_comp_th_h, theta_ref_evp_h, theta_ref_cnd_h = get_e_th_h(
        operation, theta_hs_in, theta_hs_out_h, np.sum(v_supply, axis=0), theta_ex)
    e_comp_th_c, theta_ref_evp_c, theta_ref_cnd_c = get_e_th_c(
        operation, theta_hs_in, x_hs_in, theta_hs_out_c, np.sum(v_supply, axis=0), theta_ex)

    # compression efficiency
    eta_comp_h = get_eta_comp_h(
        q_hs_cap_h, eta_comp_min_h, eta_comp_mid_h, eta_comp_rtd_h, q_min_h, q_mid_h, q_rtd_h)
    eta_comp_c = get_eta_comp_c(
        q_hs_cap_c, eta_comp_min_c, eta_comp_mid_c, eta_comp_rtd_c, q_min_c, q_mid_c, q_rtd_c)

    # compressor efficiency
    e_comp_h = get_e_comp_h(e_comp_th_h, eta_comp_h)
    e_comp_c = get_e_comp_c(e_comp_th_c, eta_comp_c)

    # compressor power, kWh/h
    e_e_comp_h = get_e_e_comp_h(q_hs_cap_h, e_comp_h)
    e_e_comp_c = get_e_e_comp_c(q_hs_cap_c, e_comp_c)

    # fan power, kWh/h
    e_e_fan_h = get_e_e_fan_h(operation, is_duct_insulated, p_fan_rtd_h, v_vent, v_supply, v_hs_rtd_h)
    e_e_fan_c = get_e_e_fan_c(operation, is_duct_insulated, p_fan_rtd_c, v_vent, v_supply, v_hs_rtd_c)

    # power, kWh/h
    e_e_h = get_e_e_h(e_e_comp_h, e_e_fan_h)
    e_e_c = get_e_e_c(e_e_comp_c, e_e_fan_c)

    # primary energy, MJ/h
    e_h_t, e_h_ut, e_c_t, e_c_ut = get_e(region, e_e_h, e_e_c, q_ut_h, q_ut_cs, q_ut_cl)

    # end region

    # region make output data class

    od = OutputData()

    od.l_duct = l_duct
    od.r_supply_des = r_supply_des
    od.v_vent = v_vent
    od.v_hs_min = v_hs_min
    od.a_prt = a_prt
    od.q_hs_rtd_h = q_hs_rtd_h
    od.q_hs_rtd_c = q_hs_rtd_c

    od.theta_ex = theta_ex
    od.x_ex = x_ex
    od.h_ex = h_ex
    od.j = j
    od.theta_sat = theta_sat
    od.q_d_hs_h = q_d_hs_h
    od.q_d_hs_c = q_d_hs_c
    od.v_d_hs_supply = v_d_hs_supply
    od.v_d_supply = v_d_supply
    od.l_h = l_h
    od.l_cs = l_cs
    od.l_cl = l_cl
    od.theta_ac = theta_ac
    od.x_ac = x_ac
    od.theta_d_nac = theta_d_nac
    od.x_d_nac = x_d_nac
    od.q_d_trs_prt = q_d_trs_prt
    od.l_d_h = l_d_h
    od.l_d_cs = l_d_cs
    od.l_d_cl = l_d_cl
    od.theta_attic = theta_attic
    od.theta_sur = theta_sur
    od.theta_d_hs_in = theta_d_hs_in
    od.x_d_hs_in = x_d_hs_in
    od.q_hs_max_h = q_hs_max_h
    od.q_hs_max_cs = q_hs_max_cs
    od.q_hs_max_cl = q_hs_max_cl
    od.theta_hs_out_max_h = theta_hs_out_max_h
    od.theta_hs_out_min_c = theta_hs_out_min_c
    od.x_hs_out_min_c = x_hs_out_min_c
    od.theta_req_h = theta_req_h
    od.theta_req_c = theta_req_c
    od.x_req_c = x_req_c
    od.theta_hs_out_h = theta_hs_out_h
    od.theta_hs_out_c = theta_hs_out_c
    od.v_supply = v_supply
    od.x_hs_out_c = x_hs_out_c
    od.q_loss_duct_h = q_loss_duct_h
    od.q_gain_duct_c = q_gain_duct_c
    od.theta_supply_h = theta_supply_h
    od.theta_supply_c = theta_supply_c
    od.x_supply_c = x_supply_c
    od.theta_ac_act = theta_ac_act
    od.x_ac_act = x_ac_act
    od.l_d_act_h = l_d_act_h
    od.l_d_act_cs = l_d_act_cs
    od.l_d_act_cl = l_d_act_cl
    od.q_ut_h = q_ut_h
    od.q_ut_cs = q_ut_cs
    od.q_ut_cl = q_ut_cl
    od.theta_nac = theta_nac
    od.x_nac = x_nac
    od.l_d_act_nac_h = l_d_act_nac_h
    od.l_d_act_nac_cs = l_d_act_nac_cs
    od.l_d_act_nac_cl = l_d_act_nac_cl
    od.q_trs_prt_h = q_trs_prt_h
    od.q_trs_prt_c = q_trs_prt_c
    od.theta_hs_in = theta_hs_in
    od.x_hs_in = x_hs_in
    od.q_hs_h = q_hs_h
    od.q_hs_c = q_hs_c
    od.q_hs_cs = q_hs_cs
    od.q_hs_cl = q_hs_cl

    od.eta_comp_rtd_h = eta_comp_rtd_h
    od.e_comp_rtd_h = e_comp_rtd_h
    od.e_comp_th_rtd_h = e_comp_th_rtd_h
    od.eta_comp_mid_h = eta_comp_mid_h
    od.e_comp_mid_h = e_comp_mid_h
    od.e_comp_th_mid_h = e_comp_th_mid_h
    od.q_mid_h = q_mid_h
    od.eta_comp_min_h = eta_comp_min_h
    od.q_min_h = q_min_h
    od.eta_comp_rtd_c = eta_comp_rtd_c
    od.e_comp_rtd_c = e_comp_rtd_c
    od.e_comp_th_rtd_c = e_comp_th_rtd_c
    od.eta_comp_mid_c = eta_comp_mid_c
    od.e_comp_mid_c = e_comp_mid_c
    od.e_comp_th_mid_c = e_comp_th_mid_c
    od.q_mid_c = q_mid_c
    od.eta_comp_min_c = eta_comp_min_c
    od.q_min_c = q_min_c

    od.q_hs_cap_h = q_hs_cap_h  # capacity
    od.q_hs_cap_c = q_hs_cap_c
    od.q_hs_cap_cs = q_hs_cap_cs
    od.q_hs_cap_cl = q_hs_cap_cl  # capacity
    od.e_comp_th_h = e_comp_th_h
    od.theta_ref_evp_h = theta_ref_evp_h
    od.theta_ref_cnd_h = theta_ref_cnd_h  # theoretical efficiency
    od.e_comp_th_c = e_comp_th_c
    od.theta_ref_evp_c = theta_ref_evp_c
    od.theta_ref_cnd_c = theta_ref_cnd_c  # theoretical efficiency
    od.eta_comp_h = eta_comp_h  # compression efficiency
    od.eta_comp_c = eta_comp_c  # compression efficiency
    od.e_comp_h = e_comp_h  # compressor efficiency
    od.e_comp_c = e_comp_c  # compressor efficiency
    od.e_e_comp_h = e_e_comp_h  # compressor power, kWh/h
    od.e_e_comp_c = e_e_comp_c  # compressor power, kWh/h
    od.e_e_fan_h = e_e_fan_h  # fan power, kWh/h
    od.e_e_fan_c = e_e_fan_c  # fan power, kWh/h
    od.e_e_h = e_e_h  # power, kWh/h
    od.e_e_c = e_e_c  # power, kWh/h
    od.e_h_t = e_h_t
    od.e_h_ut = e_h_ut
    od.e_c_t = e_c_t
    od.e_c_ut = e_c_ut  # primary energy, MJ/h

    # endregion

    return od
