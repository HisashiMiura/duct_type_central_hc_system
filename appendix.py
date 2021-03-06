from typing import List
import numpy as np

import envelope
import read_conditions


# 使用していないと思われる
class SystemSpec:
    def __init__(self, cap_rtd_h: float, cap_rtd_c: float,
                 supply_air_rtd_h: float, supply_air_rtd_c: float,
                 is_duct_insulated: bool = False,
                 vav_system: bool = False):
        """
        Args:
            cap_rtd_h: rated heating capacity, W
            cap_rtd_c: rated cooling capacity, W
            supply_air_rtd_h: rated supply air volume for heating, m3/h
            supply_air_rtd_c: rated supply air volume for cooling, m3/h
            is_duct_insulated: is the duct insulated or not
            vav_system: is vav system equipped or not
        """
        self.cap_rtd_h = cap_rtd_h
        self.cap_rtd_c = cap_rtd_c
        self.supply_air_rtd_h = supply_air_rtd_h
        self.supply_air_rtd_c = supply_air_rtd_c
        self.is_duct_insulated = is_duct_insulated
        self.vav_system = vav_system


def get_default_rated_capacity(region: int, a_a: float) -> (float, float):
    """
    get the rated capacity
    Args:
        region: region 1-8
        a_a: total floor area, m2
    Returns:
        rated heating capacity, rated cooling capacity, W
    """

    # corrected value for effects of the external temperature to the capacity
    f_ct = 1.05

    # corrected value for effects of the intermittent operation to the capacity
    f_cl = 1.0

    if region == 8:
        q_rtd_h = None
    else:
        q_rq_h = {
            1: 73.91,
            2: 64.32,
            3: 62.65,
            4: 66.99,
            5: 72.64,
            6: 61.34,
            7: 64.55,
        }[region]
        q_rtd_h = q_rq_h * a_a * f_ct * f_cl

    q_rq_c = {
        1: 37.61,
        2: 36.55,
        3: 42.34,
        4: 54.08,
        5: 61.69,
        6: 60.79,
        7: 72.53,
        8: 61.56,
    }[region]

    q_rtd_c = q_rq_c * a_a * f_ct * f_cl

    return q_rtd_h, q_rtd_c


def get_default_rated_power(q_rtd_h: float, q_rtd_c: float) -> (float, float):
    """
    Args:
        q_rtd_h: rated capacity for heating, W
        q_rtd_c: rated capacity for cooling, W
    Returns:
        rated power for heating, W
        rated power for cooling, W
    """

    return q_rtd_h/3.76, q_rtd_c/3.17


def get_default_rated_supply_air_volume(q_rtd_h: float, q_rtd_c: float) -> (float, float):
    """
    Args:
        q_rtd_h: rated capacity for heating, W
        q_rtd_c: rated capacity for cooling, W
    Returns:
        rated supply air volume for heating, m3/h
        rated supply air volume for cooling, m3/h
    """

    v_rtd_h = (1.69 * q_rtd_h * 10**(-3) + 14.5) * 60
    v_rtd_c = (1.69 * q_rtd_c * 10**(-3) + 14.5) * 60

    return v_rtd_h, v_rtd_c


def get_default_rated_fan_power(v_rtd_h: float, v_rtd_c: float) -> (float, float):
    """
    Args:
        v_rtd_h: rated supply air volume for heating, m3/h
        v_rtd_c: rated supply air volume for cooling, m3/h
    Returns:
        rated fan power for heating, W
        rated fan power for cooling, W
    """

    p_fan_rtd_h = 8.0 * v_rtd_h / 60 + 20.7
    p_fan_rtd_c = 8.0 * v_rtd_c / 60 + 20.7

    return p_fan_rtd_h, p_fan_rtd_c


def get_maximum_heating_output(region: int, q_rtd_h: float) -> np.ndarray:
    """
    calculate the maximum heating output of heat source
    Args:
        region: region
        q_rtd_h: rated heating capacity, W
    Returns:
        maximum heating output, MJ/h (8760 times)
    """

    # outdoor temperature, degree C, (8760 times)
    theta_ex = read_conditions.read_temperature(region)

    # absolute humidity, kg/kgDA, (8760 times)
    x_ex = read_conditions.read_absolute_humidity(region)

    # relative humidity, %, (8760 times)
    h_ex = read_conditions.get_relative_humidity(theta_ex, x_ex)

    # coefficient for defrosting, (8760 times)
    c_df_h = np.where((theta_ex < 5.0) & (h_ex >= 80.0), 0.77, 1.0)

    alpha_max_h = 1.0

    return q_rtd_h * c_df_h * 3600 * 10**(-6) * alpha_max_h


def get_maximum_cooling_output(q_rtd_c: float, l_d_cs: np.ndarray, l_d_cl: np.ndarray) -> (np.ndarray, np.ndarray):
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

    # sensible cooling load including heat gain through the partition, MJ/h, (8760 times)
    l_d_cs_total = np.sum(l_d_cs, axis=0)

    # latent cooling load, MJ/h, (8760 times)
    l_d_cl_total = np.sum(l_d_cl, axis=0)

    # minimum SHF for cooling
    shf_l_min_c = 0.4

    # maximum latent cooling load, MJ/h (8760 times)
    l_max_cl = l_d_cs_total * (1 - shf_l_min_c) / shf_l_min_c

    # corrected latent cooling load, MJ/h, (8760 times)
    l_d2_cl = np.minimum(l_max_cl, l_d_cl_total)

    # corrected cooling load, MJ/h, (8760 times)
    l_d2_c = l_d_cs_total + l_d2_cl

    # corrected SHF for cooling load, (8760 times)
    shf_dash = np.vectorize(lambda x, y: x / y if y > 0.0 else 0.0)(l_d_cs_total, l_d2_c)

    alpha_max_c = 1.11

    # maximum cooling output, MJ/h, (8760 times)
    q_hs_max_c = np.full(8760, q_rtd_c * 3600 * 10**(-6) * alpha_max_c)

    # maximum sensible cooling output, MJ/h, (8760 times)
    q_hs_max_cs = q_hs_max_c * shf_dash

    # maximum latent cooling output, MJ/h, (8760 times)
    q_hs_max_cl = np.minimum(q_hs_max_c * (1 - shf_dash), l_d2_cl)

    return q_hs_max_cs, q_hs_max_cl



