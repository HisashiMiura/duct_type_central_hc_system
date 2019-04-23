from typing import List, Tuple
import numpy as np

import read_conditions
import envelope
import read_load
import appendix
from appendix import SystemSpec


# region functions

# region physical property

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


def get_duct_linear_heat_loss_coefficient() -> float:
    """
    get the liner heat loss coefficient (W/mK) of the duct
    Returns:
          liner heat loss coefficient, W/mK
    """
    return 0.49


def get_heat_loss_coefficient_of_partition() -> float:
    """
    return the heat loss coefficient of the partition
    Returns:
        heat loss coefficient of the partition, W/m2K
    """
    return 1 / 0.46


# endregion


# region duct length

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


def get_duct_length(l_duct_r_i: np.ndarray, a_a: float, a_a_r: float) -> np.ndarray:
    """
    calculate duct length for each room in the estimated house
    Args:
        l_duct_r_i: duct length for each room in the standard house, m, (5 rooms)
        a_a: total floor area of the estimated house, m2
        a_a_r: total floor area of the standard house, m2
    Returns:
        duct length for each room in estimated house, m, (5 rooms)
    """

    return l_duct_r_i * np.sqrt(a_a / a_a_r)


def calc_duct_length(a_a: float) -> np.ndarray:
    """
    calculate total duct length for each room in the estimated house
    Args:
        a_a: total floor area, m2
    Returns
        total duct length for each room in the estimated house, m (5 rooms)
    """

    # duct length for each room in the standard house, m ((5 rooms), (5 rooms), (5 rooms))
    internal, external, total = get_standard_house_duct_length()

    return get_duct_length(l_duct_r_i=total, a_a=a_a, a_a_r=120.08)

# endregion


# region air conditioned temperature

def get_air_conditioned_temperature_for_heating() -> float:
    """
    get air conditioned temperature for heating
    Returns:
        temperature, degree C
    """
    return 20.0


def get_air_conditioned_temperature_for_cooling() -> float:
    """
    get air conditioned temperature for cooling
    Returns:
        temperature, degree c
    """
    return 27.0

# endregion


# region attic temperature

def get_attic_temperature_for_heating(theta_sat: np.ndarray, theta_ac_h: np.ndarray, h: float) -> np.ndarray:
    """
    calculate attic temperature for heating
    Args:
        theta_sat: SAT temperature, degree C, (8760 times)
        theta_ac_h: air conditioned temperature for heating, degree C, (8760 times)
        h: temperature difference coefficient, -
    Returns:
        attic temperature for heating, degree C, (8760 times)
    """
    return theta_sat * h + theta_ac_h * (1 - h)


def get_attic_temperature_for_cooling(theta_sat: np.ndarray, theta_ac_c: np.ndarray, h: float) -> np.ndarray:
    """
    calculate attic temperature for cooling
    Args:
        theta_sat: SAT temperature, degree C, (8760 times)
        theta_ac_c: air conditioned temperature for cooling, degree C, (8760 times)
        h: temperature difference coefficient, -
    Returns:
        attic temperature for cooling, degree C, (8760 times)
    """
    return theta_sat * h + theta_ac_c * (1 - h)

# endregion


def get_duct_ambient_air_temperature_for_heating(
        is_duct_insulated: bool,
        l_duct_in_r: np.ndarray, l_duct_ex_r: np.ndarray,
        theta_ac_h: np.ndarray, theta_attic: np.ndarray) -> np.ndarray:
    """
    calculate duct ambient air temperature for heating
    Args:
        is_duct_insulated: is the duct insulated ?
        l_duct_in_r: duct length inside the insulated area in the standard house, m, (5 rooms)
        l_duct_ex_r: duct length outside the insulated area in the standard house, m, (5 rooms)
        theta_ac_h: air conditioned temperature for heating, degree C, (8760 times)
        theta_attic: attic temperature, degree C, (8760 times)
    Returns:
        duct ambient temperature for heating, degree C, (5 rooms * 8760 times)
    """

    if is_duct_insulated:
        # If the duct insulated, the duct ambient temperatures are equals to the air conditioned temperatures.
        return np.full((5, 8760), theta_ac_h)
    else:
        # If the duct NOT insulated, the duct ambient temperatures are
        # between the attic temperatures and the air conditioned temperatures.
        l_in = l_duct_in_r.reshape(1, 5).T
        l_ex = l_duct_ex_r.reshape(1, 5).T
        return (l_in * theta_ac_h + l_ex * theta_attic) / (l_in + l_ex)


def get_duct_ambient_air_temperature_for_cooling(
        is_duct_insulated: bool,
        l_duct_in_r: np.ndarray, l_duct_ex_r: np.ndarray,
        theta_ac_c: np.ndarray, theta_attic: np.ndarray) -> np.ndarray:
    """
    calculate duct ambient air temperature for cooling
    Args:
        is_duct_insulated: is the duct insulated ?
        l_duct_in_r: duct length inside the insulated area in the standard house, m, (5 rooms)
        l_duct_ex_r: duct length outside the insulated area in the standard house, m, (5 rooms)
        theta_ac_c: air conditioned temperature for cooling, degree C, (8760 times)
        theta_attic: attic temperature, degree C, (8760 times)
    Returns:
        duct ambient temperature for cooling, degree C, (5 rooms * 8760 times)
    """

    if is_duct_insulated:
        # If the duct insulated, the duct ambient temperatures are equals to the air conditioned temperatures.
        return np.full((5, 8760), theta_ac_c)
    else:
        # If the duct NOT insulated, the duct ambient temperatures are
        # between the attic temperatures and the air conditioned temperatures.
        l_in = l_duct_in_r.reshape(1, 5).T
        l_ex = l_duct_ex_r.reshape(1, 5).T
        return (l_in * theta_ac_c + l_ex * theta_attic) / (l_in + l_ex)


def calc_duct_ambient_air_temperature(total_floor_area: float, region: int, spec: SystemSpec) \
        -> (np.ndarray, np.ndarray):
    """get duct ambient air temperature
    Args:
        total_floor_area: total floor area, m2
        region: region 1-8
        spec: central heating and cooling system spec
    Returns:
        ambient air temperature for heating season, ambient air temperature for cooling season, degree C
            temperature is the list (5 rooms * 8760 times)
            8760 is each time in the year.
            5 is the ducts connecting to each 5 rooms heated or cooled.
    """

    # air conditioned temperature, degree C
    t_ac_h = get_air_conditioned_temperature_for_heating()
    t_ac_c = get_air_conditioned_temperature_for_cooling()

    def f(l_in: float, l_ex: float, t_ac, t_attic):
        return (l_in * np.full(8760, t_ac) + l_ex * t_attic) / (l_in + l_ex)

    if spec.is_duct_insulated:
        # If the duct insulated, the duct ambient temperatures are equals to the air conditioned temperatures.
        return np.full((5, 8760), t_ac_h), np.full((5, 8760), t_ac_c)
    else:
        # get the lengths of the ducts, m connected to the each 5 rooms
        internal_lengths, external_lengths, total_lengths = get_standard_house_duct_length()
        # SAT temperatures, degree C, (8760 times)
        sat_temperature = read_conditions.get_sat_temperature(region)
        # air conditioned temperature, degree C, (8760 times)
        theta_ac_h = np.full(8760, get_air_conditioned_temperature_for_heating())
        theta_ac_c = np.full(8760, get_air_conditioned_temperature_for_cooling())
        # attic temperatures(8760), degree C
        t_attic_h = get_attic_temperature_for_heating(sat_temperature, theta_ac_h, 1.0)
        t_attic_c = get_attic_temperature_for_cooling(sat_temperature, theta_ac_c, 1.0)

        # If the duct NOT insulated, the duct ambient temperatures are
        # between the attic temperatures and the air conditioned temperatures.
        heating = np.array([f(internal_length, external_length, t_ac_h, t_attic_h)
                            for (internal_length, external_length) in zip(internal_lengths, external_lengths)])
        cooling = np.array([f(internal_length, external_length, t_ac_c, t_attic_c)
                            for (internal_length, external_length) in zip(internal_lengths, external_lengths)])
        return heating, cooling


def get_supply_air_volume_valance(zone_floor_area: envelope.FloorArea) -> np.ndarray:
    """
    calculate supply air volume valance
    Args:
        zone_floor_area: fool area of each zones 'main occupant room', 'other occupant room', 'non occupant room'
    Returns:
        the ratio of the supply air volume valance for each 5 rooms (0.0-1.0)
    """

    # floor areas (12 rooms)
    floor_areas = envelope.get_hc_floor_areas(floor_area=zone_floor_area)

    # slice the list. 1: main occupant room, 2-5: other occupant rooms
    occupant_rooms_floor_area = floor_areas[0:5]

    # calculate the ratio
    return occupant_rooms_floor_area / np.sum(occupant_rooms_floor_area)


def get_mechanical_ventilation(zone_floor_area: envelope.FloorArea) -> np.ndarray:
    """calculate mechanical ventilation of each 5 rooms
    Args:
        zone_floor_area: floor area of each zones 'main occupant room', 'other occupant room', 'non occupant room'
    Returns:
        supply air volume of mechanical ventilation, m3/h, (5 rooms)
    """

    # referenced mechanical ventilation volume, m3/h
    v_vent_r = np.array([60.0, 20.0, 40.0, 20.0, 20.0])

    # referenced floor area of the occupant room(sliced 0 to 5)
    a_hcz_r = envelope.get_referenced_floor_area()[0:5]

    # floor area of the occupant room(sliced 0 to 5)
    a_hcz = envelope.get_hc_floor_areas(zone_floor_area)[0:5]

    return v_vent_r * a_hcz / a_hcz_r


def get_minimum_air_volume(v_vent: np.ndarray) -> (float, float):
    """
    Args:
        v_vent: supply air volume of mechanical ventilation, m3/h, (5 rooms)
    Returns:
        minimum supply air volume of the system for heating and cooling
    """

    htg = np.sum(v_vent)
    clg = np.sum(v_vent)

    return htg, clg


def get_partition_area(floor_area: envelope.FloorArea):
    """
    calculate the areas of the partition
    Args:
        floor_area: floor area of 3 zones, m2
    Returns:
        the areas of the partitions, m2
    """
    # floor area of each 12 zones, m2
    a_hcz = envelope.get_hc_floor_areas(floor_area=floor_area)

    # calculate the partition area between main occupant room and non occupant room, m2
    a_part_mr = a_hcz[0:1] * floor_area.r_env * floor_area.nor / (floor_area.oor + floor_area.nor)

    # calculate the partition areas between 4 other occupant rooms and non occupant room, m2
    a_part_or = a_hcz[1:5] * floor_area.r_env * floor_area.nor / (floor_area.mor + floor_area.nor)

    # concatenate
    return np.concatenate((a_part_mr, a_part_or))


def get_rated_heating_output(system_spec: SystemSpec) -> float:
    """
    calculate the rated heating output
    Args:
        system_spec: spec of the system, class SystemSpec
    Returns:
        rated heating output, MJ/h
    """

    return system_spec.cap_rtd_h * 3600 * 10 ** (-6)


def get_rated_cooling_output(system_spec: SystemSpec) -> float:
    """
    calculate the rated cooling output
    Args:
        system_spec: spec of the system, class SystemSpec
    Returns:
        rated cooling output, MJ/h
    """

    return system_spec.cap_rtd_c * 3600 * 10 ** (-6)


def get_heating_output_for_supply_air_estimation(
        l_h: np.ndarray,
        q: float,
        theta_ac_h: np.ndarray,
        theta_ex: np.ndarray,
        mu_h: float,
        j: np.ndarray,
        a_nr: float) -> np.ndarray:
    """calculate the system supply air volume for heating
    eq.(12)
    Args:
        l_h: heating load, MJ/h, (12 rooms * 8760 times)
        q: q value, W/m2K
        theta_ac_h: air conditioned temperature for heating, degree C
        theta_ex: outdoor temperature, degree C
        mu_h: mu value, (W/m2)/(W/m2)
        j: horizontal solar radiation, W/m2
        a_nr: floor area of non occupant room, m2
    Returns:
        heating output for supply air estimation, MJ/h
    """

    # heating load in the main occupant room and the other occupant rooms, MJ/h, (5 rooms * 8760 times)
    l_h = l_h[0:5]

    q_dash_hs_h = np.sum(l_h, axis=0) + ((theta_ac_h - theta_ex) * q - j * mu_h) * a_nr * 3600 * 10 ** (-6)

    # This operation is not described in the specification document
    # The supply air has lower limitation. This operation does not eventually effect the result.
    return np.vectorize(lambda x: x if x > 0.0 else 0.0)(q_dash_hs_h)


def get_cooling_output_for_supply_air_estimation(
        l_cs: np.ndarray,
        l_cl: np.ndarray,
        q: float,
        theta_ac_c: np.ndarray,
        theta_ex: np.ndarray,
        mu_c: float,
        j: np.ndarray,
        a_nr: float) -> np.ndarray:
    """calculate the system supply air volume for cooling
    eq.(27)
    Args:
        l_cs: sensible cooling load, MJ/h, (12 rooms * 8760 times)
        l_cl: latent cooling load, MJ/h, (12 rooms * 8760 times)
        q: q value, W/m2K
        theta_ac_c: air conditioned temperature for cooling, degree C
        theta_ex: outdoor temperature, degree C
        mu_c: mu value, (W/m2)/(W/m2)
        j: horizontal solar radiation, W/m2
        a_nr: floor area of non occupant room, m2
    Returns:
        sensible and latent cooling output for supply air estimation, MJ/h
    """

    # sensible cooling load in the main occupant room and the other occupant rooms, MJ/h
    l_cs = l_cs[0:5]

    # latent cooling load in the main occupant room and the other occupant rooms, MJ/h
    l_cl = l_cl[0:5]

    q_dash_hs_c = np.sum(l_cs, axis=0) + np.sum(l_cl, axis=0) \
        + ((theta_ex - theta_ac_c) * q + j * mu_c) * a_nr * 3600 * 10 ** (-6)

    # This operation is not described in the specification document
    # The supply air has lower limitation. This operation does not eventually effect the result.
    return np.vectorize(lambda x: x if x > 0.0 else 0.0)(q_dash_hs_c)


def get_heat_source_supply_air_volume_for_heating(
        q_dash_hs_h: np.ndarray,
        q_hs_rtd_h: float,
        v_hs_min_h: float,
        v_hs_rtd_h: float) -> np.ndarray:
    """
    calculate the supply air volume for heating
    Args:
        q_dash_hs_h: heating output of the system for estimation of the supply air volume, MJ/h
        q_hs_rtd_h: rated heating output, MJ/h
        v_hs_min_h: minimum supply air volume, m3/h
        v_hs_rtd_h: rated (maximum) supply air volume, m3/h
    Returns:
        supply air volume, m3/h (8760 times)
    """

    # get the supply air volume depending on the heating output
    def f(q):
        if q < 0.0:
            return v_hs_min_h
        elif q < q_hs_rtd_h:
            return (v_hs_rtd_h - v_hs_min_h) / q_hs_rtd_h * q + v_hs_min_h
        else:
            return v_hs_rtd_h

    return np.vectorize(f)(q_dash_hs_h)


def get_heat_source_supply_air_volume_for_cooling(
        q_dash_hs_c: np.ndarray,
        q_hs_rtd_c: float,
        v_hs_min_c: float,
        v_hs_rtd_c: float) -> np.ndarray:
    """
    calculate the supply air volume for cooling
    Args:
        q_dash_hs_c: cooling output of the system for estimation of the supply air volume, MJ/h
        q_hs_rtd_c: rated cooling output, MJ/h
        v_hs_min_c: minimum supply air volume, m3/h
        v_hs_rtd_c: rated (maximum) supply air volume, m3/h
    Returns:
        supply air volume, m3/h (8760 times)
    """

    # get the supply air volume depending on the cooling output
    def f(q):
        if q < 0.0:
            return v_hs_min_c
        elif q < q_hs_rtd_c:
            return (v_hs_rtd_c - v_hs_min_c) / q_hs_rtd_c * q + v_hs_min_c
        else:
            return v_hs_rtd_c

    return np.vectorize(f)(q_dash_hs_c)


def get_each_supply_air_volume_for_heating(
        r_supply_des: np.ndarray,
        v_hs_supply_h: np.ndarray,
        v_vent: np.ndarray) -> np.ndarray:
    """
    Args:
        r_supply_des: supply air volume valance, (5 rooms)
        v_hs_supply_h: total supply air volume for heating, m3/h
        v_vent: mechanical ventilation, m3/h (5 rooms)
    Returns:
        supply air volume, m3/h (5 rooms * 8760 times)
    """

    # supply air volume valance
    r_supply_des = r_supply_des.reshape(1, 5).T

    # mechanical ventilation, m3/h (5 rooms * 1 value)
    v_vent = v_vent.reshape(1, 5).T

    return np.maximum(v_hs_supply_h * r_supply_des, v_vent)


def get_each_supply_air_volume_for_cooling(
        r_supply_des, v_hs_supply_c, v_vent) -> np.ndarray:
    """
    Args:
        r_supply_des: supply air volume valance, (5 rooms)
        v_hs_supply_c: total supply air volume for heating, m3/h, (8760 times)
        v_vent: mechanical ventilation, m3/h (5 rooms)
    Returns:
        supply air volume, m3/h (5 rooms * 8760 times)
    """

    # supply air volume valance(5 rooms * 1)
    r_supply_des = r_supply_des.reshape(1, 5).T

    # mechanical ventilation, m3/h (5 rooms * 1)
    v_vent = v_vent.reshape(1, 5).T

    return np.maximum(v_hs_supply_c * r_supply_des, v_vent)


def get_non_occupant_room_temperature_for_heating(
        q_value: float,
        theta_ex: np.ndarray,
        mu_value: float,
        j: np.ndarray,
        a_nr: float,
        c: float,
        rho: float,
        v_supply_h: np.ndarray,
        u_prt: float,
        a_prt: np.ndarray,
        theta_ac_h: np.ndarray) -> np.ndarray:
    """
    Args:
        q_value: Q value, W/m2K
        theta_ex: outdoor temperature, degree C
        mu_value: mu value, (W/m2K)/(W/m2K)
        j: horizontal solar radiation, W/m2K
        a_nr: floor area of non occupant room, m2
        c: specific heat of air, J/kg K
        rho: air density, kg/m3
        v_supply_h: supply air volume, m3/h
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2
        theta_ac_h: air conditioned temperature for heating, degree C
    Returns:
        non occupant room temperature, degree C (8760 times)
    """

    # area of the partition, m2
    a_prt = a_prt.reshape(1, 5).T

    return ((q_value * theta_ex + mu_value * j) * a_nr
            + np.sum(c * rho * v_supply_h / 3600 + u_prt * a_prt, axis=0) * theta_ac_h) \
           / (q_value * a_nr + np.sum(c * rho * v_supply_h / 3600 + u_prt * a_prt, axis=0))


def get_non_occupant_room_temperature_for_cooling(
        q_value, theta_ex, mu_value, j, a_nr, c, rho, v_supply_c, u_prt, a_prt, theta_ac_c) -> np.ndarray:
    """
    Args:
        q_value: Q value, W/m2K
        theta_ex: outdoor temperature, degree C, (8760 times)
        mu_value: mu value, (W/m2K)/(W/m2K)
        j: horizontal solar radiation, W/m2, (8760 times)
        a_nr: floor area of non occupant room, m2
        c: specific heat of air, J/kg K
        rho: air density, kg/m3
        v_supply_c: supply air volume, m3/h, (5 rooms * 8760 times)
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2, (5 rooms)
        theta_ac_c: air conditioned temperature for heating, degree C, (8760 times)
    Returns:
        non occupant room temperature, degree C (8760 times)
    """

    a_prt = a_prt.reshape(1, 5).T

    return ((q_value * theta_ex + mu_value * j) * a_nr
            + np.sum(c * rho * v_supply_c / 3600 + u_prt * a_prt, axis=0) * theta_ac_c) \
        / (q_value * a_nr + np.sum(c * rho * v_supply_c / 3600 + u_prt * a_prt, axis=0))


def get_heat_loss_through_partition_for_heating(
        u_prt: float,
        a_prt: np.ndarray,
        theta_ac_h: np.ndarray,
        theta_nac_h: np.ndarray) -> np.ndarray:
    """
    Args:
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2, (5 rooms)
        theta_ac_h: air conditioned temperature for heating, degree C
        theta_nac_h: non occupant room temperature, degree C (8760 times)
    Returns:
        heat loss through the partition, MJ/h (5 rooms * 8760 times)
    """

    # area of the partition, m2
    a_prt = a_prt.reshape(1, 5).T

    return u_prt * a_prt * (theta_ac_h - theta_nac_h) * 3600 * 10 ** (-6)


def get_heat_gain_through_partition_for_cooling(
        u_prt: float,
        a_prt: np.ndarray,
        theta_ac_c: np.ndarray,
        theta_nac_c: np.ndarray) -> np.ndarray:
    """
    Args:
        u_prt: heat loss coefficient of the partition wall, W/m2K
        a_prt: area of the partition, m2
        theta_ac_c: air conditioned temperature for heating, degree C
        theta_nac_c: non occupant room temperature, degree C (8760 times)
    Returns:
        heat gain through the partition, MJ/h (5 rooms * 8760 times)
    """

    # area of the partition, m2
    a_prt = a_prt.reshape(1, 5).T

    return u_prt * a_prt * (theta_nac_c - theta_ac_c) * 3600 * 10 ** (-6)


def get_maximum_output_for_heating(
        theta_hs_in_h: np.ndarray,
        q_hs_max_h: np.ndarray,
        c: float,
        rho: float,
        v_supply_h: np.ndarray,
        theta_ac_h: np.ndarray,
        psi: float,
        l_duct: np.ndarray,
        theta_sur_h: np.ndarray) -> np.ndarray:
    """
    calculate maximum output for heating
    Args:
        theta_hs_in_h: inlet air temperature of the heat source for heating, degree C (8760 times)
        q_hs_max_h: maximum heating output, MJ/h (8760 times)
        c: specific heat of air, J/kgK
        rho: air density, kg/m3
        v_supply_h: supply air volume for heating, m3/h (5 rooms * 8760 times)
        theta_ac_h: air conditioned temperature for heating, degree C
        psi: linear heat loss coefficient of the duct, W/mK
        l_duct: duct length, m, (5 rooms)
        theta_sur_h: ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
    Returns:
        maximum output for heating, MJ/h, (5 rooms * 8760 times)
    """

    # maximum outlet air temperature of heat source, degree C, (8760 times)
    theta_hs_out_max_h = theta_hs_in_h + q_hs_max_h / (c * rho * np.sum(v_supply_h, axis=0)) * 10 ** 6

    l_duct = np.array(l_duct).reshape(1,5).T

    return get_load_from_upside_temperature(
        t_sur=theta_sur_h, t_up=theta_hs_out_max_h, v=v_supply_h, t_ac=theta_ac_h, psi=psi, length=l_duct)


def get_maximum_output_for_cooling(
        theta_hs_in_c: np.ndarray,
        l_cs: np.ndarray,
        l_cl: np.ndarray,
        q_trs_prt_c: np.ndarray,
        q_hs_max_cs: np.ndarray,
        q_hs_max_cl: np.ndarray,
        c: float,
        rho: float,
        v_supply_c: np.ndarray,
        theta_ac_c: np.ndarray,
        psi: float,
        l_duct: np.ndarray,
        theta_sur_c: np.ndarray) -> np.ndarray:
    """
    calculate maximum output for cooling
    Args:
        theta_hs_in_c: inlet air temperature of the heat source for cooling, degree C (8760 times)
        l_cs: sensible cooling load, MJ/h (12 rooms * 8760 times)
        l_cl: latent cooling load, MJ/h (12 rooms * 8760 times)
        q_trs_prt_c: heat gain from non occupant room into occupant room through partition for cooling, MJ/h (5 rooms * 8760 times)
        q_hs_max_cs: maximum sensible cooling output, MJ/h (8760 times)
        q_hs_max_cl: maximum latent cooling output, MJ/h (8760 times)
        c: specific heat of air, J/kgK
        rho: air density, kg/m3
        v_supply_c: supply air volume for cooling, m3/h (5 rooms * 8760 times)
        theta_ac_c: air conditioned temperature for cooling, degree C
        psi: linear heat loss coefficient of the duct, W/mK
        l_duct: duct length, m, (5 rooms)
        theta_sur_c: ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
    Returns:
        maximum output for sensible cooling, MJ/h, (5 rooms * 8760 times), maximum output for latent cooling, MJ/h, (5 rooms * 8760 times)
    """

    # sensible cooling load, MJ/h (5 rooms * 8760 times)
    l_cs = l_cs[0:5]

    # latent cooling load, MJ/h (5 rooms * 8760 times)
    l_cl = l_cl[0:5]

    # minimum outlet air temperature of heat source, degree C, (8760 times)
    theta_hs_out_min_c = theta_hs_in_c - q_hs_max_cs / (c * rho * np.sum(v_supply_c, axis=0)) * 10 ** 6

    # duct length, m
    l_duct = np.array(l_duct).reshape(1, 5).T

    q_max_cs = (theta_ac_c - theta_sur_c
                + (theta_sur_c - theta_hs_out_min_c) / np.exp(psi * l_duct * 3600 / (c * rho * v_supply_c))) \
        * c * rho * v_supply_c * 10 ** (-6)

    l_cl_sum = np.sum(l_cl, axis=0)

    r = np.vectorize(lambda x, y: x / y if y > 0.0 else 0.0)(l_cl, l_cl_sum)

    q_max_cl = r * q_hs_max_cl

    return q_max_cs, q_max_cl


def get_treated_untreated_heat_load_for_heating(
        l_h: np.ndarray,
        q_trs_prt_h: np.ndarray,
        q_max_h: np.ndarray) -> np.ndarray:
    """
    Args:
        l_h: heating load, MJ/h (12 rooms * 8760 times)
        q_trs_prt_h: heat loss from the occupant room into the non occupant room through the partition, MJ/h (5 rooms * 8760 times)
        q_max_h: maximum output for heating, MJ/h
    Returns:
        (a,b)
            a: treated heating load, MJ/h, (5 rooms * 8760 times)
            b: untreated heating load, MJ/h, (5 rooms * 8760 times)
    """

    # heating load, MJ/h (5 rooms * 8760 times)
    l_h = l_h[0:5]

    # treated load, MJ/h
    q_t_h = np.clip(l_h + q_trs_prt_h, 0.0, q_max_h)

    # untreated load, MJ/h
    q_ut_h = np.maximum(l_h + q_trs_prt_h, 0.0) - q_t_h

    return q_t_h, q_ut_h


def get_treated_untreated_heat_load_for_cooling(
        l_cs: np.ndarray,
        l_cl: np.ndarray,
        q_trs_prt_c: np.ndarray,
        q_max_cs: np.ndarray,
        q_max_cl: np.ndarray) -> np.ndarray:
    """
    Args:
        l_cs: sensible cooling load, MJ/h (5 rooms * 8760 times)
        l_cl: latent cooling load, MJ/h (5 rooms, 8760 times)
        q_trs_prt_c: heat gain from the non occupant room into the occupant room through the partition, MJ/h (5 rooms * 8760 times)
        q_max_cs: maximum output for sensible cooling, MJ/h, (5 rooms * 8760 times)
        q_max_cl: maximum output for latent cooling, MJ/h, (5 rooms * 8760 times)
    Returns:
        (a,b,c,d)
            a: treated sensible heating load, MJ/h, (5 rooms * 8760 times)
            b: treated latent heating load, MJ/h, (5 rooms * 8760 times)
            c: untreated sensible heating load, MJ/h, (5 rooms * 8760 times)
            d: untreated latent heating load, MJ/h, (5 rooms * 8760 times)
    """

    # sensible cooling load, MJ/h (5 rooms * 8760 times)
    l_cs = l_cs[0:5]

    # latent cooling load, MJ/h (5 rooms, 8760 times)
    l_cl = l_cl[0:5]

    # treated load, MJ/h
    #  sensible
    q_t_cs = np.minimum(q_max_cs, np.maximum(l_cs + q_trs_prt_c, 0.0))
    #  latent
    q_t_cl = np.minimum(q_max_cl, l_cl)

    # untreated load, MJ/h
    #  sensible
    q_ut_cs = np.maximum(l_cs + q_trs_prt_c, 0.0) - q_t_cs
    #  latent
    q_ut_cl = l_cl - q_t_cl

    return q_t_cs, q_t_cl, q_ut_cs, q_ut_cl


def get_requested_supply_air_temperature_for_heating(
        theta_sur_h: np.ndarray,
        theta_ac_h: np.ndarray,
        q_t_h: np.ndarray,
        v_supply_h: np.ndarray,
        c: float,
        rho: float,
        psi: float,
        l_duct: np.ndarray) -> np.ndarray:
    """
    calculate the requested supply air temperature for heating
    Args:
        theta_sur_h: ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
        theta_ac_h: air conditioned temperature for heating, degree C
        q_t_h: treated heat load for heating, MJ/h, (5 rooms * 8760 times)
        v_supply_h: supply air volume for heating, m3/h (5 rooms * 8760 times)
        c: specific heat of air, J/kgK
        rho: air density, kg/m3
        psi: linear heat loss coefficient of the duct, W/mK
        l_duct: duct length, m, (5 rooms)
    Returns:
        requested temperature, degree C, (5 rooms * 8760 times)
    """

    l_duct = np.array(l_duct).reshape(1,5).T

    return theta_sur_h + (theta_ac_h + q_t_h * 10 ** 6 / (v_supply_h * c * rho) - theta_sur_h) \
        * np.exp(psi * l_duct * 3600 / (v_supply_h * c * rho))


def get_requested_supply_air_temperature_for_cooling(
        theta_sur_c: np.ndarray,
        theta_ac_c: np.ndarray,
        q_t_cs: np.ndarray,
        v_supply_c: np.ndarray,
        c: float,
        rho: float,
        psi: float,
        l_duct: np.ndarray) -> np.ndarray:
    """
    calculate the requested supply air temperature for heating
    Args:
        theta_sur_c: ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
        theta_ac_c: air conditioned temperature for cooling, degree C
        q_t_cs: treated heat load for cooling, MJ/h, (5 rooms * 8760 times)
        v_supply_c: supply air volume for heating, m3/h (5 rooms * 8760 times)
        c: specific heat of air, J/kgK
        rho: air density, kg/m3
        psi: linear heat loss coefficient of the duct, W/mK
        l_duct: duct length, m
    Returns:
        requested temperature, degree C, (5 rooms * 8760 times)
    """

    l_duct = np.array(l_duct).reshape(1,5).T

    return theta_sur_c - (theta_sur_c - theta_ac_c + q_t_cs * 10 ** 6 / (v_supply_c * c * rho)) \
        * np.exp(psi * l_duct * 3600 / (v_supply_c * c * rho))


def calc_decided_outlet_supply_air_temperature_for_heating(
        theta_duct_up_h: np.ndarray) -> np.ndarray:
    """
    decide the outlet supply air temperature for heating
    Args:
        theta_duct_up_h: requested temperature, degree C, (5 rooms * 8760 times)
    Returns:
        decided outlet supply air temperature, degree C, (8760 times)
    """

    return np.max(theta_duct_up_h, axis=0)


def calc_decided_outlet_supply_air_temperature_for_cooling(
        theta_duct_up_c: np.ndarray) -> np.ndarray:
    """
    decide the outlet supply air temperature for cooling
    Args:
        theta_duct_up_c: requested temperature, degree C, (5 rooms * 8760 times)
    Returns:
        decided outlet supply air temperature, degree C, (8760 times)
    """

    return np.min(theta_duct_up_c, axis=0)


def calc_heat_source_heating_output(
        theta_hs_out_h: np.ndarray,
        theta_hs_in_h: np.ndarray,
        c: float,
        rho: float,
        v_supply_h: np.ndarray) -> np.ndarray:
    """
    calculate heat source heating output
    Args:
        theta_hs_out_h: supply air temperature, degree C, (5 rooms * 8760 times)
        theta_hs_in_h: inlet air temperature of the heat source for heating, degree C (8760 times)
        c: specific heat of air, J/kgK
        rho: air density, kg/m3
        v_supply_h: supply air volume for heating, m3/h (5 rooms * 8760 times)
    Returns:
        heating output, MJ/h, (8760 times)
    """

    return np.maximum((theta_hs_out_h - theta_hs_in_h) * c * rho * np.sum(v_supply_h, axis=0) * 10 ** (-6), 0.0)


def calc_heat_source_cooling_output(
        theta_hs_in_c: np.ndarray,
        theta_hs_out_c: np.ndarray,
        c: float,
        rho: float,
        v_supply_c: np.ndarray,
        l_cl: np.ndarray) -> np.ndarray:
    """
    Args:
        theta_hs_in_c: inlet air temperature of the heat source for cooling, degree C (8760 times)
        theta_hs_out_c: supply air temperature, degree C (8760 times)
        c: specific heat of air, J/kgK
        rho: air density, kg/m3
        v_supply_c: supply air volume for cooling, m3/h (5 rooms * 8760 times)
        l_cl: latent cooling load, MJ/h, (12 rooms * 8760 times)
    """

    q_hs_cs = np.maximum((theta_hs_in_c - theta_hs_out_c) * c * rho * np.sum(v_supply_c, axis=0) * 10 ** (-6), 0.0)

    q_hs_cl = np.sum(l_cl[0:5], axis=0)

    return q_hs_cs, q_hs_cl


def get_duct_heat_loss_for_heating(
        theta_sur_h: np.array,
        theta_hs_out_h: np.array,
        v_supply_h: np.array,
        theta_ac_h: np.array,
        psi: float,
        l_duct: np.array) -> np.ndarray:
    """
    Args:
        theta_sur_h: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_h: outlet temperature of heat source, degree C, (8760 times)
        v_supply_h: supply air volume, m3/h (5 rooms * 8760 times)
        theta_ac_h: air conditioned temperature, degree C, (8760 times)
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m, (5 rooms)
    """

    l_duct = np.array(l_duct).reshape(1,5).T

    return get_duct_heat_loss_from_upside_temperature(
        theta_sur_h, theta_hs_out_h, v_supply_h, theta_ac_h, psi, l_duct)


def get_actual_treated_load_for_heating(
        theta_sur_h: np.array,
        theta_hs_out_h: np.array,
        v_supply_h: np.array,
        theta_ac_h: float,
        psi: float,
        l_duct: np.array) -> np.ndarray:
    """
    Args:
        theta_sur_h: duct ambient temperature, degree C, (5 rooms * 8760 times)
        theta_hs_out_h: supply air temperature, degree C, (5 rooms * 8760 times)
        v_supply_h: supply air volume for heating, m3/h (5 rooms * 8760 times)
        theta_ac_h: air conditioned temperature, degree C
        psi: liner heat loss coefficient, W/mK
        l_duct: duct length, m (5 rooms)
    Returns:
        actual treated load for heating, MJ/h, (5 rooms * 8760 times)
    """

    # duct length, m
    l_duct = np.array(l_duct).reshape(1, 5).T

    return get_load_from_upside_temperature(
        theta_sur_h, theta_hs_out_h, v_supply_h, theta_ac_h, psi, l_duct)


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
        duct heat loss, MJ/h
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    return (t_up - t_sur) * (1 - np.exp(- psi * length * 3600 / (c * rho * v))) * c * rho * v * 10 ** (-6)


def get_downside_temperature_from_upside_temperature(
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
        downside_temperature, degree C
    """

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    return t_sur + (t_up - t_sur) * np.exp(- psi * length * 3600 / (c * rho * v))

# endregion


def get_non_occupant_room_load(
        theta_nac_h: np.array, theta_ac_h: np.array, v_supply_h: np.array, c: float, rho: float):
    """
    calculate non occupant room load
    Args:
        theta_nac_h: non occupant room temperature, degree C (8760 times)
        theta_ac_h: air conditioned temperature, degree C, (8760 times)
        v_supply_h: supply air volume, m3/h (5 rooms * 8760 times)
        c: air specific heat, J/kg K
        rho: air density, kg/m3
    Returns:
        non occupant room load, MJ/h, (8760 times)
    """

    return (theta_ac_h - theta_nac_h) * np.sum(v_supply_h, axis=0) * c * rho * 10 ** (-6)

# endregion


def get_main_value(
        region: int,
        a_mr: float, a_or: float, a_a: float, r_env: float,
        insulation: str, solar_gain: str,
        default_heat_source_spec: bool,
        supply_air_rtd_h: float, supply_air_rtd_c: float,
        is_duct_insulated: bool, vav_system: bool,
        cap_rtd_h: float =None, cap_rtd_c: float=None):
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
        supply_air_rtd_h: rated supply air volume for heating, m3/h
        supply_air_rtd_c: rated supply air volume for cooling, m3/h
        is_duct_insulated: is the duct inside the insulated area or not
        vav_system: is VAV system applied ?
        cap_rtd_h: rated heating capacity, W
        cap_rtd_c: rated cooling capacity, W
    """

    # make envelope.FloorArea class
    floor_area = envelope.FloorArea(a_mr, a_or, a_a, r_env)

    # make envelope.Spec class
    envelope_spec = envelope.Spec(insulation, solar_gain)

    # set default value for heating and cooling capacity, W
    if default_heat_source_spec:
        cap_rtd_h, cap_rtd_c = appendix.get_rated_capacity(region, floor_area)

    # make appendix.SystemSpec class
    system_spec = SystemSpec(cap_rtd_h, cap_rtd_c, supply_air_rtd_h, supply_air_rtd_c, is_duct_insulated, vav_system)

    # air density, kg/m3
    rho = get_air_density()

    # air specific heat, J/kg K
    c = get_specific_heat()

    psi = get_duct_linear_heat_loss_coefficient()

    # heating load, MJ/h
    l_h = read_load.get_heating_load(region, envelope_spec, floor_area)
    l_cs = read_load.get_sensible_cooling_load(region, envelope_spec, floor_area)
    l_cl = read_load.get_latent_cooling_load(region, envelope_spec, floor_area)

    # duct length in the standard house, m, ((5 rooms), (5 rooms), (5 rooms))
    l_duct_in_r, l_duct_ex_r, l_duct_in_total = get_standard_house_duct_length()

    # duct length for each room, m, (5 rooms)
    l_duct_i = calc_duct_length(a_a=floor_area.total)

    # air conditioned temperature, degree C, (8760 times)
    theta_ac_h = np.full(8760, get_air_conditioned_temperature_for_heating())
    theta_ac_c = np.full(8760, get_air_conditioned_temperature_for_cooling())

    # SAT temperature, degree C, (8760 times)
    theta_sat = read_conditions.get_sat_temperature(region)

    # attic temperature, degree C, (8760 times)
    theta_attic_h = get_attic_temperature_for_heating(theta_sat, theta_ac_h, 1.0)
    theta_attic_c = get_attic_temperature_for_cooling(theta_sat, theta_ac_c, 1.0)

    # duct ambient temperature, degree C, (5 rooms * 8760 times)
    theta_sur_h = get_duct_ambient_air_temperature_for_heating(
        system_spec.is_duct_insulated, l_duct_in_r, l_duct_ex_r, theta_ac_h, theta_attic_h)
    theta_sur_c = get_duct_ambient_air_temperature_for_cooling(
        system_spec.is_duct_insulated, l_duct_in_r, l_duct_ex_r, theta_ac_c, theta_attic_c)

    r_supply_des = get_supply_air_volume_valance(floor_area)

    v_vent = get_mechanical_ventilation(floor_area)

    v_hs_min_h, v_hs_min_c = get_minimum_air_volume(v_vent)

    a_part = get_partition_area(floor_area)

    q_hs_rtd_h = get_rated_heating_output(system_spec)
    q_hs_rtd_c = get_rated_cooling_output(system_spec)

    # Q value, W/m2K
    q_value = envelope_spec.get_q_value(region=region)

    # outdoor temperature, degree C
    theta_ex = read_conditions.read_temperature(region=region)

    # mu value, (W/m2)/(W/m2)
    mu_h = envelope_spec.get_mu_h_value(region=region)
    mu_c = envelope_spec.get_mu_c_value(region=region)

    # horizontal solar radiation, W/m2K
    j = read_conditions.get_horizontal_solar(region=region)

    # floor area of non occupant room, m2
    a_nr = floor_area.nor

    q_d_hs_h = get_heating_output_for_supply_air_estimation(l_h, q_value, theta_ac_h, theta_ex, mu_h, j, a_nr)
    q_d_hs_c = get_cooling_output_for_supply_air_estimation(l_cs, l_cl, q_value, theta_ac_c, theta_ex, mu_c, j, a_nr)

    # rated (maximum) supply air volume, m3/h
    v_hs_rtd_h = system_spec.supply_air_rtd_h
    v_hs_rtd_c = system_spec.supply_air_rtd_c

    v_hs_supply_h = get_heat_source_supply_air_volume_for_heating(q_d_hs_h, q_hs_rtd_h, v_hs_min_h, v_hs_rtd_h)
    v_hs_supply_c = get_heat_source_supply_air_volume_for_cooling(q_d_hs_c, q_hs_rtd_c, v_hs_min_c, v_hs_rtd_c)

    # supply air volume, m3/h (5 rooms * 8760 times)
    v_supply_h = get_each_supply_air_volume_for_heating(r_supply_des, v_hs_supply_h, v_vent)
    v_supply_c = get_each_supply_air_volume_for_cooling(r_supply_des, v_hs_supply_c, v_vent)

    # heat loss coefficient of the partition wall, W/m2K
    u_prt = get_heat_loss_coefficient_of_partition()

    theta_nac_h = get_non_occupant_room_temperature_for_heating(
        q_value, theta_ex, mu_h, j, a_nr, c, rho, v_supply_h, u_prt, a_part, theta_ac_h)
    theta_nac_c = get_non_occupant_room_temperature_for_cooling(
        q_value, theta_ex, mu_c, j, a_nr, c, rho, v_supply_c, u_prt, a_part, theta_ac_c)

    q_trs_prt_h = get_heat_loss_through_partition_for_heating(u_prt, a_part, theta_ac_h, theta_nac_h)
    q_trs_prt_c = get_heat_gain_through_partition_for_cooling(u_prt, a_part, theta_ac_c, theta_nac_c)

    # maximum heating output, MJ/h (8760 times)
    q_hs_max_h = appendix.get_maximum_heating_output(region, system_spec)
    # maximum sensible cooling output, MJ/h (8760 times), maximum latent cooling output, MJ/h (8760 times)
    q_hs_max_cs, q_hs_max_cl = appendix.get_maximum_cooling_output(system_spec, l_cs, q_trs_prt_c, l_cl)

    # inlet air temperature of heat source,degree C, (8760 times)
    theta_hs_in_h = theta_nac_h
    theta_hs_in_c = theta_nac_c

    q_max_h = get_maximum_output_for_heating(
        theta_hs_in_h, q_hs_max_h, c, rho, v_supply_h, theta_ac_h, psi, l_duct_i, theta_sur_h)
    q_max_cs, q_max_cl = get_maximum_output_for_cooling(
        theta_hs_in_c, l_cs, l_cl, q_trs_prt_c, q_hs_max_cs, q_hs_max_cl, c, rho, v_supply_c, theta_ac_c, psi,
        l_duct_i, theta_sur_c)

    q_t_h, q_ut_h = get_treated_untreated_heat_load_for_heating(l_h, q_trs_prt_h, q_max_h)
    q_t_cs, q_t_cl, q_ut_cs, q_ut_cl = get_treated_untreated_heat_load_for_cooling(
        l_cs, l_cl, q_trs_prt_c, q_max_cs, q_max_cl)

    # requested supply air temperature, degree C, (5 rooms * 8760 times)
    theta_duct_up_h = get_requested_supply_air_temperature_for_heating(
        theta_sur_h, theta_ac_h, q_t_h, v_supply_h, c, rho, psi, l_duct_i)
    theta_duct_up_c = get_requested_supply_air_temperature_for_cooling(
        theta_sur_c, theta_ac_c, q_t_cs, v_supply_c, c, rho, psi, l_duct_i)

    # outlet temperature of heat source, degree C, (8760 times)
    theta_hs_out_h = calc_decided_outlet_supply_air_temperature_for_heating(theta_duct_up_h)
    theta_hs_out_c = calc_decided_outlet_supply_air_temperature_for_cooling(theta_duct_up_c)

    # output of heat source, MJ/h, (8760 times)
    q_hs_h = calc_heat_source_heating_output(theta_hs_out_h, theta_hs_in_h, c, rho, v_supply_h)
    q_hs_cs, q_hs_cl = calc_heat_source_cooling_output(theta_hs_in_c, theta_hs_out_c, c, rho, v_supply_c, l_cl)

    # heat loss from ducts, MJ/h, (5 rooms * 8760 times)
    q_loss_duct_h = get_duct_heat_loss_for_heating(theta_sur_h, theta_hs_out_h, v_supply_h, theta_ac_h, psi, l_duct_i)

    # actual treated load for heating, MJ/h, (5 rooms * 8760 times)
    q_act_h = get_actual_treated_load_for_heating(theta_sur_h, theta_hs_out_h, v_supply_h, theta_ac_h, psi, l_duct_i)

    l_nor = get_non_occupant_room_load(theta_nac_h, theta_ac_h, v_supply_h, c, rho)

    return {
        'constant_value': {
            'air_density': rho,  # kg/m3
            'air_specific_heat': c,  # J/kgK
            'duct_length_room1': l_duct_i[0],  # m
            'duct_length_room2': l_duct_i[1],  # m
            'duct_length_room3': l_duct_i[2],  # m
            'duct_length_room4': l_duct_i[3],  # m
            'duct_length_room5': l_duct_i[4],  # m
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
            'minimum_supply_air_volume_of_heat_source_heating': v_hs_min_h,  # m3/h
            'minimum_supply_air_volume_of_heat_source_cooling': v_hs_min_c,  # m3/h
            'partition_area_room1': a_part[0],  # m2
            'partition_area_room2': a_part[1],  # m2
            'partition_area_room3': a_part[2],  # m2
            'partition_area_room4': a_part[3],  # m2
            'partition_area_room5': a_part[4],  # m2
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
            'air_conditioned_temperature_heating': theta_ac_h,  # degree C
            'air_conditioned_temperature_cooling': theta_ac_c,  # degree C
            'sat_temperature': theta_sat,  # degree C
            'attic_temperature_heating': theta_attic_h,  # degree C
            'attic_temperature_cooling': theta_attic_c,  # degree C
            'duct_ambient_temperature_heating_room1': theta_sur_h[0],  # degree C
            'duct_ambient_temperature_heating_room2': theta_sur_h[1],  # degree C
            'duct_ambient_temperature_heating_room3': theta_sur_h[2],  # degree C
            'duct_ambient_temperature_heating_room4': theta_sur_h[3],  # degree C
            'duct_ambient_temperature_heating_room5': theta_sur_h[4],  # degree C
            'duct_ambient_temperature_cooling_room1': theta_sur_c[0],  # degree C
            'duct_ambient_temperature_cooling_room2': theta_sur_c[1],  # degree C
            'duct_ambient_temperature_cooling_room3': theta_sur_c[2],  # degree C
            'duct_ambient_temperature_cooling_room4': theta_sur_c[3],  # degree C
            'duct_ambient_temperature_cooling_room5': theta_sur_c[4],  # degree C
            'output_of_heat_source_for_supply_air_volume_estimation_heating': q_d_hs_h,  # MJ/h
            'output_of_heat_source_for_supply_air_volume_estimation_cooling': q_d_hs_c,  # MJ/h
            'supply_air_volume_of_heat_source_heating': v_hs_supply_h,  # MJ/h
            'supply_air_volume_of_heat_source_cooling': v_hs_supply_c,  # MJ/h
            'supply_air_volume_heating_room1': v_supply_h[0],  # MJ/h
            'supply_air_volume_heating_room2': v_supply_h[1],  # MJ/h
            'supply_air_volume_heating_room3': v_supply_h[2],  # MJ/h
            'supply_air_volume_heating_room4': v_supply_h[3],  # MJ/h
            'supply_air_volume_heating_room5': v_supply_h[4],  # MJ/h
            'supply_air_volume_cooling_room1': v_supply_c[0],  # MJ/h
            'supply_air_volume_cooling_room2': v_supply_c[1],  # MJ/h
            'supply_air_volume_cooling_room3': v_supply_c[2],  # MJ/h
            'supply_air_volume_cooling_room4': v_supply_c[3],  # MJ/h
            'supply_air_volume_cooling_room5': v_supply_c[4],  # MJ/h
            'non_occupant_room_temperature_heating': theta_nac_h,  # degree C
            'non_occupant_room_temperature_cooling': theta_nac_c,  # degree C
            'heat_loss_through_partition_heating_room1': q_trs_prt_h[0],  # MJ/h
            'heat_loss_through_partition_heating_room2': q_trs_prt_h[1],  # MJ/h
            'heat_loss_through_partition_heating_room3': q_trs_prt_h[2],  # MJ/h
            'heat_loss_through_partition_heating_room4': q_trs_prt_h[3],  # MJ/h
            'heat_loss_through_partition_heating_room5': q_trs_prt_h[4],  # MJ/h
            'heat_gain_through_partition_cooling_room1': q_trs_prt_c[0],  # MJ/h
            'heat_gain_through_partition_cooling_room2': q_trs_prt_c[1],  # MJ/h
            'heat_gain_through_partition_cooling_room3': q_trs_prt_c[2],  # MJ/h
            'heat_gain_through_partition_cooling_room4': q_trs_prt_c[3],  # MJ/h
            'heat_gain_through_partition_cooling_room5': q_trs_prt_c[4],  # MJ/h
            'maximum_output_heating_room1': q_max_h[0],  # MJ/h
            'maximum_output_heating_room2': q_max_h[1],  # MJ/h
            'maximum_output_heating_room3': q_max_h[2],  # MJ/h
            'maximum_output_heating_room4': q_max_h[3],  # MJ/h
            'maximum_output_heating_room5': q_max_h[4],  # MJ/h
            'maximum_output_sensible_cooling_room1': q_max_cs[0],  # MJ/h
            'maximum_output_sensible_cooling_room2': q_max_cs[1],  # MJ/h
            'maximum_output_sensible_cooling_room3': q_max_cs[2],  # MJ/h
            'maximum_output_sensible_cooling_room4': q_max_cs[3],  # MJ/h
            'maximum_output_sensible_cooling_room5': q_max_cs[4],  # MJ/h
            'maximum_output_latent_cooling_room1': q_max_cl[0],  # MJ/h
            'maximum_output_latent_cooling_room2': q_max_cl[1],  # MJ/h
            'maximum_output_latent_cooling_room3': q_max_cl[2],  # MJ/h
            'maximum_output_latent_cooling_room4': q_max_cl[3],  # MJ/h
            'maximum_output_latent_cooling_room5': q_max_cl[4],  # MJ/h
            'treated_heating_load_room1': q_t_h[0],  # MJ/h
            'treated_heating_load_room2': q_t_h[1],  # MJ/h
            'treated_heating_load_room3': q_t_h[2],  # MJ/h
            'treated_heating_load_room4': q_t_h[3],  # MJ/h
            'treated_heating_load_room5': q_t_h[4],  # MJ/h
            'treated_sensible_cooling_load_room1': q_t_cs[0],  # MJ/h
            'treated_sensible_cooling_load_room2': q_t_cs[1],  # MJ/h
            'treated_sensible_cooling_load_room3': q_t_cs[2],  # MJ/h
            'treated_sensible_cooling_load_room4': q_t_cs[3],  # MJ/h
            'treated_sensible_cooling_load_room5': q_t_cs[4],  # MJ/h
            'treated_latent_cooling_load_room1': q_t_cl[0],  # MJ/h
            'treated_latent_cooling_load_room2': q_t_cl[1],  # MJ/h
            'treated_latent_cooling_load_room3': q_t_cl[2],  # MJ/h
            'treated_latent_cooling_load_room4': q_t_cl[3],  # MJ/h
            'treated_latent_cooling_load_room5': q_t_cl[4],  # MJ/h
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
            'duct_upside_supply_air_temperature_heating_room1': theta_duct_up_h[0],  # degree C
            'duct_upside_supply_air_temperature_heating_room2': theta_duct_up_h[1],  # degree C
            'duct_upside_supply_air_temperature_heating_room3': theta_duct_up_h[2],  # degree C
            'duct_upside_supply_air_temperature_heating_room4': theta_duct_up_h[3],  # degree C
            'duct_upside_supply_air_temperature_heating_room5': theta_duct_up_h[4],  # degree C
            'duct_upside_supply_air_temperature_cooling_room1': theta_duct_up_c[0],  # degree C
            'duct_upside_supply_air_temperature_cooling_room2': theta_duct_up_c[1],  # degree C
            'duct_upside_supply_air_temperature_cooling_room3': theta_duct_up_c[2],  # degree C
            'duct_upside_supply_air_temperature_cooling_room4': theta_duct_up_c[3],  # degree C
            'duct_upside_supply_air_temperature_cooling_room5': theta_duct_up_c[4],  # degree C
            'outlet_temperature_of_heat_source_heating': theta_hs_out_h,  # degree C
            'outlet_temperature_of_heat_source_cooling': theta_hs_out_c,  # degree C
            'output_of_heat_source_heating': q_hs_h,  # MJ/h
            'output_of_heat_source_sensible_cooling': q_hs_cs,  # MJ/h
            'output_of_heat_source_latent_cooling': q_hs_cl,  # MJ/h
            'duct_heat_loss_heating_room1': q_loss_duct_h[0],  # MJ/h
            'duct_heat_loss_heating_room2': q_loss_duct_h[1],  # MJ/h
            'duct_heat_loss_heating_room3': q_loss_duct_h[2],  # MJ/h
            'duct_heat_loss_heating_room4': q_loss_duct_h[3],  # MJ/h
            'duct_heat_loss_heating_room5': q_loss_duct_h[4],  # MJ/h
            'actual_treated_load_heating_room1': q_act_h[0],  # MJ/h
            'actual_treated_load_heating_room2': q_act_h[1],  # MJ/h
            'actual_treated_load_heating_room3': q_act_h[2],  # MJ/h
            'actual_treated_load_heating_room4': q_act_h[3],  # MJ/h
            'actual_treated_load_heating_room5': q_act_h[4],  # MJ/h
            'non_occupant_room_load': l_nor,  # MJ/h
        },
    }
