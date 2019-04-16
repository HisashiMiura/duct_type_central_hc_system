from typing import List, Tuple
import numpy as np

import read_conditions
import envelope
import read_load
import appendix
from appendix import SystemSpec


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


def get_minimum_air_volume(zone_floor_area: envelope.FloorArea) -> (float, float):
    """
    Args:
        zone_floor_area: floor area of each zones, m2
    Returns:
        minimum supply air volume of the system for heating and cooling
    """

    htg = np.sum(get_mechanical_ventilation(zone_floor_area))
    clg = np.sum(get_mechanical_ventilation(zone_floor_area))

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
        region: int, floor_area: envelope.FloorArea, envelope_spec: envelope.Spec) -> np.ndarray:
    """calculate the system supply air volume for heating
    eq.(12)
    Args:
        region: region 1-8
        floor_area: floor area
        envelope_spec: envelop spec
    Returns:
        heating output for supply air estimation, MJ/h
    """

    # heating load in the main occupant room and the other occupant rooms, MJ/h
    l_h = read_load.get_heating_load(region=region, envelope_spec=envelope_spec, floor_area=floor_area)[0:5]

    # q value, W/m2K
    q = envelope_spec.get_q_value(region=region)

    # air conditioned temperature for heating, degree C
    theta_ac_h = np.full(8760, get_air_conditioned_temperature_for_heating())

    # outdoor temperature, degree C
    theta_ex = read_conditions.read_temperature(region=region)

    # mu value, (W/m2)/(W/m2)
    mu_h = envelope_spec.get_mu_h_value(region=region)

    # horizontal solar radiation, W/m2
    j = read_conditions.get_horizontal_solar(region)

    # floor area of non occupant room, m2
    a_nr = floor_area.nor

    q_dash_hs_h = np.sum(l_h, axis=0) + ((theta_ac_h - theta_ex) * q - j * mu_h) * a_nr * 3600 * 10 ** (-6)

    # This operation is not described in the specification document
    # The supply air has lower limitation. This operation does not eventually effect the result.
    return np.vectorize(lambda x: x if x > 0.0 else 0.0)(q_dash_hs_h)


def get_cooling_output_for_supply_air_estimation(
        region: int, floor_area: envelope.FloorArea, envelope_spec: envelope.Spec) -> np.ndarray:
    """calculate the system supply air volume for cooling
    eq.(27)
    Args:
        region: region 1-8
        floor_area: floor area
        envelope_spec: envelop spec
    Returns:
        sensible and latent cooling output for supply air estimation, MJ/h
    """
    # sensible cooling load in the main occupant room and the other occupant rooms, MJ/h
    l_cs = read_load.get_sensible_cooling_load(region=region, envelope_spec=envelope_spec, floor_area=floor_area)[0:5]

    # latent cooling load in the main occupant room and the other occupant rooms, MJ/h
    l_cl = read_load.get_latent_cooling_load(region=region, envelope_spec=envelope_spec, floor_area=floor_area)[0:5]

    # q value, W/m2K
    q = envelope_spec.get_q_value(region=region)

    # air conditioned temperature for cooling, degree C
    theta_ac_c = np.full(8760, get_air_conditioned_temperature_for_cooling())

    # outdoor temperature, degree C
    theta_ex = read_conditions.read_temperature(region=region)

    # mu value, (W/m2)/(W/m2)
    mu_c = envelope_spec.get_mu_c_value(region=region)

    # horizontal solar radiation, W/m2
    j = read_conditions.get_horizontal_solar(region)

    # floor area of non occupant room, m2
    a_nr = floor_area.nor

    q_dash_hs_c = np.sum(l_cs, axis=0) + np.sum(l_cl, axis=0) \
                  + ((theta_ex - theta_ac_c) * q + j * mu_c) * a_nr * 3600 * 10 ** (-6)

    # This operation is not described in the specification document
    # The supply air has lower limitation. This operation does not eventually effect the result.
    return np.vectorize(lambda x: x if x > 0.0 else 0.0)(q_dash_hs_c)


def get_heat_source_supply_air_volume_for_heating(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    calculate the supply air volume for heating
    Args:
        region: region
        floor_area: floor_area
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        supply air volume, m3/h (8760 times)
    """

    # heating output of the system for estimation of the supply air volume, MJ/h
    q_dash_hs_h = get_heating_output_for_supply_air_estimation(
        region=region, floor_area=floor_area, envelope_spec=envelope_spec)

    # rated heating output, MJ/h
    q_hs_rtd_h = get_rated_heating_output(system_spec=system_spec)

    # minimum supply air volume, m3/h
    v_hs_min_h = get_minimum_air_volume(zone_floor_area=floor_area)[0]

    # rated (maximum) supply air volume, m3/h
    v_hs_rtd_h = system_spec.supply_air_rtd_h

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
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    calculate the supply air volume for cooling
    Args:
        region: region
        floor_area: floor_area
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        supply air volume, m3/h (8760 times)
    """

    # cooling output of the system for estimation of the supply air volume, MJ/h
    q_dash_hs_c = get_cooling_output_for_supply_air_estimation(
        region=region, floor_area=floor_area, envelope_spec=envelope_spec)

    # rated cooling output, MJ/h
    q_hs_rtd_c = get_rated_cooling_output(system_spec=system_spec)

    # minimum supply air volume, m3/h
    v_hs_min_c = get_minimum_air_volume(zone_floor_area=floor_area)[1]

    # rated (maximum) supply air volume, m3/h
    v_hs_rtd_c = system_spec.supply_air_rtd_c

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
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        supply air volume, m3/h (5 rooms * 8760 times)
    """

    # supply air volume valance
    r_supply_des = get_supply_air_volume_valance(floor_area).reshape(1, 5).T

    # total supply air volume for heating, m3/h
    v_hs_supply_h = get_heat_source_supply_air_volume_for_heating(
        region=region, floor_area=floor_area, envelope_spec=envelope_spec, system_spec=system_spec)

    # mechanical ventilation, m3/h (5 rooms * 1 value)
    v_vent = get_mechanical_ventilation(floor_area).reshape(1, 5).T

    return np.maximum(v_hs_supply_h * r_supply_des, v_vent)


def get_each_supply_air_volume_for_cooling(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        supply air volume, m3/h (5 rooms * 8760 times)
    """

    # supply air volume valance
    r_supply_des = get_supply_air_volume_valance(floor_area).reshape(1, 5).T

    # total supply air volume for heating, m3/h
    v_hs_supply_c = get_heat_source_supply_air_volume_for_cooling(
        region=region, floor_area=floor_area, envelope_spec=envelope_spec, system_spec=system_spec)

    # mechanical ventilation, m3/h (5 rooms * 1 value)
    v_vent = get_mechanical_ventilation(floor_area).reshape(1, 5).T

    return np.maximum(v_hs_supply_c * r_supply_des, v_vent)


def get_non_occupant_room_temperature_for_heating(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        non occupant room temperature, degree C (8760 times)
    """

    # Q value, W/m2K
    q_value = envelope_spec.get_q_value(region=region)

    # outdoor temperature, degree C
    theta_ex = read_conditions.read_temperature(region=region)

    # mu value, (W/m2K)/(W/m2K)
    mu_value = envelope_spec.get_mu_h_value(region=region)

    # horizontal solar radiation, W/m2K
    j = read_conditions.get_horizontal_solar(region=region)

    # floor area of non occupant room, m2
    a_nr = floor_area.nor

    # specific heat of air, J/kg K
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # supply air volume, m3/h
    v_supply_h = get_each_supply_air_volume_for_heating(
        region=region, floor_area=floor_area, envelope_spec=envelope_spec, system_spec=system_spec)

    # heat loss coefficient of the partition wall, W/m2K
    u_prt = get_heat_loss_coefficient_of_partition()

    # area of the partition, m2
    a_prt = get_partition_area(floor_area=floor_area).reshape(1, 5).T

    # air conditioned temperature for heating, degree C
    theta_ac_h = get_air_conditioned_temperature_for_heating()

    return ((q_value * theta_ex + mu_value * j) * a_nr
            + np.sum(c * rho * v_supply_h / 3600 + u_prt * a_prt, axis=0) * theta_ac_h) \
           / (q_value * a_nr + np.sum(c * rho * v_supply_h / 3600 + u_prt * a_prt, axis=0))


def get_non_occupant_room_temperature_for_cooling(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        non occupant room temperature, degree C (8760 times)
    """

    # Q value, W/m2K
    q_value = envelope_spec.get_q_value(region=region)

    # outdoor temperature, degree C
    theta_ex = read_conditions.read_temperature(region=region)

    # mu value, (W/m2K)/(W/m2K)
    mu_value = envelope_spec.get_mu_c_value(region=region)

    # horizontal solar radiation, W/m2K
    j = read_conditions.get_horizontal_solar(region=region)

    # floor area of non occupant room, m2
    a_nr = floor_area.nor

    # specific heat of air, J/kg K
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # supply air volume, m3/h
    v_supply_c = get_each_supply_air_volume_for_cooling(
        region=region, floor_area=floor_area, envelope_spec=envelope_spec, system_spec=system_spec)

    # heat loss coefficient of the partition wall, W/m2K
    u_prt = get_heat_loss_coefficient_of_partition()

    # area of the partition, m2
    a_prt = get_partition_area(floor_area=floor_area).reshape(1, 5).T

    # air conditioned temperature for heating, degree C
    theta_ac_c = get_air_conditioned_temperature_for_cooling()

    return ((q_value * theta_ex + mu_value * j) * a_nr
            + np.sum(c * rho * v_supply_c / 3600 + u_prt * a_prt, axis=0) * theta_ac_c) \
           / (q_value * a_nr + np.sum(c * rho * v_supply_c / 3600 + u_prt * a_prt, axis=0))


def get_heat_loss_through_partition_for_heating(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        heat loss through the partition, MJ/h (5 rooms * 8760 times)
    """

    # heat loss coefficient of the partition wall, W/m2K
    u_prt = get_heat_loss_coefficient_of_partition()

    # area of the partition, m2
    a_prt = get_partition_area(floor_area=floor_area).reshape(1, 5).T

    # air conditioned temperature for heating, degree C
    theta_ac_h = get_air_conditioned_temperature_for_heating()

    # non occupant room temperature, degree C (8760 times)
    theta_nac_h = get_non_occupant_room_temperature_for_heating(
        region=region, floor_area=floor_area, envelope_spec=envelope_spec, system_spec=system_spec)

    return u_prt * a_prt * (theta_ac_h - theta_nac_h) * 3600 * 10 ** (-6)


def get_heat_gain_through_partition_for_cooling(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        heat gain through the partition, MJ/h (5 rooms * 8760 times)
    """

    # heat loss coefficient of the partition wall, W/m2K
    u_prt = get_heat_loss_coefficient_of_partition()

    # area of the partition, m2
    a_prt = get_partition_area(floor_area=floor_area).reshape(1, 5).T

    # air conditioned temperature for heating, degree C
    theta_ac_c = get_air_conditioned_temperature_for_cooling()

    # non occupant room temperature, degree C (8760 times)
    theta_nac_c = get_non_occupant_room_temperature_for_cooling(
        region=region, floor_area=floor_area, envelope_spec=envelope_spec, system_spec=system_spec)

    return u_prt * a_prt * (theta_nac_c - theta_ac_c) * 3600 * 10 ** (-6)


def get_maximum_output_for_heating(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    calculate maximum output for heating
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        maximum output for heating, MJ/h, (5 rooms * 8760 times)
    """

    # inlet air temperature of the heat source for heating, degree C (8760 times)
    theta_hs_in_h = get_non_occupant_room_temperature_for_heating(region, floor_area, envelope_spec, system_spec)

    # maximum heating output, MJ/h (8760 times)
    q_hs_max_h = appendix.get_maximum_heating_output(region, system_spec)

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # supply air volume for heating, m3/h (5 rooms * 8760 times)
    v_supply_h = get_each_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)

    # maximum outlet air temperature of heat source, degree C, (8760 times)
    theta_hs_out_max_h = theta_hs_in_h + q_hs_max_h / (c * rho * np.sum(v_supply_h, axis=0)) * 10 ** 6

    # air conditioned temperature for heating, degree C
    theta_ac_h = get_air_conditioned_temperature_for_heating()

    # linear heat loss coefficient of the duct, W/mK
    psi = get_duct_linear_heat_loss_coefficient()

    # duct length, m
    l_duct = np.array(calc_duct_length(floor_area.total)).reshape(1, 5).T

    # ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
    theta_sur_h, theta_sur_c = calc_duct_ambient_air_temperature(floor_area.total, region, system_spec)

    #    return ((theta_hs_out_max_h - theta_ac_h) * c * rho * v_supply_h
    #            - psi * l_duct * (theta_hs_out_max_h - theta_sur_h) * 3600) * 10**(-6)

    # return (theta_sur_h - theta_ac_h
    #         + (theta_hs_out_max_h - theta_sur_h) / np.exp(psi * l_duct * 3600 / (c * rho * v_supply_h)))\
    #     * c * rho * v_supply_h * 10**(-6)

    return get_load_from_upside_temperature(
        t_sur=theta_sur_h, t_up=theta_hs_out_max_h, v=v_supply_h, t_ac=theta_ac_h, psi=psi, length=l_duct)


def get_maximum_output_for_cooling(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    calculate maximum output for cooling
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        maximum output for sensible cooling, MJ/h, (5 rooms * 8760 times), maximum output for latent cooling, MJ/h, (5 rooms * 8760 times)
    """

    # inlet air temperature of the heat source for cooling, degree C (8760 times)
    theta_hs_in_c = get_non_occupant_room_temperature_for_cooling(region, floor_area, envelope_spec, system_spec)

    # sensible cooling load, MJ/h (5 rooms * 8760 times)
    l_cs = read_load.get_sensible_cooling_load(region, envelope_spec, floor_area)[0:5]

    # latent cooling load, MJ/h (5 rooms * 8760 times)
    l_cl = read_load.get_latent_cooling_load(region, envelope_spec, floor_area)[0:5]

    # heat gain from non occupant room into occupant room through partition for cooling, MJ/h (5 rooms * 8760 times)
    q_trs_part_c = get_heat_gain_through_partition_for_cooling(region, floor_area, envelope_spec, system_spec)

    # maximum sensible cooling output, MJ/h (8760 times), maximum latent cooling output, MJ/h (8760 times)
    q_hs_max_cs, q_hs_max_cl = appendix.get_maximum_cooling_output(system_spec, l_cs, q_trs_part_c, l_cl)

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # supply air volume for cooling, m3/h (5 rooms * 8760 times)
    v_supply_c = get_each_supply_air_volume_for_cooling(region, floor_area, envelope_spec, system_spec)

    # minimum outlet air temperature of heat source, degree C, (8760 times)
    theta_hs_out_min_c = theta_hs_in_c - q_hs_max_cs / (c * rho * np.sum(v_supply_c, axis=0)) * 10 ** 6

    # air conditioned temperature for cooling, degree C
    theta_ac_c = get_air_conditioned_temperature_for_cooling()

    # linear heat loss coefficient of the duct, W/mK
    psi = get_duct_linear_heat_loss_coefficient()

    # duct length, m
    l_duct = np.array(calc_duct_length(floor_area.total)).reshape(1, 5).T

    # ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
    theta_sur_h, theta_sur_c = calc_duct_ambient_air_temperature(floor_area.total, region, system_spec)

    #    q_max_cs = ((theta_ac_c - theta_hs_out_min_c) * c * rho * v_supply_c
    #                - psi * l_duct * (theta_sur_c - theta_hs_out_min_c) * 3600) * 10**(-6)

    q_max_cs = (theta_ac_c - theta_sur_c
                + (theta_sur_c - theta_hs_out_min_c) / np.exp(psi * l_duct * 3600 / (c * rho * v_supply_c))) \
               * c * rho * v_supply_c * 10 ** (-6)

    l_cl_sum = np.sum(l_cl, axis=0)

    r = np.vectorize(lambda x, y: x / y if y > 0.0 else 0.0)(l_cl, l_cl_sum)

    q_max_cl = r * q_hs_max_cl

    return q_max_cs, q_max_cl


def get_treated_untreated_heat_load_for_heating(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        (a,b)
            a: treated heating load, MJ/h, (5 rooms * 8760 times)
            b: untreated heating load, MJ/h, (5 rooms * 8760 times)
    """

    # heating load, MJ/h (5 rooms * 8760 times)
    l_h = read_load.get_heating_load(region, envelope_spec, floor_area)[0:5]

    # heat loss from the occupant room into the non occupant room through the partition, MJ/h (5 rooms * 8760 times)
    q_trs_prt_h = get_heat_loss_through_partition_for_heating(region, floor_area, envelope_spec, system_spec)

    # maximum output for heating, MJ/h
    q_max_h = get_maximum_output_for_heating(region, floor_area, envelope_spec, system_spec)

    # treated load, MJ/h
    #    q_t_h = np.minimum(q_max_h, np.maximum(l_h + q_trs_prt_h, 0.0))
    q_t_h = np.clip(l_h + q_trs_prt_h, 0.0, q_max_h)

    # untreated load, MJ/h
    q_ut_h = np.maximum(l_h + q_trs_prt_h, 0.0) - q_t_h

    return q_t_h, q_ut_h


def get_treated_untreated_heat_load_for_cooling(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        (a,b,c,d)
            a: treated sensible heating load, MJ/h, (5 rooms * 8760 times)
            b: treated latent heating load, MJ/h, (5 rooms * 8760 times)
            c: untreated sensible heating load, MJ/h, (5 rooms * 8760 times)
            d: untreated latent heating load, MJ/h, (5 rooms * 8760 times)
    """

    # sensible cooling load, MJ/h (5 rooms * 8760 times)
    l_cs = read_load.get_sensible_cooling_load(region, envelope_spec, floor_area)[0:5]

    # latent cooling load, MJ/h (5 rooms, 8760 times)
    l_cl = read_load.get_latent_cooling_load(region, envelope_spec, floor_area)[0:5]

    # heat gain from the non occupant room into the occupant room through the partition, MJ/h (5 rooms * 8760 times)
    q_trs_prt_c = get_heat_gain_through_partition_for_cooling(region, floor_area, envelope_spec, system_spec)

    # maximum output for sensible cooling, MJ/h, (5 rooms * 8760 times)
    # maximum output for latent cooling, MJ/h, (5 rooms * 8760 times),
    q_max_cs, q_max_cl = get_maximum_output_for_cooling(region, floor_area, envelope_spec, system_spec)

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
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    calculate the requested supply air temperature for heating
    Args
        region: region
        floor_area: floor area
        envelope_spec: envelope spec
        system_spec: system spec
    Returns
        requested temperature, degree C, (5 rooms * 8760 times)
    """

    # ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
    theta_sur_h, theta_sur_c = calc_duct_ambient_air_temperature(floor_area.total, region, system_spec)

    # air conditioned temperature for heating, degree C
    theta_ac_h = get_air_conditioned_temperature_for_heating()

    # get treated heat load for heating, MJ/h, (5 rooms * 8760 times), ---(not used)
    q_t_h, q_ut_h = get_treated_untreated_heat_load_for_heating(region, floor_area, envelope_spec, system_spec)

    # supply air volume for heating, m3/h (5 rooms * 8760 times)
    v_supply_h = get_each_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # linear heat loss coefficient of the duct, W/mK
    psi = get_duct_linear_heat_loss_coefficient()

    # duct length, m
    l_duct = np.array(calc_duct_length(floor_area.total)).reshape(1, 5).T

    return theta_sur_h + (theta_ac_h + q_t_h * 10 ** 6 / (v_supply_h * c * rho) - theta_sur_h) \
        * np.exp(psi * l_duct * 3600 / (v_supply_h * c * rho))


def get_requested_supply_air_temperature_for_cooling(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    calculate the requested supply air temperature for heating
    Args
        region: region
        floor_area: floor area
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        requested temperature, degree C, (5 rooms * 8760 times)
    """

    # ambient temperature around the ducts, degree C, (5 rooms * 8760 times)
    theta_sur_h, theta_sur_c = calc_duct_ambient_air_temperature(floor_area.total, region, system_spec)

    # air conditioned temperature for cooling, degree C
    theta_ac_c = get_air_conditioned_temperature_for_cooling()

    # get treated heat load for cooling, MJ/h, (5 rooms * 8760 times), ---(not used)
    q_t_cs, q_ut_cs, q_ut_cl, q_ut_cl \
        = get_treated_untreated_heat_load_for_cooling(region, floor_area, envelope_spec, system_spec)

    # supply air volume for heating, m3/h (5 rooms * 8760 times)
    v_supply_c = get_each_supply_air_volume_for_cooling(region, floor_area, envelope_spec, system_spec)

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # linear heat loss coefficient of the duct, W/mK
    psi = get_duct_linear_heat_loss_coefficient()

    # duct length, m
    l_duct = np.array(calc_duct_length(floor_area.total)).reshape(1, 5).T

    return theta_sur_c - (theta_sur_c - theta_ac_c + q_t_cs * 10 ** 6 / (v_supply_c * c * rho)) \
        * np.exp(psi * l_duct * 3600 / (v_supply_c * c * rho))


def calc_decided_outlet_supply_air_temperature_for_heating(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    decide the outlet supply air temperature for heating
    Args:
        region: region
        floor_area: floor area
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        decided outlet supply air temperature, degree C, (8760 times)
    """

    theta_duct_up_h = get_requested_supply_air_temperature_for_heating(region, floor_area, envelope_spec, system_spec)
    return np.max(theta_duct_up_h, axis=0)


def calc_decided_outlet_supply_air_temperature_for_cooling(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    decide the outlet supply air temperature for cooling
    Args:
        region: region
        floor_area: floor area
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        decided outlet supply air temperature, degree C, (8760 times)
    """

    theta_duct_up_c = get_requested_supply_air_temperature_for_cooling(region, floor_area, envelope_spec, system_spec)
    return np.min(theta_duct_up_c, axis=0)


def calc_heat_source_heating_output(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    calculate heat source heating output
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    Returns:
        heating output, MJ/h, (8760 times)
    """

    # supply air temperature, degree C
    theta_hs_out_h = calc_decided_outlet_supply_air_temperature_for_heating(
        region, floor_area, envelope_spec, system_spec)

    # inlet air temperature of the heat source for heating, degree C (8760 times)
    theta_hs_in_h = get_non_occupant_room_temperature_for_heating(region, floor_area, envelope_spec, system_spec)

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # supply air volume for heating, m3/h (5 rooms * 8760 times)
    v_supply_h = get_each_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)

    return np.maximum((theta_hs_out_h - theta_hs_in_h) * c * rho * np.sum(v_supply_h, axis=0) * 10 ** (-6), 0.0)


def calc_heat_source_cooling_output(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    """

    # inlet air temperature of the heat source for cooling, degree C (8760 times)
    theta_hs_in_c = get_non_occupant_room_temperature_for_cooling(region, floor_area, envelope_spec, system_spec)

    # supply air temperature, degree C
    theta_hs_out_c = calc_decided_outlet_supply_air_temperature_for_cooling(
        region, floor_area, envelope_spec, system_spec)

    # specific heat of air, J/kgK
    c = get_specific_heat()

    # air density, kg/m3
    rho = get_air_density()

    # supply air volume for cooling, m3/h (5 rooms * 8760 times)
    v_supply_c = get_each_supply_air_volume_for_cooling(region, floor_area, envelope_spec, system_spec)

    # latent cooling load, MJ/h
    l_cl = read_load.get_latent_cooling_load(region, envelope_spec, floor_area)

    q_hs_cs = np.maximum((theta_hs_in_c - theta_hs_out_c) * c * rho * np.sum(v_supply_c, axis=0) * 10 ** (-6), 0.0)

    q_hs_cl = np.sum(l_cl[0:5], axis=0)

    return q_hs_cs, q_hs_cl


def get_duct_heat_loss_for_heating(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    """

    # duct ambient temperature, degree C
    theta_sur_h, theta_sur_c = calc_duct_ambient_air_temperature(floor_area.total, region, system_spec)

    # supply air temperature, degree C
    theta_hs_out_h = calc_decided_outlet_supply_air_temperature_for_heating(
        region, floor_area, envelope_spec, system_spec)

    # supply air volume for heating, m3/h (5 rooms * 8760 times)
    v_supply_h = get_each_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)

    # air conditioned temperature, degree C
    t_ac = get_air_conditioned_temperature_for_heating()

    psi = get_duct_linear_heat_loss_coefficient()

    # duct length, m
    l_duct = np.array(calc_duct_length(floor_area.total)).reshape(1, 5).T

    return get_duct_heat_loss_from_upside_temperature(
        theta_sur_h, theta_hs_out_h, v_supply_h, t_ac, psi, l_duct)


def get_actual_treated_load_for_heating(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec, system_spec: SystemSpec) -> np.ndarray:
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    """

    # duct ambient temperature, degree C
    theta_sur_h, theta_sur_c = calc_duct_ambient_air_temperature(floor_area.total, region, system_spec)

    # supply air temperature, degree C
    theta_hs_out_h = calc_decided_outlet_supply_air_temperature_for_heating(
        region, floor_area, envelope_spec, system_spec)

    # supply air volume for heating, m3/h (5 rooms * 8760 times)
    v_supply_h = get_each_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)

    # air conditioned temperature, degree C
    t_ac = get_air_conditioned_temperature_for_heating()

    psi = get_duct_linear_heat_loss_coefficient()

    # duct length, m
    l_duct = np.array(calc_duct_length(floor_area.total)).reshape(1, 5).T

    return get_load_from_upside_temperature(
        theta_sur_h, theta_hs_out_h, v_supply_h, t_ac, psi, l_duct)


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


def get_main_value(
        region: int, floor_area: envelope.FloorArea,
        envelope_spec: envelope.Spec,
        system_spec: SystemSpec):
    """
    Args:
        region: region
        floor_area: floor area class
        envelope_spec: envelope spec
        system_spec: system spec
    """

    # air density, kg/m3
    rho = get_air_density()

    # air specific heat, J/kg K
    c = get_specific_heat()

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

    v_hs_min_h, v_hs_min_c = get_minimum_air_volume(floor_area)

    a_part = get_partition_area(floor_area)

    q_hs_rtd_h = get_rated_heating_output(system_spec)
    q_hs_rtd_c = get_rated_cooling_output(system_spec)

    q_d_hs_h = get_heating_output_for_supply_air_estimation(region, floor_area, envelope_spec)
    q_d_hs_c = get_cooling_output_for_supply_air_estimation(region, floor_area, envelope_spec)

    v_hs_supply_h = get_heat_source_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)
    v_hs_supply_c = get_heat_source_supply_air_volume_for_cooling(region, floor_area, envelope_spec, system_spec)

    v_supply_h = get_each_supply_air_volume_for_heating(region, floor_area, envelope_spec, system_spec)
    v_supply_c = get_each_supply_air_volume_for_cooling(region, floor_area, envelope_spec, system_spec)

    theta_nac_h = get_non_occupant_room_temperature_for_heating(region, floor_area, envelope_spec, system_spec)
    theta_nac_c = get_non_occupant_room_temperature_for_cooling(region, floor_area, envelope_spec, system_spec)

    q_trs_prt_h = get_heat_loss_through_partition_for_heating(region, floor_area, envelope_spec, system_spec)
    q_trs_prt_c = get_heat_gain_through_partition_for_cooling(region, floor_area, envelope_spec, system_spec)

    q_max_h = get_maximum_output_for_heating(region, floor_area, envelope_spec, system_spec)
    q_max_cs, q_max_cl = get_maximum_output_for_cooling(region, floor_area, envelope_spec, system_spec)

    q_t_h, q_ut_h = get_treated_untreated_heat_load_for_heating(region, floor_area, envelope_spec, system_spec)
    q_t_cs, q_t_cl, q_ut_cs, q_ut_cl = get_treated_untreated_heat_load_for_cooling(region, floor_area, envelope_spec, system_spec)

    theta_duct_up_h = get_requested_supply_air_temperature_for_heating(region, floor_area, envelope_spec, system_spec)
    theta_duct_up_c = get_requested_supply_air_temperature_for_cooling(region, floor_area, envelope_spec, system_spec)

    # outlet temperature of heat source, degree C, (8760 times)
    theta_hs_out_h = calc_decided_outlet_supply_air_temperature_for_heating(region, floor_area, envelope_spec, system_spec)
    theta_hs_out_c = calc_decided_outlet_supply_air_temperature_for_cooling(region, floor_area, envelope_spec, system_spec)

    # output of heat source, MJ/h, (8760 times)
    q_hs_h = calc_heat_source_heating_output(region, floor_area, envelope_spec, system_spec)
    q_hs_cs, q_hs_cl = calc_heat_source_cooling_output(region, floor_area, envelope_spec, system_spec)

    q_loss_duct_h = get_duct_heat_loss_for_heating(region, floor_area, envelope_spec, system_spec)
    q_act_h = get_actual_treated_load_for_heating(region, floor_area, envelope_spec, system_spec)

    constant_value = {
        'air_density_kg/m3': rho,
        'air_specific_heat_J/kgK': c,
        'duct_length_room1_m': l_duct_i[0],
        'duct_length_room2_m': l_duct_i[1],
        'duct_length_room3_m': l_duct_i[2],
        'duct_length_room4_m': l_duct_i[3],
        'duct_length_room5_m': l_duct_i[4],
        'supply_air_valance_room1': r_supply_des[0],
        'supply_air_valance_room2': r_supply_des[1],
        'supply_air_valance_room3': r_supply_des[2],
        'supply_air_valance_room4': r_supply_des[3],
        'supply_air_valance_room5': r_supply_des[4],
        'mechanical_ventilation_volume_room1_m3/h': v_vent[0],
        'mechanical_ventilation_volume_room2_m3/h': v_vent[1],
        'mechanical_ventilation_volume_room3_m3/h': v_vent[2],
        'mechanical_ventilation_volume_room4_m3/h': v_vent[3],
        'mechanical_ventilation_volume_room5_m3/h': v_vent[4],
        'minimum_supply_air_volume_of_heat_source_heating_m3/h': v_hs_min_h,
        'minimum_supply_air_volume_of_heat_source_cooling_m3/h': v_hs_min_c,
        'partition_area_room1_m2': a_part[0],
        'partition_area_room2_m2': a_part[1],
        'partition_area_room3_m2': a_part[2],
        'partition_area_room4_m2': a_part[3],
        'partition_area_room5_m2': a_part[4],
        'rated_capacity_heating_MJ/h': q_hs_rtd_h,
        'rated_capacity_cooling_MJ/h': q_hs_rtd_c,
    }

    time_value = {
        'heating_load_room1_MJ/h': l_h[0],
        'heating_load_room2_MJ/h': l_h[1],
        'heating_load_room3_MJ/h': l_h[2],
        'heating_load_room4_MJ/h': l_h[3],
        'heating_load_room5_MJ/h': l_h[4],
        'old_heating_load_sum_of_12_rooms_MJ/h': np.sum(l_h, axis=0),
        'sensible_cooling_load_room1_MJ/h': l_cs[0],
        'sensible_cooling_load_room2_MJ/h': l_cs[1],
        'sensible_cooling_load_room3_MJ/h': l_cs[2],
        'sensible_cooling_load_room4_MJ/h': l_cs[3],
        'sensible_cooling_load_room5_MJ/h': l_cs[4],
        'old_sensible_cooling_load_sum_of_12_rooms_MJ/h': np.sum(l_cs, axis=0),
        'latent_cooling_load_room1_MJ/h': l_cl[0],
        'latent_cooling_load_room2_MJ/h': l_cl[1],
        'latent_cooling_load_room3_MJ/h': l_cl[2],
        'latent_cooling_load_room4_MJ/h': l_cl[3],
        'latent_cooling_load_room5_MJ/h': l_cl[4],
        'old_latent_cooling_load_sum_of_12_rooms_MJ/h': np.sum(l_cl, axis=0),
        'air_conditioned_temperature_heating_degree_C': theta_ac_h,
        'air_conditioned_temperature_cooling_degree_C': theta_ac_c,
        'sat_temperature_degree_C': theta_sat,
        'attic_temperature_heating_degree_C': theta_attic_h,
        'attic_temperature_cooling_degree_C': theta_attic_c,
        'duct_ambient_temperature_heating_room1_degree_C': theta_sur_h[0],
        'duct_ambient_temperature_heating_room2_degree_C': theta_sur_h[1],
        'duct_ambient_temperature_heating_room3_degree_C': theta_sur_h[2],
        'duct_ambient_temperature_heating_room4_degree_C': theta_sur_h[3],
        'duct_ambient_temperature_heating_room5_degree_C': theta_sur_h[4],
        'duct_ambient_temperature_cooling_room1_degree_C': theta_sur_c[0],
        'duct_ambient_temperature_cooling_room2_degree_C': theta_sur_c[1],
        'duct_ambient_temperature_cooling_room3_degree_C': theta_sur_c[2],
        'duct_ambient_temperature_cooling_room4_degree_C': theta_sur_c[3],
        'duct_ambient_temperature_cooling_room5_degree_C': theta_sur_c[4],
        'output_of_heat_source_for_supply_air_volume_estimation_heating_MJ/h': q_d_hs_h,
        'output_of_heat_source_for_supply_air_volume_estimation_cooling_MJ/h': q_d_hs_c,
        'supply_air_volume_of_heat_source_heating_m3/h': v_hs_supply_h,
        'supply_air_volume_of_heat_source_cooling_m3/h': v_hs_supply_c,
        'supply_air_volume_heating_room1_m3/h': v_supply_h[0],
        'supply_air_volume_heating_room2_m3/h': v_supply_h[1],
        'supply_air_volume_heating_room3_m3/h': v_supply_h[2],
        'supply_air_volume_heating_room4_m3/h': v_supply_h[3],
        'supply_air_volume_heating_room5_m3/h': v_supply_h[4],
        'supply_air_volume_cooling_room1_m3/h': v_supply_c[0],
        'supply_air_volume_cooling_room2_m3/h': v_supply_c[1],
        'supply_air_volume_cooling_room3_m3/h': v_supply_c[2],
        'supply_air_volume_cooling_room4_m3/h': v_supply_c[3],
        'supply_air_volume_cooling_room5_m3/h': v_supply_c[4],
        'non_occupant_room_temperature_heating_degree_C': theta_nac_h,
        'non_occupant_room_temperature_cooling_degree_C': theta_nac_c,
        'heat_loss_through_partition_heating_room1_MJ/h': q_trs_prt_h[0],
        'heat_loss_through_partition_heating_room2_MJ/h': q_trs_prt_h[1],
        'heat_loss_through_partition_heating_room3_MJ/h': q_trs_prt_h[2],
        'heat_loss_through_partition_heating_room4_MJ/h': q_trs_prt_h[3],
        'heat_loss_through_partition_heating_room5_MJ/h': q_trs_prt_h[4],
        'heat_gain_through_partition_cooling_room1_MJ/h': q_trs_prt_c[0],
        'heat_gain_through_partition_cooling_room2_MJ/h': q_trs_prt_c[1],
        'heat_gain_through_partition_cooling_room3_MJ/h': q_trs_prt_c[2],
        'heat_gain_through_partition_cooling_room4_MJ/h': q_trs_prt_c[3],
        'heat_gain_through_partition_cooling_room5_MJ/h': q_trs_prt_c[4],
        'maximum_output_heating_room1_MJ/h': q_max_h[0],
        'maximum_output_heating_room2_MJ/h': q_max_h[1],
        'maximum_output_heating_room3_MJ/h': q_max_h[2],
        'maximum_output_heating_room4_MJ/h': q_max_h[3],
        'maximum_output_heating_room5_MJ/h': q_max_h[4],
        'maximum_output_sensible_cooling_room1_MJ/h': q_max_cs[0],
        'maximum_output_sensible_cooling_room2_MJ/h': q_max_cs[1],
        'maximum_output_sensible_cooling_room3_MJ/h': q_max_cs[2],
        'maximum_output_sensible_cooling_room4_MJ/h': q_max_cs[3],
        'maximum_output_sensible_cooling_room5_MJ/h': q_max_cs[4],
        'maximum_output_latent_cooling_room1_MJ/h': q_max_cl[0],
        'maximum_output_latent_cooling_room2_MJ/h': q_max_cl[1],
        'maximum_output_latent_cooling_room3_MJ/h': q_max_cl[2],
        'maximum_output_latent_cooling_room4_MJ/h': q_max_cl[3],
        'maximum_output_latent_cooling_room5_MJ/h': q_max_cl[4],
        'treated_heating_load_room1_MJ/h': q_t_h[0],
        'treated_heating_load_room2_MJ/h': q_t_h[1],
        'treated_heating_load_room3_MJ/h': q_t_h[2],
        'treated_heating_load_room4_MJ/h': q_t_h[3],
        'treated_heating_load_room5_MJ/h': q_t_h[4],
        'treated_sensible_cooling_load_room1_MJ/h': q_t_cs[0],
        'treated_sensible_cooling_load_room2_MJ/h': q_t_cs[1],
        'treated_sensible_cooling_load_room3_MJ/h': q_t_cs[2],
        'treated_sensible_cooling_load_room4_MJ/h': q_t_cs[3],
        'treated_sensible_cooling_load_room5_MJ/h': q_t_cs[4],
        'treated_latent_cooling_load_room1_MJ/h': q_t_cl[0],
        'treated_latent_cooling_load_room2_MJ/h': q_t_cl[1],
        'treated_latent_cooling_load_room3_MJ/h': q_t_cl[2],
        'treated_latent_cooling_load_room4_MJ/h': q_t_cl[3],
        'treated_latent_cooling_load_room5_MJ/h': q_t_cl[4],
        'untreated_heating_load_room1_MJ/h': q_ut_h[0],
        'untreated_heating_load_room2_MJ/h': q_ut_h[1],
        'untreated_heating_load_room3_MJ/h': q_ut_h[2],
        'untreated_heating_load_room4_MJ/h': q_ut_h[3],
        'untreated_heating_load_room5_MJ/h': q_ut_h[4],
        'untreated_sensible_cooling_load_room1_MJ/h': q_ut_cs[0],
        'untreated_sensible_cooling_load_room2_MJ/h': q_ut_cs[1],
        'untreated_sensible_cooling_load_room3_MJ/h': q_ut_cs[2],
        'untreated_sensible_cooling_load_room4_MJ/h': q_ut_cs[3],
        'untreated_sensible_cooling_load_room5_MJ/h': q_ut_cs[4],
        'untreated_latent_cooling_load_room1_MJ/h': q_ut_cl[0],
        'untreated_latent_cooling_load_room2_MJ/h': q_ut_cl[1],
        'untreated_latent_cooling_load_room3_MJ/h': q_ut_cl[2],
        'untreated_latent_cooling_load_room4_MJ/h': q_ut_cl[3],
        'untreated_latent_cooling_load_room5_MJ/h': q_ut_cl[4],
        'duct_upside_supply_air_temperature_heating_room1_degree_C': theta_duct_up_h[0],
        'duct_upside_supply_air_temperature_heating_room2_degree_C': theta_duct_up_h[1],
        'duct_upside_supply_air_temperature_heating_room3_degree_C': theta_duct_up_h[2],
        'duct_upside_supply_air_temperature_heating_room4_degree_C': theta_duct_up_h[3],
        'duct_upside_supply_air_temperature_heating_room5_degree_C': theta_duct_up_h[4],
        'duct_upside_supply_air_temperature_cooling_room1_degree_C': theta_duct_up_c[0],
        'duct_upside_supply_air_temperature_cooling_room2_degree_C': theta_duct_up_c[1],
        'duct_upside_supply_air_temperature_cooling_room3_degree_C': theta_duct_up_c[2],
        'duct_upside_supply_air_temperature_cooling_room4_degree_C': theta_duct_up_c[3],
        'duct_upside_supply_air_temperature_cooling_room5_degree_C': theta_duct_up_c[4],
        'outlet_temperature_of_heat_source_heating_degree_C': theta_hs_out_h,
        'outlet_temperature_of_heat_source_cooling_degree_C': theta_hs_out_c,
        'output_of_heat_source_heating_MJ/h': q_hs_h,
        'output_of_heat_source_sensible_cooling_MJ/h': q_hs_cs,
        'output_of_heat_source_latent_cooling_MJ/h': q_hs_cl,
        'duct_heat_loss_heating_room1_MJ/h': q_loss_duct_h[0],
        'duct_heat_loss_heating_room2_MJ/h': q_loss_duct_h[1],
        'duct_heat_loss_heating_room3_MJ/h': q_loss_duct_h[2],
        'duct_heat_loss_heating_room4_MJ/h': q_loss_duct_h[3],
        'duct_heat_loss_heating_room5_MJ/h': q_loss_duct_h[4],
        'actual_treated_load_heating_room1_MJ/h': q_act_h[0],
        'actual_treated_load_heating_room2_MJ/h': q_act_h[1],
        'actual_treated_load_heating_room3_MJ/h': q_act_h[2],
        'actual_treated_load_heating_room4_MJ/h': q_act_h[3],
        'actual_treated_load_heating_room5_MJ/h': q_act_h[4],
    }
    return constant_value, time_value
