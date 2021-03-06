import pandas as pd
import os
from functools import lru_cache
from typing import List, Tuple
import numpy as np


@lru_cache()
def read_csv(region: int):
    """
    read conditions
    cache the file
    Args:
        region: region 1-8
    Returns:
        data:
    """

    filename = 'climateData_' + str(region) + '.csv'

    path = os.path.join('climate_data', filename)

    return pd.read_csv(path, skiprows=1, nrows=24*365, names=(
        'date', 'hour', 'time', 'temperature', 'absolute_humidity', 'direct_solar', 'horizontal_sky_solar',
        'night_radiation', 'solar_altitude', 'solar_azimuth', 'blank'), encoding='shift_jis')


def read_temperature(region: int) -> np.ndarray:
    """
    read temperatures(8760)
    Args:
        region: region 1-8
    Returns:
        temperature, degree C, (8760 times)
    """

    data = read_csv(region)

    return np.array(data['temperature'].values)


def read_absolute_humidity(region: int) -> np.ndarray:
    """
    read absolute humidity
    Args:
        region: region 1-8
    Returns:
        absolute humidity, kg/kgDA, (8760 times)
    """

    data = read_csv(region)

    return np.array(data['absolute_humidity'].values)


def read_direct_solar(region: int) -> np.ndarray:
    """
    read direct solar(8760)
    Args:
        region: region 1-8
    Returns:
        direct solar, W/m2, (8760 times)
    """

    data = read_csv(region)

    return np.array(data['direct_solar'].values)


def read_horizontal_sky_solar(region: int) -> np.ndarray:
    """
    read horizontal sky solar(8760)
    Args:
        region: region 1-8
    Returns:
        horizontal sky solar, W/m2, (8760 times)
    """

    data = read_csv(region)

    return np.array(data['horizontal_sky_solar'].values)


def read_night_radiation(region: int) -> np.ndarray:
    """
    read night radiation(8760)
    Args:
        region: region 1-8
    Returns:
        night radiation, W/m2, (8760 times)
    """

    data = read_csv(region)

    return np.array(data['night_radiation'].values)


def read_solar_altitude(region: int) -> np.ndarray:
    """
    read solar altitude(8760)
    Args:
        region: region 1-8
    Returns:
        solar altitude, degree, (8760 times)
    """

    data = read_csv(region)

    return np.array(data['solar_altitude'].values)


def read_solar_azimuth(region: int) -> np.ndarray:
    """
    read solar azimuth(8760)
    Args:
        region: region 1-8
    Returns:
        solar azimuth, degree, (8760 times)
    """

    data = read_csv(region)

    return np.array(data['solar_azimuth'].values)


def get_horizontal_solar(region: int) -> np.ndarray:
    """
    calculate horizontal solar radiation
    Args:
        region: region
    Returns:
        horizontal solar radiation, W/m2, (8760 times)
    """

    # direct solar radiation, W/m2
    direct_solar = read_direct_solar(region=region)

    # horizontal sky radiation, W/m2
    horizontal_sky_solar = read_horizontal_sky_solar(region=region)

    # solar altitude, degree
    solar_altitude = read_solar_altitude(region=region)

    # direct solar radiation on the horizontal surface, W/m2
    direct = direct_solar * np.sin(np.radians(solar_altitude))

    #  horizontal sky radiation on the horizontal surface, W/m2
    sky = horizontal_sky_solar

    # horizontal solar radiation, W/m2
    horizontal_solar_radiation = np.vectorize(lambda d, s: d + s if d > 0.0 else s)(direct, sky)

    return horizontal_solar_radiation


def get_sat_temperature(region: int) -> np.ndarray:
    """
    calculate SAT temperature
    Args:
        region: region 1-8
    Returns:
        SAT temperatures, degree C (8760 times)
    """

    # outdoor temperature, degree C
    temperature = read_temperature(region=region)

    # horizontal solar radiation, W/m2
    horizontal_solar_radiation = get_horizontal_solar(region=region)

    # SAT temperature, degree C
    # 0.034 is the value of the solar absorption ratio divided by heat transfer coefficient(W/m2K)
    # Normally, solar absorption ratio is 0.8, heat transfer coefficient is 23.0.
    sat_temperature = temperature + 0.034 * horizontal_solar_radiation

    return sat_temperature


def get_relative_humidity(theta_ex: np.ndarray, x_ex: np.ndarray) -> np.ndarray:
    """
    calculate relative humidity
    Args:
        theta_ex: outdoor temperature, degree C, (8760 times)
        x_ex: outdoor absolute humidity, kg/kgDA, (8760 times)
    Returns:
        relative humidity, %, (8760 times)
    """

    # convert unit from kg/kgDA to g/kgDA
    x_ex = x_ex * 1000

    # absolute temperature, K
    t_ex = theta_ex + 273.16

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
    k = np.where(theta_ex > 0.0,
                 a1 / t_ex + a2 + a3 * t_ex + a4 * t_ex ** 2 + a5 * np.log(t_ex),
                 b1 / t_ex + b2 + b3 * t_ex + b4 * t_ex ** 2 + b5 * np.log(t_ex))
    p_vs = np.e ** k

    # vapour pressure, Pa
    p_v = 101325 * x_ex / (622 + x_ex)

    # relative humidity, %
    h_ex = p_v / p_vs * 100

    return h_ex
