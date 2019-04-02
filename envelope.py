from typing import List
import numpy as np


class FloorArea:

    def __init__(self, a_mr, a_or, a_a, r_env):
        self.mor = a_mr
        self.oor = a_or
        self.total = a_a
        self.nor = a_a - a_mr - a_or
        # the ratio of the total area of envelope parts to the total floor area
        self.r_env = r_env
        if self.nor < 0.0:
            raise ValueError()


class Spec:

    def __init__(self, insulation: str = 's55', solar_gain: str = 'small'):
        """
        set the spec
        Args:
            insulation: insulation level. specify the level as string following below:
                's55': Showa 55 era level
                'h4': Heisei 4 era level
                'h11': Heisei 11 era level
                'h11more': more than Heisei 11 era level
        solar_gain: solar gain level. specify the level as string following below.
            'small': small level
            'middle': middle level
            'large': large level
        """
        self.insulation = insulation
        self.solar_gain = solar_gain

    def get_q_value(self, region: int) -> float:
        col_num = region - 1

        q_value = {
            's55': [2.8, 2.8, 4.0, 4.7, 5.19, 5.19, 8.27, 8.27],
            'h4': [1.8, 1.8, 2.7, 3.3, 4.2, 4.2, 4.59, 8.01],
            'h11': [1.6, 1.6, 1.9, 2.4, 2.7, 2.7, 2.7, 3.7],
            'h11more': [1.4, 1.4, 1.4, 1.9, 1.9, 1.9, 1.9, 3.7],
        }[self.insulation][col_num]

        return q_value

    def get_mu_h_value(self, region: int) -> float:
        col_num = region - 1

        mu_h_value = {
            's55': {
                'small': [0.029, 0.027, 0.044, 0.048, 0.062, 0.061, 0.129],
                'middle': [0.079, 0.074, 0.091, 0.112, 0.138, 0.134, 0.206],
                'large': [0.115, 0.071, 0.123, 0.161, 0.197, 0.191, 0.268],
            },
            'h4': {
                'small': [0.029, 0.021, 0.040, 0.046, 0.057, 0.056, 0.063],
                'middle': [0.075, 0.070, 0.087, 0.102, 0.132, 0.128, 0.140],
                'large': [0.109, 0.068, 0.119, 0.142, 0.191, 0.185, 0.202],
            },
            'h11': {
                'small': [0.025, 0.024, 0.030, 0.033, 0.038, 0.037, 0.038],
                'middle': [0.071, 0.066, 0.072, 0.090, 0.104, 0.101, 0.107],
                'large': [0.106, 0.098, 0.104, 0.130, 0.153, 0.148, 0.158],
            },
            'h11more': {
                'small': [0.024, 0.022, 0.022, 0.026, 0.030, 0.029, 0.030],
                'middle': [0.070, 0.065, 0.065, 0.078, 0.090, 0.087, 0.092],
                'large': [0.104, 0.096, 0.096, 0.116, 0.137, 0.132, 0.141],
            },
        }[self.insulation][self.solar_gain][col_num]

        return mu_h_value

    def get_mu_c_value(self, region: int) -> float:
        col_num = region - 1

        mu_c_value = {
            's55': {
                'small': [0.021, 0.022, 0.036, 0.039, 0.050, 0.048, 0.106, 0.110],
                'middle': [0.052, 0.052, 0.065, 0.080, 0.095, 0.090, 0.146, 0.154],
                'large': [0.106, 0.071, 0.083, 0.107, 0.124, 0.117, 0.172, 0.184],
            },
            'h4': {
                'small': [0.027, 0.022, 0.032, 0.037, 0.044, 0.043, 0.046, 0.129],
                'middle': [0.049, 0.049, 0.061, 0.072, 0.089, 0.085, 0.086, 0.174],
                'large': [0.101, 0.068, 0.079, 0.094, 0.119, 0.112, 0.111, 0.204],
            },
            'h11': {
                'small': [0.019, 0.019, 0.023, 0.026, 0.027, 0.026, 0.025, 0.023],
                'middle': [0.046, 0.046, 0.049, 0.061, 0.066, 0.062, 0.059, 0.068],
                'large': [0.065, 0.065, 0.067, 0.082, 0.090, 0.084, 0.080, 0.098],
            },
            'h11more': {
                'small': [0.017, 0.017, 0.017, 0.019, 0.021, 0.020, 0.019, 0.019],
                'middle': [0.045, 0.045, 0.043, 0.052, 0.056, 0.053, 0.050, 0.050],
                'large': [0.063, 0.063, 0.060, 0.072, 0.078, 0.073, 0.070, 0.065],
            },
        }[self.insulation][self.solar_gain][col_num]

        return mu_c_value


def get_referenced_floor_area() -> np.ndarray:
    """
    get the referenced floor area of 12 rooms
    Returns:
        Referenced floor area of 12 rooms. (12)
    """
    return np.array([
        29.81,
        16.56,
        13.25,
        10.76,
        10.77,
        3.31,
        1.66,
        3.31,
        13.25,
        4.97,
        10.77,
        1.66,
    ])


def get_hc_floor_areas(floor_area: FloorArea) -> np.ndarray:
    """
    calculate the floor areas of each 12 rooms
    Args:
        floor_area: floor area of zones 'main occupant room', 'other occupant room' and 'non occupant room'
    Returns:
        floor areas of each 12 rooms
        numpy.ndarray 1 * 12
    """

    # the list of the referenced floor area, m2
    as_hcz_r = get_referenced_floor_area()

    # the referenced floor area in the main occupant room, other occupant room and non occupant room, m2
    a_mr_r = 29.81
    a_or_r = 51.34
    a_nr_r = 38.93

    def f(a_hcz_r, i):
        if i < 0:
            raise ValueError
        elif i == 0:
            return a_hcz_r * floor_area.mor / a_mr_r
        elif i < 5:
            return a_hcz_r * floor_area.oor / a_or_r
        elif i < 12:
            return a_hcz_r * floor_area.nor / a_nr_r

    # list(length=12) of floor area
    as_hcz = np.array([f(a_hcz_r, i) for i, a_hcz_r in enumerate(as_hcz_r)])

    return as_hcz
