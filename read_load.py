import pandas as pd
import os
from functools import lru_cache
from typing import List, Tuple
import numpy as np

import envelope


def get_filename(hc: str, region: int, mode: str, nvi: int, j: int, k: int, tsi: int, h_exc: bool) -> str:
    """負荷ファイル名の取得
    :param hc: heating or cooling.  'H' is heating. 'C' is cooling.
    :param region: region 1-8
    :param mode: 'い','ろ','は','全館連続','居室連続','居室間歇'
    :param nvi: 通風レベル 1-3 (int)
    :param j: 断熱水準 1-4 (int)
    :param k: 日射遮蔽レベル 1-3 (int)
    :param tsi: 蓄熱の利用の程度の区分 1-2 (int)
    :param h_exc: 熱交換換気の有無 (boolean)
    :return: ファイル名 (str)    
    """

    # 3.運転モード
    if mode == 'い' or mode == '全館連続' or mode == '全館連続':
        mode_s = 11
    elif mode == 'は' or mode == '居室間歇':
        mode_s = 12
    elif mode == 'ろ' or mode == '居室連続':
        mode_s = 13
    else:
        raise ValueError(mode)

    # 4.断熱水準・熱交換換気の有無
    if j == 1:
        j_s = 2
    elif j == 2:
        j_s = 3 if not h_exc else 4
    elif j == 3:
        j_s = 5 if not h_exc else 7
    elif j == 4:
        j_s = 9 if not h_exc else 11
    else:
        raise ValueError(j)

    # 5.日射遮蔽
    if k == 1:
        k_s = 3
    elif k == 2:
        k_s = 2
    elif k == 3:
        k_s = 1
    else:
        raise ValueError(k)

    # 6.通風設定
    if nvi == 1:
        nvi_s = 1
    elif nvi == 2:
        nvi_s = 3
    elif nvi == 3:
        nvi_s = 5
    else:
        raise ValueError(nvi)

    filename = "I{hc}{region}x_{mode:02d}_{j:02d}_{k}{nvi}2_{tsi}SS.csv" \
        .format(hc=hc, region=region, mode=mode_s, nvi=nvi_s, j=j_s, k=k_s, tsi=tsi)

    return filename


@lru_cache()
def read_csv(hc, filename, debug=False):
    """read load file
    cache the file
    :param hc: 'H' or 'C'
        'H' is heating.
        'C' is cooling.
    :param filename: load file name
    :param debug: In case True, the file name is printed.
    :return:pandas file
    """
    path = os.path.join('data', filename)
    if debug:
        print(path)
    if hc == 'H':
        df = pd.read_csv(path, skiprows=4, nrows=24 * 365, names=(
            'date', 'hour', 'holiday', 'temp', 'humid', '1_HS', '1_HL', '2_HS', '2_HL', '3_HS', '3_HL', '4_HS', '4_HL',
            '5_HS', '5_HL', '6_HS', '6_HL', '7_HS', '7_HL', '8_HS', '8_HL', '9_HS', '9_HL', '10_HS', '10_HL', '11_HS',
            '11_HL', '12_HS', '12_HL'), encoding='shift_jis')
    elif hc == 'C':
        df = pd.read_csv(path, skiprows=4, nrows=24 * 365, names=(
            'date', 'hour', 'holiday', 'temp', 'humid', '1_CS', '1_CL', '2_CS', '2_CL', '3_CS', '3_CL', '4_CS', '4_CL',
            '5_CS', '5_CL', '6_CS', '6_CL', '7_CS', '7_CL', '8_CS', '8_CL', '9_CS', '9_CL', '10_CS', '10_CL', '11_CS',
            '11_CL', '12_CS', '12_CL'), encoding='shift_jis')
    else:
        raise NotImplementedError()
    return df


def get_L_dash_H_R_TSl_Qj_muH_j_k_d_t_i(region: int, mode: str, l: int, j: int, k: int, i: int, debug: bool) \
        -> List[float]:
    """按分しない暖房負荷を取得
    Args:
        region: 地域の区分(1-8)
        mode: 運転モード
            全館連続
            居室間歇
            居室連続
        l: 蓄熱の利用の程度の区分 (1-2)
            1: 蓄熱なし
            2: 蓄熱あり
        j: 断熱水準
            1: S55
            2: H4
            3: H11
            4: H11超
        k: 日射遮蔽レベル
            1: 大
            2: 中
            3: 小
        i: room number
        debug: debug mode
    Returns:
        暖房負荷(MJ/h) 8760時間分
    """

    filename = get_filename('H', region, mode, 1, j, k, l, False)

    df = read_csv('H', filename, debug)

    return df['%d_HS' % i].values / 1000


def get_L_dash_CS_R_NVl_Qj_muH_j_k_d_t_i(region: int, mode: str, l: int, j: int, k: int, i: int, debug: bool) \
        -> List[float]:
    """按分しない冷房顕熱負荷を取得
    Args:
        region: 地域の区分(1-8)
        mode: 運転モード
            全館連続
            居室間歇
            居室連続
        l: 通風レベル
            1: 通風なし
            2: 通風あり（換気回数5回/h）
            3: 通風あり（換気回数20回/h）
        j: 断熱水準
            1: S55
            2: H4
            3: H11
            4: H11超
        k: 日射遮蔽レベル
            1: 大
            2: 中
            3: 小
        i : room number
        debug : debug mode
    Returns:
        冷房顕熱負荷(MJ/h) 8760時間分
    """
    filename = get_filename('C', region, mode, l, j, k, 1, False)
    df = read_csv('C', filename, debug)
    return df['%d_CS' % i].values * -1.0 / 1000


def get_L_dash_CL_R_NVl_Qj_muH_j_k_d_t_i(region: int, mode: int, l: int, j: int, k: int, i: int, debug: bool) \
        -> List[float]:
    """按分しない冷房潜熱負荷を取得
    Args:
        region: 地域の区分(1-8)
        mode: 運転モード
            全館連続
            居室間歇
            居室連続
        l: 通風レベル
            1: 通風なし
            2: 通風あり（換気回数5回/h）
            3: 通風あり（換気回数20回/h）
        j: 断熱水準
            1: S55
            2: H4
            3: H11
            4: H11超
        k: 日射遮蔽レベル
            1: 大
            2: 中
            3: 小
        i: room number
        debug: debug mode
    Returns:
        冷房潜熱負荷(MJ/h) 8760時間分
    """

    filename = get_filename('C', region, mode, l, j, k, 1, False)

    df = read_csv('C', filename, debug)

    return df['%d_CL' % i].values * -1.0 / 1000


def get_envelope_number(insulation: str, solar_gain: str) -> (float, float):
    """
    断熱水準・日射遮蔽レベルから該当する番号j,kを取得する
        Args:
            insulation: 断熱水準
                s55: S55年基準相当レベル
                h4: H4年基準相当レベル
                h11: H11年基準相当レベル
                h11more: H11年基準超相当レベル
            solar_gain: 日射遮蔽レベル
                small: 小
                middle: 中
                large: 大
        Returns:
            断熱水準を表す番号, 日射遮蔽レベルを表す暗号
    """
    j = {
        's55': 1,
        'h4': 2,
        'h11': 3,
        'h11more': 4,
    }[insulation]
    k = {
        'small': 3,
        'middle': 2,
        'large': 1,
    }[solar_gain]
    return j, k


def get_size_factor(floor_area: envelope.FloorArea) -> np.ndarray:
    """
    get size factor
    Args:
        floor_area: floor area
    Returns:
        size factor (float, 12 rooms * 1 value)
    """

    # floor areas of the referenced house, m2 (12 rooms)
    a_r = envelope.get_referenced_floor_area()

    # floor area, m2 (12 rooms)
    a = envelope.get_hc_floor_areas(floor_area=floor_area)

    # size factor, (12 rooms)
    r = a / a_r

    # change dimension to 1 * 12
    return r.reshape(1, 12).T


def get_heating_load(region: int, envelope_spec: envelope.Spec, floor_area: envelope.FloorArea,
                     debug: bool = False) -> np.ndarray:
    """ get hourly heating load
    Args:
        region: region(1-8)
        envelope_spec: envelop spec
            insulation: insulation level. specify the level as string following below:
                's55': Showa 55 era level
                'h4': Heisei 4 era level
                'h11': Heisei 11 era level
                'h11more' : more than Heisei 11 era level
            solar_gain: solar gain level. specify the level as string following below.
                'small': small level
                'middle': middle level
                'large': large level
        floor_area: floor_area
        debug: in case of True, the file name of heat load is printed.
    Returns:
        heating load, MJ/h (12 rooms * 8760 times)
    """

    if region == 8:
        # Heating load is not defined in region 8.
        # In case of region 8, the array(5 rooms * 8760 times) of value 0.0 as heating load is returned.
        referenced = np.full((12, 8760), 0.0)
    else:
        j, k = get_envelope_number(envelope_spec.insulation, envelope_spec.solar_gain)
        referenced = np.array([get_L_dash_H_R_TSl_Qj_muH_j_k_d_t_i(
            region=region, mode='全館連続', l=1, j=j, k=k, i=i, debug=debug) for i in np.arange(1, 13)])

    r = get_size_factor(floor_area)

    return referenced * r


def get_sensible_cooling_load(region: int, envelope_spec: envelope.Spec, floor_area: envelope.FloorArea,
                              debug: bool = False) -> np.ndarray:
    """ get hourly sensible cooling load
    Args:
        region: region(1-8)
        envelope_spec: envelop spec
            insulation: insulation level. specify the level as string following below:
                's55': Showa 55 era level
                'h4': Heisei 4 era level
                'h11': Heisei 11 era level
                'h11more' : more than Heisei 11 era level
            solar_gain: solar gain level. specify the level as string following below.
                'small': small level
                'middle': middle level
                'large': large level
        floor_area: floor_area
        debug: in case of True, the file name of heat load is printed.
    Returns:
        sensible cooling load, MJ/h (12 rooms * 8760 times)
    """
    j, k = get_envelope_number(envelope_spec.insulation, envelope_spec.solar_gain)
    referenced = np.array([get_L_dash_CS_R_NVl_Qj_muH_j_k_d_t_i(
        region=region, mode='全館連続', l=1, j=j, k=k, i=i, debug=debug) for i in np.arange(1, 13)])
    r = get_size_factor(floor_area)

    return referenced * r


def get_latent_cooling_load(region: int, envelope_spec: envelope.Spec, floor_area: envelope.FloorArea,
                            debug: bool = False) -> np.ndarray:
    """ get hourly latent cooling load
    Args:
        region: region(1-8)
        envelope_spec: envelop spec
            insulation: insulation level. specify the level as string following below:
                's55': Showa 55 era level
                'h4': Heisei 4 era level
                'h11': Heisei 11 era level
                'h11more' : more than Heisei 11 era level
            solar_gain: solar gain level. specify the level as string following below.
                'small': small level
                'middle': middle level
                'large': large level
        floor_area: floor_area
        debug: in case of True, the file name of heat load is printed.
    Returns:
        latent cooling load, MJ/h (12 rooms * 8760 times)
    """
    j, k = get_envelope_number(envelope_spec.insulation, envelope_spec.solar_gain)
    referenced = np.array([get_L_dash_CL_R_NVl_Qj_muH_j_k_d_t_i(
        region=region, mode='全館連続', l=1, j=j, k=k, i=i, debug=debug) for i in np.arange(1, 13)])
    r = get_size_factor(floor_area)

    return referenced * r


