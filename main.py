import pandas as pd

import central_system as cs
import analyze_control as ac
import json


if __name__ == '__main__':

    filename = 'input.json'
    js = open('input.json', 'r', encoding='utf-8')
    args = json.load(js)

    print("read args:")
    print(args)

    d = cs.get_main_value(**args)

#    d = cs.get_main_value(region=6,
#                          a_mr=29.81, a_or=51.34, a_a=120.08, r_env=266.0/90.0,
#                          insulation='h11', solar_gain='middle',
#                          default_heat_source_spec=True, supply_air_rtd_h=1800.0, supply_air_rtd_c=1800.0,
#                          is_duct_insulated=True, vav_system=False)

    t = d['time_value']

    df_t = pd.DataFrame(t)

    df_t.to_csv('result.csv')
