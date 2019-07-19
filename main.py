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

    t = d.get_time_value_dict()

    df_t = pd.DataFrame(t)

    df_t.to_csv('result.csv')
