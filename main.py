import numpy as np
import pandas as pd

import analyze_control as ac


if __name__ == '__main__':

    CASE = 6

    d = ac.get_main_value(CASE)

    c = d['constant_value']
    t = d['time_value']

    df_t = pd.DataFrame(t)

    df_t.to_csv('test.csv')
