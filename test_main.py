import unittest
import numpy as np
import pandas as pd

import analyze_control as ac


class TestMain(unittest.TestCase):

    def test_total_result(self):

        CASE = 6

        result = ac.get_main_value(CASE)

        t = result['time_value']

        df_t = pd.DataFrame(t)

        self.assertAlmostEqual(6.2304730398910255, df_t.heating_load_room1[0])
        self.assertAlmostEqual(17.85745693530602, df_t.output_of_heat_source_heating[0])
        self.assertAlmostEqual(1.0609561341074933, df_t.output_of_heat_source_sensible_cooling[4798])
        self.assertAlmostEqual(1.187266007959914, df_t.output_of_heat_source_latent_cooling[4798])


if __name__ == '__main__':
    unittest.main()
