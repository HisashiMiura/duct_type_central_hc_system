import unittest
import numpy as np
import pandas as pd

import analyze_control as ac


class TestMain(unittest.TestCase):

    def test_total_result(self):

        CASE = 6

        result = ac.get_main_value(CASE)

        t = result.get_time_value_dict()

        df_t = pd.DataFrame(t)

        self.assertAlmostEqual(6.253757008549898, df_t.heating_load_room1[0])
        self.assertAlmostEqual(17.958762933266666, df_t.output_of_heat_source_heating[0])
        self.assertAlmostEqual(1.07990470028857, df_t.output_of_heat_source_sensible_cooling[4798])
        self.assertAlmostEqual(1.1893886188521219, df_t.output_of_heat_source_latent_cooling[4798])


if __name__ == '__main__':
    unittest.main()
