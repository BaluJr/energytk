#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
from nilmtk.tests.testingtools import data_dir
from nilmtk import (Appliance, MeterGroup, ElecMeter, HDFDataStore, 
                    global_meter_group, TimeFrame, DataSet)
from nilmtk.utils import tree_root, nodes_adjacent_to_root
from nilmtk.elecmeter import ElecMeterID
from nilmtk.building import BuildingID
from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk.simulate import PredefinedStateMachines
from nilmtk.metrics import f1_score
import pandas as pd

class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'co_test.h5')
        cls.dataset = DataSet(filename)
        
    @classmethod
    def tearDownClass(cls):
        cls.dataset.store.close()

    def test_f1(self):
        pass
        # The code below doesn't work yet because it complains that
        # AttributeError: Attribute 'metadata' does not exist in node: '/'
        """
        co = CombinatorialOptimisation()
        co.train(self.dataset.buildings[1].elec)
        disag_filename = join(data_dir(), 'co-disag.h5')
        output = HDFDataStore(disag_filename, 'w')
        co.disaggregate(self.dataset.buildings[1].elec.mains(), output)
        output.close()
        disag = DataSet(disag_filename)
        disag_elec = disag.buildings[1].elec
        f1 = f1_score(disag_elec, self.dataset.buildings[1].elec)
        """


    def test_precision:
        
        testing_appliance = pd.DataFrame()
        testing_appliance['orig'] = np.arr[[2, 0, 0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0]]
        testing_appliance['test_exact'] = np.arr[[1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]]
        testing_appliance['test_half'] = np.arr[[2, 0, 0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0]]
        testing_appliance['test_double'] = np.arr[[4, 0, 0, 4, 4, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0]]
        testing_appliance['test_halfprecise'] = np.arr[[2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 0]]
        testing_appliance['test_orig'] = np.arr[[0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0]]
        testing_appliance['test_halfprecise'] = np.arr[[2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 0]]

        simulator = PredefinedStateMachines()
        simulator.simulate
        

if __name__ == '__main__':
    unittest.main()
