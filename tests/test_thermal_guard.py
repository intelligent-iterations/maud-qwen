"""Boundary decisions for the independent host temperature interlock (stdlib only)."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import subprocess

spec = importlib.util.spec_from_file_location('thermal_guard', Path(__file__).parents[1]/'scripts/thermal_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class ThermalDecisionTest(unittest.TestCase):
    def setUp(self):
        self.config = {'cpu_pause_c':85, 'gpu_pause_c':80, 'cpu_resume_c':75, 'gpu_resume_c':72,
                       'expected_gpu_limit_w':200, 'expected_cpu_max_khz':3000000}
        self.reading = {'cpu_c':70, 'gpu_c':68, 'gpu_limit_w':200, 'cpu_max_khz':3000000, 'gpu_thermal_slowdown':False}

    def test_either_temperature_at_boundary_pauses(self):
        for key,value in [('cpu_c',85),('gpu_c',80)]:
            self.assertEqual(guard.decision({**self.reading,key:value},self.config),'hot')

    def test_both_temperatures_must_cool(self):
        self.assertEqual(guard.decision(self.reading,self.config),'cool')
        for key,value in [('cpu_c',76),('gpu_c',73)]:
            self.assertEqual(guard.decision({**self.reading,key:value},self.config),'hold')

    def test_driver_thermal_flag_pauses_below_threshold(self):
        self.assertEqual(guard.decision({**self.reading,'gpu_thermal_slowdown':True},self.config),'hot')

    def test_reverted_limits_prevent_resume(self):
        for key,value in [('gpu_limit_w',350),('cpu_max_khz',5486441)]:
            self.assertEqual(guard.decision({**self.reading,key:value},self.config),'limits_changed')

    def test_sensor_failure_retains_driver_diagnostic(self):
        failure = subprocess.CompletedProcess(['nvidia-smi'], 255, 'Failed to initialize NVML: Unknown Error\n', '')
        with patch.object(guard.subprocess, 'run', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'GPU telemetry exit 255: Failed to initialize NVML: Unknown Error'):
                guard.sensors(Path('/not-read-on-failure'))


if __name__ == '__main__':
    unittest.main()
