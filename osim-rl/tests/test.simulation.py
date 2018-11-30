from osim.env import L2RunEnv
import numpy as np
import unittest

class SimulationTest(unittest.TestCase):
    def test1(self):
        
        env = L2RunEnv(visualize=False)
        observation = env.reset()

        action = env.action_space.sample()
        action[5] = np.NaN
        self.assertRaises(ValueError, env.step, action)

if __name__ == '__main__':
    unittest.main()
