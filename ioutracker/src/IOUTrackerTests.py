# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: 0.0.1
@date: 2020/03
@desc: The script implements the unit tests of the script IOUTracker.
@note:
  Style: pylint_2015
@reference:
"""

import unittest
import numpy as np

try:
  from ioutracker import IOUTracker
except ModuleNotFoundError:
  # The relative path is under the home directory.
  import sys
  import os

  relativePaths = [os.path.join(".", "ioutracker", "src"),
                   os.path.join(".", "src")]
  for rPath in relativePaths:
    sys.path.append(rPath)

  from IOUTracker import IOUTracker

# In[]

class FuncTest(unittest.TestCase):
  """FuncTest tests the correctness of the function."""

  def testFilteringDetections(self):
    """testFilteringDetections tests the filtering detections."""
    testCases = [[[0., 0., 10., 10., 0.8], [20., 20., 30., 30., 0.3]], [[]]]
    testAns = [[0., 0., 10., 10., 0.8], []]
    for case, ans in zip(testCases, testAns):
      res = IOUTracker.filter_detections(detections=case, \
                                                     detection_threshold=0.5)
      assert np.array_equiv(np.array(res), np.array(ans)), \
        "Result {} is not the same to {}.".format(res, ans)

# In[]

if __name__ == "__main__":
  unittest.main()
