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
  from ioutracker import IOUTracker, BBoxIOU
except ModuleNotFoundError:
  # The relative path is under the home directory.
  import sys
  import os

  relativePaths = [os.path.join(".", "ioutracker", "src"),
                  os.path.join(".", "src")]
  for rPath in relativePaths:
    sys.path.append(rPath)

  from IOUTracker import IOUTracker, BBoxIOU

# In[]

class FuncTest(unittest.TestCase):
  """FuncTest tests the correctness of the function."""

  def testBBoxIOU(self):
    """testBBoxIOU tests the BBoxIOU function."""
    testCases = [([0, 0, 10, 10], [5, 5, 15, 15]),
                 ([0, 0, 5, 5], [10, 10, 15, 15]),
                 ([0, 0, 15, 15], [10, 10, 20, 20])]
    testAns = [0.1428572, 0.0, 0.0833334]

    for (boxA, boxB), ans in zip(testCases, testAns):
      _test_res = BBoxIOU(boxA, boxB)
      _test_res = round(_test_res, 7)
      assert str(_test_res) == str(ans), \
        "Test IOU of ({}), ({}) != {}, but {}".format(boxA, boxB, ans, _test_res)

  def testFilteringDetections(self):
    """testFilteringDetections tests the filtering detections."""
    testCases = [[[0., 0., 10., 10., 0.8], [20., 20., 30., 30., 0.3]], [[]]]
    testAns = [[0., 0., 10., 10., 0.8], []]
    for case, ans in zip(testCases, testAns):
      res = IOUTracker.filter_detections(detections=case, \
                                                     detection_threshold=0.5)
      assert np.array_equiv(np.array(res), np.array(ans)), \
        "Result {} is not the same to {}.".format(res, ans)

  def testDetectionsTransform(self):
    """testDetectionsTransform tests the func detections_transform."""
    testCase = [[0, 1, 10, 20], [0, 0, 0, 0], []]
    testAns = [[0, 1, 10, 21], [0, 0, 0, 0], []]
    for case, ans in zip(testCase, testAns):
      res = IOUTracker.detections_transform(case)
      assert np.array_equal(res, ans), \
        "Result {} is not the same to {}.".format(res, ans)


# In[]

if __name__ == "__main__":
  unittest.main()
