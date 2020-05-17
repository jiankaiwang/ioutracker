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

  def testIOUTracker(self):
    """testIOUTracker tests the functionality of IOUTracker."""
    testCases = [[[10, 10, 10, 10, 0.8], [30, 30, 10, 10, 0.9], [10, 50, 10, 10, 0.7]],
                 [[11, 11, 10, 10, 0.85], [11, 49, 10, 10, 0.8]],
                 []]
    testDetRes = [[{"tid": 1,  "numFrames": 1, "largerThanMinT": True},
                   {"tid": 2,  "numFrames": 1, "largerThanMinT": True},
                   {"tid": 3,  "numFrames": 1, "largerThanMinT": True}],
                  [{"tid": 1,  "numFrames": 2, "largerThanMinT": True},
                   {"tid": 3,  "numFrames": 2, "largerThanMinT": True}],
                  []]
    testFinisheds = [[{}],
                     [{"ftid": 2,  "numFrames": 1, "largerThanMinT": True}],
                     [{"ftid": 2,  "numFrames": 1, "largerThanMinT": True},
                      {"ftid": 1,  "numFrames": 2, "largerThanMinT": True},
                      {"ftid": 3,  "numFrames": 2, "largerThanMinT": True}]]

    iouTracker = IOUTracker()
    for case, det, finished in zip(testCases, testDetRes, testFinisheds):
      detected_tracks, finished_tracks = iouTracker(case, returnFinishedTrackers=True, runPreviousVersion=True)
      assert np.array_equiv(np.array(det), np.array(detected_tracks)), \
        "[Previous] (Active Tracks) Result \n{} \nis not the same to\n {}.".format(det, detected_tracks)
      assert np.array_equiv(np.array(finished), np.array(finished_tracks)), \
        "[Previous] (Finished Tracks) Result \n{} \nis not the same to\n {}.".format(finished, finished_tracks)

    iouTracker = IOUTracker()
    for case, det, finished in zip(testCases, testDetRes, testFinisheds):
      detected_tracks, finished_tracks = iouTracker(case, returnFinishedTrackers=True)
      assert np.array_equiv(np.array(det), np.array(detected_tracks)), \
        "[Latest] (Active Tracks) Result \n{} \nis not the same to\n {}.".format(det, detected_tracks)
      assert np.array_equiv(np.array(finished), np.array(finished_tracks)), \
        "[Latest] (Finished Tracks) Result \n{} \nis not the same to\n {}.".format(finished, finished_tracks)

# In[]

if __name__ == "__main__":
  unittest.main()
