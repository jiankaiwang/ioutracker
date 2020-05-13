# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: 0.0.1
@date: 2020/04
@desc: This script tests the MOTMetrics functionalities.
@note:
  Style: pylint_2015
@reference:
  MOT Benchmark Article: https://arxiv.org/pdf/1603.00831.pdf
  Hungarian: https://blog.csdn.net/u014754127/article/details/78086014
"""

import os
import sys
import numpy as np
import pandas as pd
import unittest

try:
  from ioutracker import Hungarian
except Exception:
  # The relative path is under the home directory.
  relativePaths = [os.path.join(".", "ioutracker", "src"),
                   os.path.join(".", "src")]
  for rPath in relativePaths:
    sys.path.append(rPath)

  from Helpers import Hungarian, BBoxIOU, detections_transform
  
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

  def testDetectionsTransform(self):
    """testDetectionsTransform tests the func detections_transform."""
    testCase = [[0, 1, 10, 20], [0, 0, 0, 0], []]
    testAns = [[0, 1, 10, 21], [0, 0, 0, 0], []]
    for case, ans in zip(testCase, testAns):
      res = detections_transform(case)
      assert np.array_equal(res, ans), \
        "Result {} is not the same to {}.".format(res, ans)  

# In[]

class HungarianTests(unittest.TestCase):

  def testTfCoordAndCalIOU(self):
    """testTfCoordAndCalIOU tests the static method transformCoordAndCalIOU."""
    testBox1 = [[10, 10, 30, 30], [0, 0, 10, 10]]
    testBox2 = [[15, 15, 30, 30], [20, 20, 10, 10]]
    testAns = [0.531915, 0.0]
    for box1, box2, ans in zip(testBox1, testBox2, testAns):
      iouRes = round(Hungarian.transformCoordAndCalIOU(box1, box2), 6)
      assert np.equal(iouRes, ans)

  def testAdditionApproach(self):
    """testTfCoordAndCalIOU tests the static method additionApproach."""
    testCases = [[25, 40, 0, 0, 0, 20, 30, 0, 0],
                 [15, 15, 0, 0, 0, 10, 5, 5, 0],
                 [1, 0, 0, 2, 3, 4, 0, 5, 0],
                 [4, 5, 0, 0, 6, 1, 2, 3, 4, 0, 5, 6, 7, 0, 8, 9, 1, 0, 2, 3],
                 [4, 5, 0, 0, 6, 1, 2, 3, 4, 0, 5, 6, 7, 0, 0, 9, 1, 0, 2, 3],
                 [0, 0, 0, 1, 2, 0, 3, 4, 0, 0, 5, 6, 7, 8, 9, 10, 0, 0, 1, 0, 2, 3, 4, 5],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 5, 2, 0, 3, 6, 0, 4, 0, 0, 7, 8, 0, 9],
                 [0, 1, 2, 0, 3, 4, 0, 5, 6],
                 [1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 0, 0, 9, 1, 2, 3, 4, 0, 5]]
    testCaseDims = [(3, 3), (3, 3), (3, 3), (4, 5), (4, 5), \
                    (4, 6), (3, 3), (4, 4), (3, 3), (4, 5)]
    testAns = [True, False, False, False, False, \
               True, True, True, False, True]
    testLines = [3, 2, 2, 3, 3, \
                 4, 3, 4, 1, 4]
    for idx, (case, dims, ans, testline) in enumerate( \
      zip(testCases, testCaseDims, testAns, testLines)):
      array = np.array(case).reshape(dims)
      table = pd.DataFrame(array)
      solved, _, _, _, lines = Hungarian.additionApproach(table)
      assert ans == solved and testline == lines, \
        "Index {} Expected answer {}(lines: {}) != the executed one {}(lines: {})".format(\
          idx, ans, testline, solved, lines)


  def testShiftZeros(self):
    """testShiftZeros tests the static method shiftZeros."""
    testCases = [[15, 15, 0, 0, 0, 10, 5, 5, 0]]
    testLabels = [[0, 0, 1, 1, 1, 2, 0, 0, 1]]
    testDims = [(3, 3)]
    testAns = [[10, 10 ,0, 0, 0, 15, 0, 0, 0]]
    for idx, (case, label, dim, ans) in enumerate( \
      zip(testCases, testLabels, testDims, testAns)):
      case = pd.DataFrame(np.array(case).reshape(dim))
      label = pd.DataFrame(np.array(label).reshape(dim))
      res = Hungarian.shiftZeros(case, label)
      res = np.array(res).reshape(-1)
      assert np.array_equal(ans, res), \
        "Index {} Expected array {} != executed one {}".format(\
          idx, ans, res)

  def testMakeAssignments(self):
    """testMakeAssignments test the static method makeAssignments"""
    testIOUs = [[0, 0, 0.9, 0.8, 0.7, 0, 0, 0.7, 0.8],
                [0, 0, 0, 0, 0.8, 0.9, 0.8, 0, 0, 0, 0, 0.8, 0.7, 0, 0, 0, 0, 0, 1.0, 0],
                [0, 0, 0, 0, 0.8, 0.9, 0.7, 0, 0, 0, 0.9, 0.8, 0.8, 0, 0, 0, 0, 0, 1.0, 0],
                [0, 0, 0, 0, 0.7, 0.8, 0.9, 0.7, 0, 0, 0.9, 0.8, 0.9, 0, 0, 0, 0, 0, 1.0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0.16, 0]
                ]
    testLabels = [[0, 0, 1, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]]
    testDims = [(3, 3), (4, 5), (4, 5), (4, 5), (3, 6)]
    testAns = [[0, 0, 1, 1, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]

    for idx, (iou, label, dim, ans) in enumerate(zip(testIOUs, testLabels, testDims, testAns)):
      iou = pd.DataFrame(np.array(iou).reshape(dim))
      label = pd.DataFrame(np.array(label).reshape(dim))
      res = Hungarian.makeAssignments(iou, label)
      res = np.array(res).reshape(-1)
      assert np.array_equal(ans, res), \
        "Index {} Expected answer {} != the exectued one {}".format(idx, ans, res)

    for idx, (iou, label, dim, ans) in enumerate(zip(testIOUs, testLabels, testDims, testAns)):
      iou = pd.DataFrame(np.array(iou).reshape(dim)).transpose()
      label = pd.DataFrame(np.array(label).reshape(dim)).transpose()
      res = Hungarian.makeAssignments(iou, label).transpose()
      res = np.array(res).reshape(-1)
      assert np.array_equal(ans, res), \
        "(Transposed) Index {} Expected answer {} != the exectued one {}".format(idx, ans, res)

  def testMatchingByIOU(self):
    """testMatchingByIOU tests the static method matchingByIOU."""
    testCases = [[40, 60, 15, 25, 30, 45, 55, 30, 25]]
    testReversedDist = [False]
    testDims = [(3, 3)]
    testAns = [(0, 0, 1, 1, 0, 0, 0, 1, 0)]
    for idx, (case, reversedFlag, dim, ans) in enumerate(\
      zip(testCases, testReversedDist, testDims, testAns)):
      case = pd.DataFrame(np.array(case).reshape(dim))
      res = Hungarian.matchingByIOU(case, reversedDistance=reversedFlag)
      res = np.array(res).reshape(-1)
      assert np.array_equal(ans, res), \
        "(testMatchingByIOU) Index {} Expected answer {} != the executed one {}".format(\
          idx, ans, res)

  def testHungarian(self):
    """testHungarian tests the Hungarian matching algorithm on the IOU data."""
    testGTs = [[[10, 10, 90, 90], [200, 200, 50, 50], [150, 150, 35, 40]],
               [[10, 10, 20, 20], [40, 40, 20, 20]]
               ]  # [x, y, w, h]
    testPreds = [[[205, 205, 55, 55], [145, 145, 35, 45], [20, 20, 95, 95]],
                 [[12, 12, 22, 22], [70, 70, 20, 20], [100, 100, 20, 20]]
                 ]  # [x, y, w, h]
    testAsgs = [[0, 1, 0, 0, 0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0, 0]]
    testIOUs = [[0, 0.578571, 0, 0, 0, 0.676056, 0.596737, 0, 0],
                [0.578571, 0, 0, 0, 0, 0]]
    hungarian = Hungarian()
    for idx, (gt, pred, asg, iou) in enumerate(\
      zip(testGTs, testPreds, testAsgs, testIOUs)):
      iouTable, assignmentTable = hungarian(gt, pred)
      iouTableArray = np.array(iouTable).reshape(-1)
      assignmentTableArray = np.array(assignmentTable).reshape(-1)
      assert np.array_equal(iou, iouTableArray), \
        "(testHungarian:IOU) Index {} Expected answer {} != the executed one {}".format(\
          idx, iou, iouTableArray)
      assert np.array_equal(asg, assignmentTableArray), \
        "(testHungarian:Asg) Index {} Expected answer {} != the executed one {}".format(\
          idx, asg, assignmentTableArray)

# In[]

if __name__ == "__main__":
  unittest.main()
