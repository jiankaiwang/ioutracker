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
"""

import os
import sys
import numpy as np
import pandas as pd
import unittest

try:
  from ioutracker import FN, FP, ConfusionMatrix
except ModuleNotFoundError:
  # The relative path is under the home directory.
  relativePaths = [os.path.join(".", "ioutracker", "metrics"),
                   os.path.join(".", "ioutracker", "src"),
                   os.path.join(".", "metrics"),
                   os.path.join(".", "src")]
  for rPath in relativePaths:
    sys.path.append(rPath)

  from MOTMetrics import FN, FP, ConfusionMatrix

# In[]

class FNTests(unittest.TestCase):

  def testFN(self):
    """testFN implements the tests for false negatives."""
    testIOUs = [[0, 0.8, 0, 0, 1.0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.3],
                [0, 0, 0, 1.0, 0, 0, 0.8, 0, 0.2, 0, 0, 0],
                [0, 0, 0.8, 0, 0.6, 0, 0, 0],
                [0.578571, 0, 0, 0, 0, 0]]
    testCases = [[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 1, 0, 0, 0],
                 [1, 0, 0, 1, 0, 0]]
    testDims = [(4, 4), (3, 4), (2, 4), (3, 2)]
    # this threshold can't be changed
    # or the testAns and testFNList requires to change the values too
    testThreshold = 0.5
    testAns = [1, 2, 2, 1]
    testFNList = [[3], [0, 1], [1, 3], [1]]
    fn = FN(iouThreshold=testThreshold)
    for idx, (iou, case, dim, ans, fnList) in enumerate(\
      zip(testIOUs, testCases, testDims, testAns, testFNList)):
      iou = pd.DataFrame(np.array(iou).reshape(dim))
      case = pd.DataFrame(np.array(case).reshape(dim))
      res, resList = fn(iou, case)
      assert np.equal(ans, res), \
        "Index {} Expected answer {} != the executed one {}".format(idx, ans, res)
      assert np.array_equal(fnList, resList), \
        "Index {} Expected answer {} != the executed one {}".format(idx, fnList, resList)

# In[]

class FPTests(unittest.TestCase):

  def testFP(self):
    """testFP tests the functionality of class false positives."""
    testIOUs = [[0, 0.8, 0.4, 0, 0, 0],
                [0, 0.8, 0, 0, 1.0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.3],
                [0, 0, 0, 1.0, 0, 0, 0.8, 0, 0.2, 0, 0, 0],
                [0.578571, 0, 0, 0, 0, 0]]
    testCases = [[0, 1, 1, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                 [1, 0, 0, 1, 0, 0]]
    testDims = [(3, 2), (4, 4), (3, 4), (3, 2)]
    # this threshold can't be changed
    # or the testAns and testFPList requires to change the values too
    testThreshold = 0.5
    testAns = [2, 1, 1, 2]
    testFPList = [[1, 2], [3], [2], [1, 2]]
    fp = FP(iouThreshold=testThreshold)
    for idx, (iou, case, dim, ans, fpList) in enumerate(\
      zip(testIOUs, testCases, testDims, testAns, testFPList)):
      iou = pd.DataFrame(np.array(iou).reshape(dim))
      case = pd.DataFrame(np.array(case).reshape(dim))
      res, resList = fp(iou, case)
      assert np.equal(ans, res), \
        "Index {} Expected answer {} != the executed one {}".format(idx, ans, res)
      assert np.array_equal(fpList, resList), \
        "Index {} Expected answer {} != the executed one {}".format(idx, fpList, resList)

# In[]

class CMtests(unittest.TestCase):

  def testCMs(self):
    """testCMs tests the functionality of class ConfusionMatrix."""
    testIOUs = [[0, 0.8, 0, 0, 1.0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0.8],
                [0, 0, 0.8, 0.3, 0, 0, 0, 0.7, 0, 0, 0, 0]]
    testAgs = [[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
               [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]]
    testDims = [(4, 4), (4, 3)]
    # this threshold can't be changed
    testThreshold = 0.5
    testCms = [[4, 0, 0, 0], [2, 2, 1, 0]]
    testTPList = [[0, 1, 2, 3], [0, 2]]
    testFPList = [[], [1, 3]]
    testFNList = [[], [0]]
    confusionMatrix = ConfusionMatrix(iouThreshold=testThreshold)
    for idx, (iou, ag, dim, cm, tp, fp, fn) in \
      enumerate(zip(testIOUs, testAgs, testDims, testCms, testTPList, testFPList, testFNList)):
      iou = pd.DataFrame(np.array(iou).reshape(dim))
      ag = pd.DataFrame(np.array(ag).reshape(dim))
      resCM, resTP, resFP, resFN = confusionMatrix(iouTable=iou, assignmentTable=ag)
      resCM = np.array(resCM).reshape(-1)
      assert np.array_equal(cm, resCM), \
        "Index {}, CM went error. {} != {}".format(cm, resCM)
      assert np.array_equal(tp, resTP), \
        "Index {}, TP went error. {} != {}".format(tp, resTP)
      assert np.array_equal(fp, resFP), \
        "Index {}, FP went error. {} != {}".format(fp, resFP)
      assert np.array_equal(fn, resFN), \
        "Index {}, FN went error. {} != {}".format(fn, resFN)

# In[]

if __name__ == "__main__":
  unittest.main()
