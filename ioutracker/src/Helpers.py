#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: 0.0.1
@date: 2020/04
@desc: This script implements several helpful algorithms.
@note:
  Style: pylint_2015
@reference:
  Hungarian: https://blog.csdn.net/u014754127/article/details/78086014
"""

# In[]

import numpy as np
import pandas as pd

# In[]:

def BBoxIOU(boxA, boxB):
  """BBoxIOU implements the IOU ratio.

  Args:
    boxA: the first bbox in shape (4,) of (x1, y1, x2, y2)
    boxB: the second bbox in shape (4,) of (x1, y1, x2, y2)

  Returns:
    iou: a float value represents the IOU ratio
  """
  iou = 0.0

  # determine the coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # compute the area of the intersection area
  interArea = max(0, xB - xA + 1e-6) * max(0, yB - yA + 1e-6)

  # compute the area of both the prediction and ground-truth rectangle
  boxAArea = (boxA[2] - boxA[0] + 1e-6) * (boxA[3] - boxA[1] + 1e-6)
  boxBArea = (boxB[2] - boxB[0] + 1e-6) * (boxB[3] - boxB[1] + 1e-6)

  # calculate the iou based on the set theory
  iou = float(interArea) / float(boxAArea + boxBArea - interArea)

  return iou

# In[]

def detections_transform(detection):
  """detections_transform transforms coordinates into [bX1, bY1, bX2, bY2].

  Args:
    detections: [bX, bY, bWidth, bHeight, visible]

  Returns:
    transformed_detections: [bX1, bY1, bX2, bY2]
  """
  if len(detection) < 4:
    return detection
  x1 = detection[0]
  y1 = detection[1]
  x2 = detection[0] + detection[2]
  y2 = detection[1] + detection[3]
  return [x1, y1, x2, y2]

# In[]

class Hungarian():
  """Hungarian helps to match the bbox from the result file to the ground truth."""

  def __init__(self):
    pass

  @staticmethod
  def transformCoordAndCalIOU(bbox1, bbox2):
    """Transform the coordinates and calculate the IOU ratio.

    Args:
      bbox1: the list of the single box information in the shape of [x, y, w, h]
      bbox2: the list of another box information in the shape of [x, y, w, h]

    Returns:
      bboxIOU: a floating value represents the IOU ratio
    """
    _bbox1 = detections_transform(bbox1)
    _bbox2 = detections_transform(bbox2)
    bboxIOU = BBoxIOU(_bbox1, _bbox2)
    return bboxIOU

  @staticmethod
  def additionApproach(iouTable):
    """additionApproach implements the way of addition on optimal assignments.

    Args:
      iouTable: a pandas dataframe whose columns are the ground truth, and
                whose rows are the predictions

    Returns:
      solved: bool to indicate whether find the optimal assignments
      iouLabel: the iou ratio table with reduction
      linesInfo: the operation adding lines
      countZero: the number of erased zeros
      countLine: the number of min lines
    """

    def __searchFromPrediction(iouLabel, foundValue, linesInfo):
      """__searchFromPrediction implements searching strategy from the prediction.

      Args:
        iouLabel: a pandas dataframe uses 1's for 0.0 and 0's for other values
        foundValue: whether the row in the dataframe contains the number of zeros
        linesInfo: the operation adding lines

      Returns:
        iouLabelCopy: a pandas dataframe that erased zero-flags
        linesInfoCopy: a pandas dataframe that shows the operation of adding lines
        partCountZero: how many zero-flags are erased
        partCountLine: how many lines are used to erase the zero-flags
      """
      iouLabelCopy = iouLabel.copy()
      linesInfoCopy = linesInfo.copy()
      partCountZero = 0
      partCountLine = 0
      # get the number of zero value
      predAxis = iouLabelCopy.apply(lambda row: np.count_nonzero(row), axis=1)
      # filled in the iouLabel table
      matchPredIndexes = list(predAxis[(predAxis == foundValue)].index)
      for idx in matchPredIndexes:
        if np.any(iouLabelCopy.loc[idx, :]):
          # there are n-zeros
          zeros = np.where(iouLabelCopy.loc[idx, :])[0]
          partCountZero += len(zeros)
          iouLabelCopy.loc[idx, :] = False  # remove the zero-flag
          linesInfoCopy.loc[idx, :] += 1
          partCountLine += 1
      return iouLabelCopy, linesInfoCopy, partCountZero, partCountLine

    def __searchFromGT(iouLabel, foundValue, linesInfo):
      """__searchFromGT implements searching strategy from the ground truth.

      Args:
        iouLabel: a pandas dataframe uses 1's for 0.0 and 0's for other values
        foundValue: whether the column in the dataframe contains the number of zeros
        linesInfo: the operation adding lines

      Returns:
        iouLabelCopy: a pandas dataframe that erased zero-flags
        linesInfoCopy: a pandas dataframe that shows the operation of adding lines
        partCountZero: how many zero-flags are erased
        partCountLine: how many lines are used to erase the zero-flags
      """
      iouLabelCopy = iouLabel.copy()
      linesInfoCopy = linesInfo.copy()
      partCountZero = 0
      partCountLine = 0
      # get the number of zero value
      gtAxis = iouLabelCopy.apply(lambda col: np.count_nonzero(col), axis=0)
      # filled in the iouLabel table
      matchGTIndexes = list(gtAxis[(gtAxis == found)].index)
      for idx in matchGTIndexes:
        if np.any(iouLabelCopy.loc[:, idx]):
          # there are n-zeros
          zeros = np.where(iouLabelCopy.loc[:, idx])[0]
          partCountZero += len(zeros)
          iouLabelCopy.loc[:, idx] = False  # remove the zero-flag
          linesInfoCopy.loc[:, idx] += 1
          partCountLine += 1
      return iouLabelCopy, linesInfoCopy, partCountZero, partCountLine

    numPreds, numGTs = iouTable.shape
    maxN = max(numPreds, numGTs)
    minN = min(numPreds, numGTs)
    countZero, countLine = 0, 0

    iouLabel = iouTable.apply(lambda row: np.equal(row, 0.0), axis=1)
    iouLabel.astype('int')

    linesInfo = np.zeros(shape=(numPreds*numGTs,), dtype=np.int).reshape((numPreds, numGTs))
    linesInfo = pd.DataFrame(linesInfo)

    found = maxN
    while found > 0:
      # check two solutions
      predIOULabel, predLinesInfo, predCountZero, predCountLine = \
        __searchFromPrediction(iouLabel, found, linesInfo)
      gtIOULabel, gtLinesInfo, gtCountZero, gtCountLine = \
        __searchFromGT(iouLabel, found, linesInfo)

      if predCountZero >= gtCountZero:
        iouLabel = predIOULabel
        countZero += predCountZero
        countLine += predCountLine
        linesInfo = predLinesInfo
      else:
        iouLabel = gtIOULabel
        countZero += gtCountZero
        countLine += gtCountLine
        linesInfo = gtLinesInfo

      if predCountZero == 0 and gtCountZero == 0:
        found -= 1

      # check whether the zero-flag exists or not
      if not np.any(iouLabel):
        # no zero-flag exists
        break

    solved = countLine == minN
    iouLabel = iouTable.apply(lambda row: np.equal(row, 0.0), axis=1)
    return solved, iouLabel, linesInfo, countZero, countLine

  @staticmethod
  def shiftZeros(iouTable, linesInfo):
    """shiftZeros helps to decrease the distance for the other non-zero values.

    Args:
      iouTable: a pandas dataframe that contains original IOU values
      linesInfo: a pandas dataframe that shows the optimal assignments (lines) information

    Returns:
      newIOUTable: a pandas dataframe that shifts zeros
    """
    iouTableShape = iouTable.shape
    flatten = np.array(iouTable).reshape(-1)
    lines = np.array(linesInfo).reshape(-1)
    minVal = flatten[np.where(flatten > 0)].min()
    # minus the min value
    flatten[np.where(lines == 0)[0]] -= minVal
    # add the min value to the cross value
    flatten[np.where(lines == 2)[0]] += minVal
    # reshape back to the original shape
    iouTableArray = flatten.reshape(iouTableShape)
    return pd.DataFrame(iouTableArray)

  @staticmethod
  def makeAssignments(iouTable, iouLabel):
    """makeAssignments implements the assignments.

    Args:
      iouTable: a pandas dataframe that shows the original correlation between
                each pair, here it is the IOU information
      iouLabel: a pandas dataframe that shows the solution for each pair of
                ground truth and prediction
    Returns:
      assignmentTable: a pandas dataframe that shows the final assignments
    """
    numPreds, numGTs = iouLabel.shape

    if numPreds > numGTs:
      # assignments from the ground truth
      iouLabelCopy = iouLabel.transpose().copy().astype('int')
      iouTableCopy = iouTable.transpose().copy()
    else:
      # assignments from the predictions
      iouLabelCopy = iouLabel.copy().astype('int')
      iouTableCopy = iouTable.copy()

    assignmentTable = pd.DataFrame(np.zeros_like(iouLabelCopy))

    numChoices = iouLabelCopy.apply(lambda row: np.count_nonzero(row), axis=1)
    choiceCount = numChoices.min()
    maxChoice = numChoices.max()

    while choiceCount <= maxChoice:
      # choice table must be updated
      numChoices = iouLabelCopy.apply(lambda row: np.count_nonzero(row), axis=1)
      selected = list(numChoices[numChoices == choiceCount].index)
      for index in selected:
        series = iouLabelCopy.loc[index]
        options = list(series[series == 1].index)
        if choiceCount == 1:
          colIndex = options[0]
        else:
          # more options (>=2)
          optionCounts = iouLabelCopy.loc[:, options].apply(lambda col: np.count_nonzero(col),
                                                            axis=0)
          # TODO: here we select the index whose IOU is the highest one,
          # if there are more than one indexes whose IOU ratios are similar,
          # choose the first one by default
          minAvailOptions = optionCounts.min()
          availableMinOptions = list(optionCounts[optionCounts == minAvailOptions].index)
          availableIOUInfo = iouTableCopy.loc[index, availableMinOptions]
          totalCandidateIndex = list(availableIOUInfo[availableIOUInfo == availableIOUInfo.max()].index)
          colIndex = totalCandidateIndex[0]

        # mark for assignment
        assert assignmentTable.loc[index, colIndex] == 0, "Assignment failed."
        assignmentTable.loc[index, :] = -1
        assignmentTable.loc[:, colIndex] = -1
        assignmentTable.loc[index, colIndex] = 1
        # update iouLabelCopy
        iouLabelCopy.loc[index, :] = 0
        iouLabelCopy.loc[:, colIndex] = 0

      if len(selected) < 1:
        # no more index
        choiceCount += 1

    # change -1 to 0
    assignmentTable = assignmentTable.replace(-1, 0)

    if numPreds > numGTs:
      # assignments from the ground truth, transpose the assignment back
      assignmentTable = assignmentTable.transpose()

    return assignmentTable

  @staticmethod
  def matchingByIOU(iouTable, reversedDistance=True):
    """matchingByIOU implements the Hungarian matching algorithm on IOU information.

    Args:
      iouTable: a pandas dataframe that contains the iou information
      reversedDistance: a flag for reverseing the values in the iouTable

    Returns:
      assignmentTable: a pandas dataframe that shows the assignment information
    """
    IOUTable = iouTable.copy()
    if reversedDistance:
      # use 1 - IOU_ratio to represent the relative distance
      # more closer to 0, more relating; closer to 1, farther away
      IOUTable = 1 - iouTable.copy()

    # row reduction
    IOUTable = IOUTable.apply(lambda row: row - np.min(row), axis=1)

    # column reduction
    IOUTable = IOUTable.apply(lambda col: col - np.min(col), axis=0)

    # test for an optimal assignment
    solved, iouLabel, linesInfo, countZero, countLine = \
      Hungarian.additionApproach(IOUTable)

    # shift zero until the problem is solved
    while not solved:
      IOUTable = Hungarian.shiftZeros(IOUTable, linesInfo)
      solved, iouLabel, linesInfo, countZero, countLine = \
        Hungarian.additionApproach(IOUTable)

    # making the final assignment
    assignmentTable = Hungarian.makeAssignments(iouTable, iouLabel)

    return assignmentTable

  def __call__(self, groundTruth, prediction):
    """Runs the matching algorithms.

    Args:
      groundTruth: a list of ground truth bboxes in the shape of
                   [[x, y, w, h], [], ...]
      prediction: a list of prediction bboxes in the shape of
                  [[x, y, w, h], [], ...]


    Returns:
      oriIOUTable: a pandas dataframe that contains the each pair of its IOU information
      assignmentTables: a pandas dataframe that shows the assignment results
                        in which the column respresents the ground truth and
                        the row represents the prediction
    """
    numGTs = len(groundTruth)
    numPreds = len(prediction)
    if numGTs < 1 or numPreds < 1:
      raise ValueError("The number of ground truth or prediction object was invalid.")
    ttlIOUChecks = np.zeros(shape=(numGTs * numPreds,), dtype=np.int)\
      .reshape((numPreds, numGTs))

    # create an IOU table for the distribution
    IOUTable = pd.DataFrame(data=ttlIOUChecks,
                            index=list(range(numPreds)),
                            columns=list(range(numGTs)))

    # Reversed IOU Distancing
    # row: each prediction, column: each ground truth
    for pred in range(numPreds):
      for gt in range(numGTs):
        iouRatio = round(Hungarian.transformCoordAndCalIOU(
          bbox1 = prediction[pred], bbox2 = groundTruth[gt]), 6)
        IOUTable.loc[pred, gt] = iouRatio

    # making the final assignment
    assignmentTable = Hungarian.matchingByIOU(IOUTable)

    return IOUTable, assignmentTable

# In[]

class FN():
  """FN implements the evaluation of false negatives on each frame.

  The following conditions are considered as the false negatives (FN).
  1. the number of prediction is lower than the number of ground truth
  2. the assigned pair whose IOU is lower than the threshold so the ground truth
     is regraded as the false negative
  """

  def __init__(self, iouThreshold=0.5):
    """Constructor.

    Args:
      iouThreshold: the IOU Threshold for considering false negatives
                    (the invalid assignment)
    """
    self.__iouThreshold = iouThreshold

  def __call__(self, iouTable, assignmentTable):
    """Runs to calculate the false negatives.

    Args:
      iouTable: a pandas dataframe that contains the IOU information for
                each pair of boxes like

                    0   1   2   3
                0   0 0.8   0   0
                1 1.0   0   0   0
                2   0   0 0.7   0
                3   0   0   0 0.8

                OR, a pandas dataframe that receives from the Hungarian algorithm.

      assignment: a pandas dataframe that 1s are the assignments for the pair,
                  0s are the default value (no assignment), like

                    0 1 2 3
                  0 0 1 0 0
                  1 1 0 0 0
                  2 0 0 1 0
                  3 0 0 0 1

                  OR a pandas dataframe that receives from the Hungarian algorithm.
    Returns:
      numFNs: the number of false negatives
      fnList: a list contains the index of the ground truth that does not be predicted
    """
    filteredIOU = iouTable * assignmentTable
    filteredIOU = filteredIOU >= self.__iouThreshold
    filteredIOU = filteredIOU.astype('int')
    gtSeries = filteredIOU.apply(lambda col: np.count_nonzero(col), axis=0)
    fnList = list(gtSeries[gtSeries == 0].index)
    numFNs = len(fnList)
    return numFNs, fnList

# In[]

class FP():
  """FP implements the evaluation of false positives on each frame.

  The following conditions are considered as the false positives (FP).
  1. the number of the prediction is more than the number of the ground truth
  2. the assigned pair whose IOU ratio is lower than the threshold so the prediction
     is regarded as the false positives
  """

  def __init__(self, iouThreshold):
    """Constructor.

    Args:
      iouThreshold: the IOU Threshold for considering false positives
                    (the invalid assignment)
    """
    self.__iouThreshold = iouThreshold

  def __call__(self, iouTable, assignmentTable):
    """Runs to calculate the false negatives.

    Args:
      iouTable: a pandas dataframe whose columns represent the ground truth,
                and whose rows represent the prediction, like

                     0   1   2   3
                0    0 0.8   0   0
                1  1.0   0   0   0
                2    0   0 0.6   0
                3    0   0   0 0.8

                OR a pandas dataframe that receives from the Hungarian algorithm.
      assignment: a pandas dataframe that 1s are the assignments for the pair,
                  0s are the default value (no assignment), like

                    0 1 2 3
                  0 0 1 0 0
                  1 1 0 0 0
                  2 0 0 1 0
                  3 0 0 0 1

                  OR a pandas dataframe that receives from the Hungarian algorithm.
    Returns:
      numFPs: the number of false positives
      fpList: a list of each false positive index
    """
    filteredIOU = iouTable * assignmentTable
    filteredIOU = filteredIOU >= self.__iouThreshold
    filteredIOU = filteredIOU.astype('int')
    predSeries = filteredIOU.apply(lambda row: np.count_nonzero(row), axis=1)
    fpList = list(predSeries[predSeries == 0].index)
    numFPs = len(fpList)
    return numFPs, fpList

# In[]

class ConfusionMatrix():
  """ConfusionMatrix implements the confusion matrix on the tracking result."""

  def __init__(self, iouThreshold):
    """Constructor.

    Args:
      iouThreshold: the IOU threshold for false negatives and false positives
    """
    self.__iouThreshold = iouThreshold
    self.__fn = FN(iouThreshold = iouThreshold)
    self.__fp = FP(iouThreshold = iouThreshold)

  def __call__(self, iouTable, assignmentTable):
    """Runs to calculate the confusion matrix on the result of each frame.

    Args:
      iouTable: a pandas dataframe whose columns represent the ground truth,
                and whose rows represent the prediction, like

                        0   1   2
                0(a)    0   0 0.8
                1(b)  0.3   0   0
                2(c)    0 0.7   0
                3(d)    0   0   0

                OR a pandas dataframe that receives from the Hungarian algorithm.
      assignment: a pandas dataframe that 1s are the assignments for the pair,
                  0s are the default value (no assignment), like

                    0 1 2
                  0 0 0 1
                  1 1 0 0
                  2 0 1 0
                  3 0 0 0

                  OR a pandas dataframe that receives from the Hungarian algorithm.
    Returns:
      cmRes: a pandas dataframe represents the confusion matrix, like

                    GTP  GTN
             PredP    2    2
             PredN    1    0

             GTP: the ground truth positives
             GTN: the ground truth negatives
             PredP: the prediction positives
             PredN: the prediction negatives

      tpList: a list for the true positives (note indexes are prediction-based), like

              [0, 2] or [a, c]

      fpList: a list for the false positives, like

              [1, 3] or [b, d]

      fnList: a list for the false negatives, like

              [0]
    """
    numPreds, numGTs = iouTable.shape

    filteredIOU = iouTable * assignmentTable
    filteredIOU = filteredIOU >= self.__iouThreshold
    filteredIOU = filteredIOU.astype('int')

    # tpList is a list that contains the TP prediction indexes
    tpList = filteredIOU.apply(lambda row: np.count_nonzero(row), axis=1)
    tpList = list(tpList[tpList == 1].index)
    numTPs = len(tpList)

    numFPs, fpList = self.__fp(iouTable, assignmentTable)
    numFNs, fnList = self.__fn(iouTable, assignmentTable)

    # assert the number of each TP, FP, FN
    assert numPreds == numTPs + numFPs, \
      "precision error: Pred. {} != {} TPs + FPs".format(numPreds, numTPs + numFPs)
    assert numGTs == numTPs + numFNs, \
      "recall error: GT. {} != {} TPs + FNs".format(numGTs, numTPs + numFNs)

    cmArray = np.array([numTPs, numFPs, numFNs, 0], dtype=np.int).reshape((2, 2))
    cmRes = pd.DataFrame(cmArray, index=["PredP","PredN"], columns=["GTP", "GTN"])
    return cmRes, tpList, fpList, fnList

# In[]

if __name__ == "__main__":
  pass
