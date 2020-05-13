#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: 0.0.1
@date: 2020/04
@desc: This script implements the multiple object tracking (MOT) metrics.
@note:
  Style: pylint_2015
@reference:
  MOT Benchmark Article: https://arxiv.org/pdf/1603.00831.pdf
  Hungarian: https://blog.csdn.net/u014754127/article/details/78086014
"""

# In[]

import numpy as np
import pandas as pd
import tqdm
import logging

try:
  from ioutracker import IOUTracker, BBoxIOU, loadLabel
except ModuleNotFoundError:
  # The relative path is under the home directory.
  import sys
  import os

  relativePaths = [os.path.join(".", "ioutracker", "dataloaders"),
                   os.path.join(".", "dataloaders"),
                   os.path.join(".", "ioutracker", "src"),
                   os.path.join(".", "src")]
  for rPath in relativePaths:
    sys.path.append(rPath)

  from IOUTracker import IOUTracker, BBoxIOU
  from MOTDataLoader import loadLabel

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
    _bbox1 = IOUTracker.detections_transform(bbox1)
    _bbox2 = IOUTracker.detections_transform(bbox2)
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

class GTTrajectory():
  """GTTrajectory simulates the trajectory of the ground truth."""

  __frameCheck = False
  __trackerID = ""
  __gtUID = ""
  __numSwitchID = 0
  __numFragments = 0
  __numFrameTracked = 0
  __frameCount = 0
  __fragment = False
  __GTObjects = None

  def __init__(self, uid):
    """Constructor.

    Args:
      uid: the unique ID of this ground truth trajectory
    """
    self.__frameCheck = False
    self.__trackerID = ""
    self.__gtUID = uid
    self.__numSwitchID = 0
    self.__numFragments = 0
    self.__numFrameTracked = 0
    self.__frameCount = 0
    self.__fragment = False
    self.__GTObjects = []

  @property
  def gtUID(self):
    """gtUID.getter"""
    return self.__gtUID

  @gtUID.setter
  def gtUID(self, gtUID):
    """gtUID.setter"""
    self.__gtUID = gtUID

  @property
  def frameCheck(self):
    """frameCheck.getter"""
    return self.__frameCheck

  @frameCheck.setter
  def frameCheck(self, frameCheck):
    """frameCheck.setter"""
    self.__frameCheck = frameCheck

  @property
  def numSwitchID(self):
    """numSwitchID.getter"""
    return self.__numSwitchID

  @numSwitchID.setter
  def numSwitchID(self, numSwitchID):
    """numSwitchID.setter"""
    self.__numSwitchID = numSwitchID

  @property
  def numFragments(self):
    """numFragments.getter"""
    return self.__numFragments

  @numFragments.setter
  def numFragments(self, numFragments):
    """numFragments.setter"""
    self.__numFragments = numFragments

  @property
  def fragment(self):
    """fragment.getter"""
    return self.__fragment

  @fragment.setter
  def fragment(self, fragment):
    """fragment.setter"""
    self.__fragment = fragment

  @property
  def numFrameTracked(self):
    """numFrameTracked.getter"""
    return self.__numFrameTracked

  @property
  def frameCount(self):
    """frameCount.getter"""
    return self.__frameCount

  def __call__(self, groundTruth, assignedTID):
    """Judges the ID switch and fragments based on the ground truth
    and the tracker.

    Args:
      groundTruth: a ground-truth object in a list whose shape is the same to
                   the one defined in the MOTDataLoader

                   {x | a list, None}

      assignedTID: the assigned tracker to this ground-truth trajectory, it is a string

                   {x | a string, None}
    """
    if groundTruth:
      # add the ground-truth object
      self.__GTObjects.append(groundTruth)

      # count the frame
      self.__frameCount += 1

      if not assignedTID:
        # FN, fragment is always set to True, no matter wether the fragment
        # is set to True (fragment continues) or False (fragment begins)
        self.__fragment = True
        return

    else:
      # no ground truth exists, FP (tracker is assigned) or TN (tracker is also None)
      # fragment is always set to True, no matter whether the fragment
      # is set to True (fragment continues) or False (fragment begins)
      self.__fragment = True
      return

    # TP: both an object assigned and a tracker assigned are available
    # count the frame tracked no matter whether or not the tracker ID is changed
    self.__numFrameTracked += 1

    if self.__trackerID == assignedTID:
      if self.fragment:
        # same tracker with a fragment exists
        self.__numFragments += 1

      # if there is no fragment, no more action to take
    else:
      # tracker changed
      if self.__fragment:
        # a fragment is also available
        self.__numFragments += 1
        self.__numSwitchID += 1
      else:
        # no fragment exists
        self.__numSwitchID += 1

      # set the new tracker ID to the ground truth trajectory
      self.__trackerID = assignedTID

    # at the end, fragment is set to False
    # because a tracker assigned to this ground truth trajectory
    self.__fragment = False

  def getResult(self):
    """Returns the final results of this trajectory.

    Args: None

    Returns:
      a dictionary contains the following tracking information:
        tid: the last tracker ID
        numSwitchID: the number of switch ID
        numFragments: the number of fragments
        numFrameTracked: the number of frame tracked by a tracker
        frameCount: the number of frame in the ground truth trajectory
        fragment: whether this trajectory is under fragment
        objects: a list contains all ground truth objects
    """
    return {"tid": self.__trackerID, \
            "numSwitchID": self.__numSwitchID, \
            "numFragments": self.__numFragments, \
            "numFrameTracked": self.__numFrameTracked, \
            "frameCount": self.__frameCount, \
            "fragment": self.__fragment, \
            "objects": self.__GTObjects}

# In[]

class EvaluateByFrame():
  """EvaluateByFrame implements several MOT metrics frame by frame."""

  __numGT = 0
  __numTP = 0
  __numFP = 0
  __numFN = 0
  __hungarian = None
  __cm = None
  __iouTracker = None
  __gtTrajectoryDict = {}  # {uid: GTTrajectory.Object}
  __filteredProbability = 0.0
  __tidCount = 1

  def __init__(self, detection_conf=0.2, iouThreshold=0.2, min_t = 1,
               track_min_conf=0.5):
    """Constrcutor.

    Args:
      detection_conf (sigma_l): the detection was removed when its confident score
                                is lower than detection_conf
      iouThreshold (sigma_IOU): the min IOU threshold between a detection and
                                active tracks for IOUTracker and Confusion Matrix
      min_t: the track is filtered out when its length is shorter than min_t
      track_min_conf (sigma_h): the track is filtered out when all of its detections'
                                confident scores are less than the track_min_conf
    """
    self.__numGT = 0
    self.__numTP = 0
    self.__numFP = 0
    self.__numFN = 0
    self.__hungarian = Hungarian()
    self.__cm = ConfusionMatrix(iouThreshold=iouThreshold)
    self.__iouTracker = IOUTracker(detection_conf=detection_conf,
                                              iou_threshold=iouThreshold,
                                              min_t=min_t,
                                              track_min_conf=track_min_conf)
    self.__gtTrajectoryDict = {}
    self.__filteredProbability = detection_conf

    # initializations
    # start index of the tracker is 1
    self.__tidCount = 1

  def __call__(self, groundTruth, prediction):
    """Run the whole flow.

    Args:
      groundTruth: a list contains the BBox information on each frame.
                   Here, we recommended using the MOTDataLoader object.
      prediction: the bbox information predicted by another model, and which is
                   a list contains the BBox information on each frame like
                   [[X1, Y1, W, H, Prob.], [X1, Y1, W, H, Prob.]]
    """
    # filter the prediction whose probabilities are lower than the threshold
    predArray = np.array(prediction)
    predIndexes = predArray[:, 4] >= self.__filteredProbability
    filterPreds = predArray[predIndexes].tolist()

    # the filtered prediction (probability is lower than the threshold) is the false positive
    self.__numFP += (len(predArray) - predIndexes.astype('int').sum())

    # make a hungarian distribution
    iouTable, assignmentTable = self.__hungarian(groundTruth, filterPreds)

    # get the number of TP, FP, and FN
    _, tpList, fpList, fnList = self.__cm(iouTable, assignmentTable)

    self.__numTP += len(tpList)
    self.__numFP += len(fpList)
    self.__numFN += len(fnList)

    # here we use the filtered ground truth objects by the probability (or visibility)
    # not the total ground truth
    self.__numGT += len(tpList) + len(fnList)

    # start the tracking algorithm
    self.__iouTracker.read_detections_per_frame(filterPreds)
    activeTracks = self.__iouTracker.get_active_tracks()
    addedDetections = []
    for track in activeTracks:
      if not track.tid:
        track.tid = self.__tidCount
        self.__tidCount += 1
      # get all added detections
      addedDetections.append(track.previous_detections())
    assert len(addedDetections) == len(filterPreds), \
      "The number of detections ({}) is not the same to the number of filtered prediction ({}).".format(\
      len(addedDetections),len(filterPreds))

    # groundTruth contains the ground truth with its self UID, or the GT trajectory
    # addedDetections represents the information to the tracker ID
    # the connection between the ground truth and the prediction is the filterPreds
    # rows: filterPreds, cols: ground truth
    tableGTFilter = assignmentTable
    # rows: addedDetections, cols: filterPreds
    _, tableFilterAdded = self.__hungarian(filterPreds, addedDetections)

    # assign the ground truth trajectory
    for key, _ in self.__gtTrajectoryDict.items():
      # initialize the flag for processing the frame information
      self.__gtTrajectoryDict[key].frameCheck = False
    for gtIdx in range(0, len(groundTruth), 1):
      gt = groundTruth[gtIdx]
      try:
        gtUID = int(gt[5])
      except:
        raise ValueError("Ground Truth UID {} was not an int.".format(gt[5]))
      allUIDs = list(self.__gtTrajectoryDict.keys())
      if gtUID not in allUIDs:
        newGTTrajectory = GTTrajectory(uid=gtUID)
        self.__gtTrajectoryDict[gtUID] = newGTTrajectory

      gtSeries = tableGTFilter.loc[:, gtIdx]
      gt2Preds = (gtSeries == 1)
      gt2PredsAvail = gt2Preds.astype('int').sum() > 0

      if gt2PredsAvail:
        # both the ground truth and the tracker are available
        gt2PredsIdx = gtSeries[gt2Preds].index[0]
        filterPredSeries = tableFilterAdded.loc[:, gt2PredsIdx] == 1
        filterPred2Detn = filterPredSeries[filterPredSeries].index[0]
        assignedTID = activeTracks[filterPred2Detn].tid
        self.__gtTrajectoryDict[gtUID](gt, assignedTID)
      else:
        # the ground truth is available, but no prediction
        # (== no detection == no tracker)
        self.__gtTrajectoryDict[gtUID](gt, "")
      # the ground truth trajectory was processed
      self.__gtTrajectoryDict[gtUID].frameCheck = True

    # the ground truth is not processed, this causes a fragment
    # in other words, no ground truth object is added to the trajectory
    #
    # no need to handle the condition that no ground truth, but the tracker exists
    for key, _ in self.__gtTrajectoryDict.items():
      if not self.__gtTrajectoryDict[key].frameCheck:
        self.__gtTrajectoryDict[key]([], "")
        self.__gtTrajectoryDict[key].frameCheck = True

  def __getGTTrajectoryResult(self):
    """getGTTrajectoryResult calculates the number of the fragments and the switch IDs.

    Args: None

    Returns:
      numFragments: the total number of fragment on all trajectories
      numSwitchID: the total number of switch ID on all trajectories
    """
    numFragments = 0
    numSwitchID = 0

    for trajKey in list(self.__gtTrajectoryDict.keys()):
      traj = self.__gtTrajectoryDict[trajKey]
      numFragments += traj.numFragments
      numSwitchID += traj.numSwitchID

    return numFragments, numSwitchID

  def getMetricsMeta(self, printOnScreen=False):
    """getMetricsMeta returns the metadata of each premetric.

    Args:
      printOnScreen: whether or not to print the meta information on the screen

    Returns:
      results: a dict json,
               {"TP": 0, "FP": 0, "FN": 0, "GT": 0, "numFragments": 0, "numSwitchID": 0}
    """
    numFragments, numSwitchID = self.__getGTTrajectoryResult()

    if printOnScreen:
      print("TP:{:6}".format(self.__numTP))
      print("FP:{:6}".format(self.__numFP))
      print("FN:{:6}".format(self.__numFN))
      print("GT:{:6}".format(self.__numGT))
      print("Fragment Number: {:6}".format(numFragments))
      print("SwitchID Number: {:6}".format(numSwitchID))

    return {"TP": self.__numTP,
            "FP": self.__numFP,
            "FN": self.__numFN,
            "GT": self.__numGT,
            "numFragments": numFragments,
            "numSwitchID": numSwitchID}

  def getMOTA(self, printOnScreen=False):
    """getMOTA calculate the Multiple Object Tracking Accuracy (MOTA) metric.

    Args:
      printOnScreen: whether or not to print the meta information on the screen

    Returns:
      mota: a float number ranging from -unlimit (Worst) to 100 (Best)
    """

    metaRes = self.getMetricsMeta()
    fn = metaRes["FN"]
    fp = metaRes["FP"]
    idsw = metaRes["numSwitchID"]
    gt = metaRes["GT"]
    mota = 1 - (fn + fp + idsw) / gt

    if printOnScreen:
      print("MOTA: {:3.6f}".format(mota))

    return mota

  def getTrackQuality(self, printOnScreen=False, mt=0.8, ml=0.2):
    """getTrackQuality calculate the MT, PT, ML ratios.

    Args:
      printOnScreen: whether or not to print the meta information on the screen
      mt: the rate of Most Tracked (MT)
      ml: the rate of Most Lost (ML)

    Returns:
      a dictionary that contains the information of the tracker quality
      numMT: the number of most tracked trajectory
      numPT: the number of partial tracked trajectory
      numML: the number of most lost trajectory
      numTraj: the number of all trajectories
      rateMT: the ratio of most tracked trajectory
      ratePT: the ratio of partial tracked trajectory
      rateML: the ratio of most lost trajectory
    """

    numTrajectories = len(self.__gtTrajectoryDict)
    numMT = 0
    numPT = 0
    numML = 0
    for trajKey in list(self.__gtTrajectoryDict.keys()):
      traj = self.__gtTrajectoryDict[trajKey]
      frameCount = traj.frameCount
      numFrameTracked = traj.numFrameTracked
      trackedRate = numFrameTracked / frameCount
      if trackedRate >= mt:
        numMT += 1
      elif trackedRate < ml:
        numML += 1
      else:
        numPT += 1
    rateMT = round(numMT / numTrajectories, 6)
    ratePT = round(numPT / numTrajectories, 6)
    rateML = round(numML / numTrajectories, 6)

    if printOnScreen:
      print("Total trajectories: {}".format(numTrajectories))
      print("MT Number: {}, Ratio: {:3.3f}%".format(numMT, rateMT * 100))
      print("PT Number: {}, Ratio: {:3.3f}%".format(numPT, ratePT * 100))
      print("ML Number: {}, Ratio: {:3.3f}%".format(numML, rateML * 100))

    assert numMT + numPT + numML == numTrajectories, \
      "The number of trajectory is not correct."

    return {"numMT": numMT, "numPT": numPT, "numML": numML, "numTraj": numTrajectories, \
            "rateMT": rateMT, "ratePT": ratePT, "rateML": rateML}

  def getCM(self, printOnScreen=False):
    """getCM calculate the confusion matrix and its relative rates.

    Args:
      printOnScreen: whether or not to print the meta information on the screen

    Returns:
      cmRes: a confusion matrix in a dictionary showing the number of each conditions,
             and their rates as well.
    """

    metaRes = self.getMetricsMeta()
    tp = metaRes["TP"]
    fn = metaRes["FN"]
    fp = metaRes["FP"]
    gt = metaRes["GT"]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = tp / gt
    f1score = 2 * (recall * precision) / (recall + precision)

    if printOnScreen:
      print("Recall: {:3.3f}%".format(recall * 100))
      print("Precision: {:3.3f}%".format(precision * 100))
      print("Accuracy: {:3.3f}%".format(accuracy * 100))
      print("F1 Score: {:1.3f}".format(f1score))

    return {"TP": tp, "FN": fn, "FP": fp, "GT": gt,
            "recall": recall, "precision": precision, "accuracy": accuracy,
            "f1score": f1score}

# In[]

def ExampleEvaluateOnFrame():
  """Example1 concatenates the above operations to a flow.
  This example shows how to evaluate these metrics on the frame data.
  """
  groundTruth = [[10, 10, 20, 20], [40, 40, 20, 20]]
  prediction = [[12, 12, 22, 22], [70, 70, 20, 20], [100, 100, 20, 20]]
  arrayGT = np.array(groundTruth)
  arrayPred = np.array(prediction)
  print("Ground Truth BBox (X1,Y1,W,H): {}".format(groundTruth))
  print("Prediction BBox (X1,Y1,W,H): {}".format(prediction), end='\n\n')

  # step.1: makes the assignments matching ground truth with prediction
  hungarian = Hungarian()
  iouTable, assignmentTable = hungarian(groundTruth, prediction)
  print("IOU Table:")
  print(iouTable, end='\n\n')
  print("Assignments:")
  print(assignmentTable, end='\n\n')

  # step.2: calculate the confusion matrix
  cm = ConfusionMatrix(iouThreshold=0.5)
  cmRes, tpList, fpList, fnList = cm(iouTable, assignmentTable)
  print("Confusion Matrix:")
  print(cmRes, end='\n\n')
  print("TP:", arrayPred[tpList])
  print("FP:", arrayPred[fpList])
  print("FN:", arrayGT[fnList])

# In[]

def ExampleEvaluateMOTDatasets(labelFilePath, predictions=None,
                               filteredProbability=0.2, iouThreshold=0.2,
                               min_t=1, track_min_conf=0.5,
                               printOnScreen=False):
  """ExampleEvaluateMOTDatasets implements the evaluation on MOT datasets.

  Args:
    labelFilePath: the label file path pointing to the MOT datasets
    predictions: a dictionary that keeps all object detection information,
                 it is similiar to the LABELS information from the loadLabel()
    filteredProbability (= detection_conf): filtered probability both for
                                            the ground truth and the prediction
    iouThreshold: the iou threshold between the ground truth and the prediction
    min_t: the min timestamp is required as the active track
    track_min_conf: at least one timestamp in the track, its detection conf
                    must be over this track_min_conf
    printOnScreen: whether or not to print the meta information on the screen

  Returns:
    metaRes: refer to @getMetricsMeta
    cmRes: refer to @getCM
    motaRes: refer to @getMOTA
    trajRes: refer to @getTrackQuality
  """
  # here we filter the ground-truth object whose visible is over the threshold
  LABELS, DFPERSONS = loadLabel(
    src=labelFilePath, is_path=True, load_Pedestrian=True, load_Static_Person=True,
    visible_thresholde=filteredProbability, format_style="metrics_dict")

  evalFrame = EvaluateByFrame(detection_conf=filteredProbability,
                              iouThreshold=iouThreshold,
                              min_t=min_t,
                              track_min_conf=track_min_conf)

  for fid in tqdm.trange(1, len(LABELS), 1):
    # continue to add detections frame by frame
    # here the ground truth and prediction datasets are the same
    # instead, you can replace them with the result from the another model
    # if you use another model to get the prediction, remember to filter them
    # by the probability
    GTFrameInfo = LABELS[fid]
    if not predictions:
      prediction = GTFrameInfo
    else:
      prediction = predictions[fid]
    evalFrame(GTFrameInfo, prediction)

  metaRes = evalFrame.getMetricsMeta(printOnScreen=printOnScreen)
  cmRes = evalFrame.getCM(printOnScreen=printOnScreen)
  motaRes = evalFrame.getMOTA(printOnScreen=printOnScreen)
  trajRes = evalFrame.getTrackQuality(printOnScreen=printOnScreen)

  return metaRes, cmRes, motaRes, trajRes

# In[]

class EvaluateOnMOTDatasets():
  """EvaluateOnMOTDatasets evaluates the metrics on the MOT Datasets."""

  __numDatasets = 0
  __sumMetrics = {}
  __aveMetrics = {}

  __sumOperations = ["TP", "FP", "FN", "GT", "numFragments", "numSwitchID",
                     "numMT", "numPT", "numML", "numTraj"]
  __aveOperations = ["mota", "recall", "precision", "accuracy", "f1score",
                     "rateMT", "ratePT", "rateML"]
  __allOperations = []

  @property
  def numDatasets(self):
    """numDatasets.getter"""
    return self.__numDatasets

  @numDatasets.setter
  def numDatasets(self, numDatasets):
    """numDatasets.setter"""
    self.__numDatasets = numDatasets

  def __init__(self):
    """Constructor."""
    self.__numDatasets = 0

    for metric in self.__sumOperations:
      self.__sumMetrics[metric] = 0
    for metric in self.__aveOperations:
      self.__aveMetrics[metric] = 0.0

    self.__allOperations = list(set(self.__sumOperations + self.__aveOperations))

  def __addMetric(self, key, value):
    """__addMetric handles the addition of the specific metric.

    Args:
      key: the key defined in self.__aveOperations and self.__sumOperations
      value: the relative value to that key
    """
    if key not in self.__allOperations:
      raise Exception("The {} is not an allowed metric {}.".format(key, self.__allOperations))

    if key in self.__sumOperations:
      self.__sumMetrics[key] += value
      return

    if key in self.__aveMetrics:
      self.__aveMetrics[key] += value
      return

  def __call__(self, evaluator):
    """Adds each metric from the evaluator.

    Args:
      evaluator: the evaluated result from the function ExampleEvaluateMOTDatasets.
    """
    self.__numDatasets += 1

    metaRes, cmRes, motaRes, trajRes = evaluator

    self.__addMetric("mota", motaRes)

    for res in [metaRes, cmRes, trajRes]:
      for key, value in res.items():
        if key in self.__allOperations:
          self.__addMetric(key, value)
        else:
          logging.info("Metric {} is not considered as the output one.".format(res))

  def getResults(self, printOnScreen=True):
    """getResults returns the averaging result of the total added evaluators.

    Args:
      printOnScreen: whether or not to print the result information on the screen

    Returns:
      aveMetrics: the metrics required averaging
      sumMetrics: the metrics required summation
    """
    # average the metrics required
    for key in list(self.__aveMetrics.keys()):
      self.__aveMetrics[key] /= self.__numDatasets

    if printOnScreen:
      for metricsDict in [self.__aveMetrics, self.__sumMetrics]:
        for key in list(metricsDict.keys()):
          print("Metric {}: Value {}".format(key, metricsDict[key]))

    return self.__aveMetrics, self.__sumMetrics

# In[]

if __name__ == "__main__":
  pass
