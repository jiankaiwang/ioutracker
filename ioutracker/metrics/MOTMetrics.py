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
"""

# In[]

import numpy as np
import pandas as pd
import tqdm
import logging

try:
  from ioutracker import IOUTracker, loadLabel, Hungarian
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

  from Helpers import Hungarian
  from IOUTracker import IOUTracker
  from MOTDataLoader import loadLabel

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
  __trackerID = None
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
    self.__trackerID = None
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

      assignedTID: the assigned tracker to this ground-truth trajectory, it is
                   recommended as an integer or a string

                   {x | a int, None}
    """
    if groundTruth:
      # add the ground-truth object
      self.__GTObjects.append(groundTruth)

      # count the frame
      self.__frameCount += 1

      if assignedTID is None:
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
      if self.__trackerID is not None:
      # prevent from the first assignment

        # tracker changed
        if self.__fragment:
          # a fragment is also available
          self.__numFragments += 1
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
  __requiredTracking = True

  def __init__(self, detection_conf=0.2, iouThreshold=0.2, min_t = 1,
               track_min_conf=0.5, requiredTracking=True):
    """Constrcutor.

    Args:
      detection_conf (sigma_l): the detection was removed when its confident score
                                is lower than detection_conf
      iouThreshold (sigma_IOU): the min IOU threshold between a detection and
                                active tracks for IOUTracker and Confusion Matrix
      min_t: the track is filtered out when its length is shorter than min_t
      track_min_conf (sigma_h): the track is filtered out when all of its detections'
                                confident scores are less than the track_min_conf
      requiredTracking: whether to run the IOUTracker to get the tracker ID.
                        If it is set to False, it will be going to use the
                        evaluateOnPredsWithTrackerID().
    """
    self.__numGT = 0
    self.__numTP = 0
    self.__numFP = 0
    self.__numFN = 0
    self.__hungarian = Hungarian()
    self.__cm = ConfusionMatrix(iouThreshold=iouThreshold)
    self.__gtTrajectoryDict = {}
    self.__requiredTracking = requiredTracking

    # initializations
    if self.__requiredTracking:
      self.__filteredProbability = detection_conf
      self.__iouTracker = IOUTracker(detection_conf=detection_conf,
                                                iou_threshold=iouThreshold,
                                                min_t=min_t,
                                                track_min_conf=track_min_conf)
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
    if not self.__requiredTracking:
      raise Exception("You initialized the object with wrong parameters, requiredTracking must be True.")

    lenGT = 0 if np.array_equal(groundTruth, [[]]) else len(groundTruth)
    lenPred = 0 if np.array_equal(prediction, [[]]) else len(prediction)

    if lenPred > 0:
      # filter the prediction whose probabilities are lower than the threshold
      predArray = np.array(prediction)
      predIndexes = predArray[:, 4] >= self.__filteredProbability
      filterPreds = predArray[predIndexes].tolist()

      # the filtered prediction (probability is lower than the threshold) is the false positive
      self.__numFP += (len(predArray) - predIndexes.astype('int').sum())

    if lenGT > 0 and lenPred > 0:
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

    if lenPred > 0:
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

    if lenGT > 0 and lenPred > 0:
      # groundTruth contains the ground truth with its self UID, or the GT trajectory
      # addedDetections represents the information to the tracker ID
      # the connection between the ground truth and the prediction is the filterPreds
      # rows: filterPreds, cols: ground truth
      tableGTFilter = assignmentTable

    if lenPred > 0:
      # rows: addedDetections, cols: filterPreds
      _, tableFilterAdded = self.__hungarian(filterPreds, addedDetections)

    # assign the ground truth trajectory
    for key, _ in self.__gtTrajectoryDict.items():
      # initialize the flag for processing the frame information
      self.__gtTrajectoryDict[key].frameCheck = False

    if lenGT > 0:
      for gtIdx in range(0, len(groundTruth), 1):
        gt = groundTruth[gtIdx]
        # it is not required to be an integer
        gtUID = gt[5]
        assert type(gtUID) in [int, str], "The ground truth UID must be an integer or a string."
        allUIDs = list(self.__gtTrajectoryDict.keys())
        if gtUID not in allUIDs:
          newGTTrajectory = GTTrajectory(uid=gtUID)
          self.__gtTrajectoryDict[gtUID] = newGTTrajectory

        if lenPred > 0:
          gtSeries = tableGTFilter.loc[:, gtIdx]
          gt2Preds = (gtSeries == 1)
          gt2PredsAvail = gt2Preds.astype('int').sum() > 0

          if gt2PredsAvail:
            # both the ground truth and the tracker are available
            gt2PredsIdx = gtSeries[gt2Preds].index[0]
            filterPredSeries = tableFilterAdded.loc[:, gt2PredsIdx] == 1
            filterPred2Detn = filterPredSeries[filterPredSeries].index[0]
            assignedTID = activeTracks[filterPred2Detn].tid
            assert type(assignedTID) in [int, str], "The tracker UID must be an integer or a string."
            self.__gtTrajectoryDict[gtUID](gt, assignedTID)
          else:
            # the ground truth is available, but no prediction
            # (== no detection == no tracker)
            self.__gtTrajectoryDict[gtUID](gt, None)
        else:
          # no prediction available
          self.__gtTrajectoryDict[gtUID](gt, None)

        # the ground truth trajectory was processed
        self.__gtTrajectoryDict[gtUID].frameCheck = True
    else:
      # unnecessary matching a tracker ID when it is no ground truth available
      pass

    # the ground truth is not processed, this causes a fragment
    # in other words, no ground truth object is added to the trajectory
    #
    # no need to handle the condition that no ground truth, but the tracker exists
    for key, _ in self.__gtTrajectoryDict.items():
      if not self.__gtTrajectoryDict[key].frameCheck:
        self.__gtTrajectoryDict[key]([], None)
        self.__gtTrajectoryDict[key].frameCheck = True

  def evaluateOnPredsWithTrackerID(self, groundTruth, prediction):
    """Run the whole flow.

       Similar to the caller, this function takes a ground truth
       and a prediction result. The difference between this function and the caller
       is the tracker ID. it is generated on the caller, but it is available in
       this function.
       This function is mainly used on evaluating a prediction from the other
       model or algorithm.

    Args:
      groundTruth: a list contains the BBox information on each frame.
                   Here, we recommended using the MOTDataLoader object.
      prediction: the bbox information predicted by another model, and which is
                   a list contains the BBox information on each frame like
                   [[X1, Y1, W, H, Prob., TID], [X1, Y1, W, H, Prob., TID]]
    """
    if self.__requiredTracking:
      logging.warning("You initialized the object with wrong parameters, requiredTracking should be False.")

    lenGT = 0 if np.array_equal(groundTruth, [[]]) else len(groundTruth)
    lenPred = 0 if np.array_equal(prediction, [[]]) else len(prediction)

    if lenGT > 0 and lenPred > 0:
      # make a hungarian distribution, and it is only available while
      # both ground truth and prediction each contains more than one element
      iouTable, tableGTFilter = self.__hungarian(groundTruth, prediction)

      # get the number of TP, FP, and FN
      _, tpList, fpList, fnList = self.__cm(iouTable, tableGTFilter)

      self.__numTP += len(tpList)
      self.__numFP += len(fpList)
      self.__numFN += len(fnList)

      # here we use the filtered ground truth objects by the probability (or visibility)
      # not the total ground truth
      self.__numGT += len(tpList) + len(fnList)
    elif lenGT > 0:
      # prediction is empty, increasing false negatives
      self.__numFN += lenGT
      self.__numGT += lenGT
    elif lenPred > 0:
      # ground truth is empty, increasing false positives
      self.__numFP += lenPred
    # skip the true negatives

    # initialize each GT trajectory
    for key, _ in self.__gtTrajectoryDict.items():
      # initialize the flag for processing the frame information
      self.__gtTrajectoryDict[key].frameCheck = False

    if lenGT > 0:
      # only consider the condition while ground truth is available
      # the prediction for the tracker is unnecessary to add the detections
      # because in this function, it is under the condition that the tracker ID
      # is provided

      if lenPred > 0:
        # create an identity matrix for the matching
        tableFilterAdded = pd.DataFrame(np.eye(lenPred, dtype=np.int))

      # assign the ground truth trajectory
      for gtIdx in range(0, lenGT, 1):
        gt = groundTruth[gtIdx]
        gtUID = gt[5]
        assert type(gtUID) in [int, str], "The ground truth UID must be an integer or a string."
        allUIDs = list(self.__gtTrajectoryDict.keys())
        if gtUID not in allUIDs:
          newGTTrajectory = GTTrajectory(uid=gtUID)
          self.__gtTrajectoryDict[gtUID] = newGTTrajectory

        if lenPred > 0:
          gtSeries = tableGTFilter.loc[:, gtIdx]
          gt2Preds = (gtSeries == 1)
          gt2PredsAvail = gt2Preds.astype('int').sum() > 0

          if gt2PredsAvail:
            # both the ground truth and the tracker are available
            gt2PredsIdx = gtSeries[gt2Preds].index[0]
            filterPredSeries = tableFilterAdded.loc[:, gt2PredsIdx] == 1
            filterPred2Detn = filterPredSeries[filterPredSeries].index[0]

            try:
              # get the fifth element that represents the tracker ID
              assignedTID = prediction[filterPred2Detn][5]
              assert type(assignedTID) in [int, str], "The tracker UID must be an integer or a string."
            except Exception:
              raise IndexError("Each prediction element requires a tracker ID.")
            self.__gtTrajectoryDict[gtUID](gt, assignedTID)
          else:
            # the ground truth is available, but no prediction
            # (== no detection == no tracker)
            self.__gtTrajectoryDict[gtUID](gt, None)
        else:
          # no prediction is available
          self.__gtTrajectoryDict[gtUID](gt, None)

        # the ground truth trajectory was processed
        self.__gtTrajectoryDict[gtUID].frameCheck = True

    # the ground truth is not processed, this causes a fragment
    # in other words, no ground truth object is added to the trajectory
    #
    # no need to handle the condition that no ground truth, but the tracker exists
    for key, _ in self.__gtTrajectoryDict.items():
      if not self.__gtTrajectoryDict[key].frameCheck:
        self.__gtTrajectoryDict[key]([], None)
        self.__gtTrajectoryDict[key].frameCheck = True

  def getAllGTTrajectory(self):
    """Returns all ground truth trajectories.

    Args: None

    Returns:
      a dictionary keeps each ground truth trajectory pair whose key is uid and
      value is the GTTrajectory object
    """
    return self.__gtTrajectoryDict

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
      print("FN: {}".format(fn))
      print("FP: {}".format(fp))
      print("IDSW: {}".format(idsw))
      print("GT: {}".format(gt))

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
    # label data type transformation
    for gt in GTFrameInfo:
      # transform the datatype of uid to an integer
      gt[5] = int(gt[5])
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
