#!/usr/bin/env python
# coding: utf-8

"""IOU Tracker Implementation
@author: jiankaiwang
@version: 0.0.1
@date: 2020/03
@desc: The script implements the IOU tracker algorithm.
@note:
  Style: pylint_2015
@reference:
  Article: http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf
"""

# In[]

import logging
import numpy as np

try:
  from ioutracker import BBoxIOU, detections_transform, Hungarian
except Exception:
  import os
  import sys
  # The relative path is under the home directory.
  relativePaths = [os.path.join(".", "ioutracker", "src"),
                   os.path.join(".", "src")]
  for rPath in relativePaths:
    sys.path.append(rPath)

  from Helpers import BBoxIOU, detections_transform, Hungarian

# In[]

class Track():
  """Track implements the track in the consecutive frames."""

  __tid = None
  __active = False         # True while the track is finished
  __min_t = 0
  __highest_score = 0.0

  # a list for detections keeps all objects,
  # each object keeps the same structure with the input to IOUTracker object
  __detections = []

  def __init__(self, min_t=1):
    """Constructor initializes a track.

    Args:
      min_t: inherits from IOUTracker class.
    """
    self.__tid = None
    self.__active = True
    self.__min_t = min_t
    self.__detections = []

  @property
  def tid(self):
    """__tid getter"""
    return self.__tid

  @tid.setter
  def tid(self, tid):
    """tid setter"""
    assert type(tid) in [type("str"), type(10)]
    self.__tid = tid

  @property
  def active(self):
    """__active getter"""
    return self.__active

  @active.setter
  def active(self, active):
    """__active setter"""
    assert type(active) == type(True)
    self.__active = active

  @property
  def highest_score(self):
    """highest_score getter: returns the highest scores of the detections."""
    return self.__highest_score

  def add_detection(self, detection):
    """Add a detection to a list.

    Args:
      detection: a list keeps the info [bX, bY, bW, bH, visible(prob)]
    """
    visible = detection[4]
    if visible > self.__highest_score:
      self.__highest_score = visible
    self.__detections.append(detection)

  def previous_detections(self):
    """previous_detections returns the last detection in the track.

    Args: None

    Returns:
      detection: the last (or latest) detection in the track, its information is
                 the same with the input to add_detection
    """
    return self.__detections[-1]

  def larger_than_min_t(self):
    """larger_than_min_t checks the track is over the min requirement
       of timestamps.

    Args: None

    Returns:
      bool: whether total time is larger than min time requirement
      total_t: the number of total time stamps
    """
    total_t = len(self.__detections)
    return total_t >= self.__min_t, total_t

  def __del__(self):
    """Deconstructor is called while deleting the object."""
    del self.__tid
    del self.__active
    del self.__min_t
    del self.__detections

# In[]

class IOUTracker():
  """IOUTracker implements the IOU tracker algorithm details."""

  __detection_conf = 0.0
  __iou_threshold = 0.0
  __min_t = 0.0
  __track_min_conf = 0.0
  active_tracks = []
  finished_tracks = []

  __assignedTID = True
  __tidIncrement = 1
  __matchingMethod = None

  def __init__(self, detection_conf=0.2, iou_threshold=0.5, min_t = 1,
               track_min_conf=0.2, assignedTID=True):
    """Constructor.

    Args:
      detection_conf (sigma_l): the detection was removed when its confident score
                                is lower than detection_conf
      iou_threshold (sigma_IOU): the min IOU threshold between a detection and
                                 active tracks
      min_t: the track is filtered out when its length is shorter than min_t
      track_min_conf (sigma_h): the track is filtered out when all of its detections'
                                confident scores are less than the track_min_conf
      assignedTID: the flag to automatically assign a tracker ID to a track,
                   if it is set to False, you can assign a customized TID later
    """
    self.__detection_conf = detection_conf
    self.__iou_threshold = iou_threshold
    self.__min_t = min_t
    self.__track_min_conf = track_min_conf
    self.active_tracks = []
    self.finished_tracks = []

    self.__assignedTID = assignedTID
    self.__tidIncrement = 1
    self.__matchingMethod = Hungarian()

  @property
  def detection_conf(self):
    """detection_conf.getter"""
    return self.__detection_conf

  @property
  def iou_threshold(self):
    """iou_threshold.getter"""
    return self.__iou_threshold

  @property
  def min_t(self):
    """min_t.getter"""
    return self.__min_t

  @property
  def track_min_conf(self):
    """track_min_conf.getter"""
    return self.__track_min_conf

  @staticmethod
  def filter_detections(detections, detection_threshold):
    """Filter the detections whose scores are lower than the IOU threshold.

    Args:
      detections: a list of multiple detections per frame
      detection_threshold: the minimum confident score

    Returns:
      available_detections: a list that removes the detections whose visible
                            (or probability) is lower than the detection_conf
    """
    detections = np.array(detections)
    if detections.shape[0] < 1 or detections.shape[1] < 1:
      return detections
    available_idx = detections[:, 4] >= detection_threshold
    available_detections = detections[available_idx].tolist()
    return available_detections

  def read_detections_per_frame(self, detections):
    """read_detections_per_frame: start to parse the detections per frame.

    Args:
      detections: a list contains multiple detections per frame, each detection
                  keeps [[bX, bY, bWidth, bHeight, visible], [], []]

    Returns: None
    """

    # detections in the shape (num_detections, 5)
    detections = IOUTracker.filter_detections(detections=detections,
                                              detection_threshold=self.__detection_conf)

    logging.debug("detections after being filtering: {}".format(len(detections)))
    for act_track in self.active_tracks:
      # [bX, bY, bW, bH, visible] -> [bX, bY, bX+bW, bY+bH, visible]
      act_track_last_obj = act_track.previous_detections()
      act_track_last_obj = detections_transform(act_track_last_obj)

      detections_iou = [BBoxIOU(act_track_last_obj,
                                detections_transform(detection)) \
                        for detection in detections]
      if len(detections_iou) < 1:
        # solve no detections available
        detections_iou = [-1]
      max_iou = np.max(detections_iou)

      if max_iou >= self.__iou_threshold:
        max_iou_idx = np.argmax(detections_iou)
        act_track.add_detection(detections[max_iou_idx])
        del detections[max_iou_idx]
      else:
        if act_track.highest_score >= self.__track_min_conf and \
          act_track.larger_than_min_t()[0]:
          self.finished_tracks.append(act_track)
        act_track.active = False

    # remove the inactive tracks from the active_tracks list
    # keep the active tracks in the list
    self.active_tracks = [act_track for act_track in self.active_tracks if act_track.active]

    # start a new track with the remained detections
    for detection in detections:
      new_track = Track(min_t=self.__min_t)
      if self.__assignedTID:
        new_track.tid = self.__tidIncrement
        self.__tidIncrement += 1
      new_track.add_detection(detection)
      self.active_tracks.append(new_track)

  def read_detections_per_frame_v2(self, detections):
    """read_detections_per_frame: start to parse the detections per frame.

    Args:
      detections: a list contains multiple detections per frame, each detection
                  keeps [[bX, bY, bWidth, bHeight, visible], [], []]

    Returns:
      detections: the same list of the input one, but with a tracker ID at the
                  end of each detection
    """
    if not self.__assignedTID:
      raise Exception("The latest version requires setting assignedTID True.")

    lenPred = 0 if np.array_equal(detections, []) or np.array_equal(detections, [[]]) \
      else len(detections)

    if lenPred > 0:
      detections = np.array(detections)
      detNums, detInfo = detections.shape
      tidIdx = detInfo
      # the flag -99 would be initialized value
      tidInit = np.repeat(-99, repeats=detNums).reshape((-1, 1))
      detections = np.concatenate((detections, tidInit), axis=-1)

      unfitted_idx = detections[:, 4] < self.__detection_conf
      # the flag -1 would be unfitted detections
      detections[unfitted_idx, tidIdx] = -1

    for act_track in self.active_tracks:
      # [bX, bY, bW, bH, visible] -> [bX, bY, bX+bW, bY+bH, visible]
      act_track_last_obj = act_track.previous_detections()
      act_track_last_obj = detections_transform(act_track_last_obj)

      # it is necessary to process fittable detections
      detections_iou = []
      if lenPred > 0:
        for detection in detections:
          if detection[tidIdx] == -99:
            # only consider the detection whose tid is not assigned
            detections_iou.append(\
              BBoxIOU(act_track_last_obj, detections_transform(detection)))
          else:
            # includes unfiited and already assigned
            detections_iou.append(-1)

      # detections_iou outputs might be (1) empty, (2) all -1, (3) at least one IOU >= 0.0
      if len(detections_iou) < 1:
        # solve no detection available, in other words, it is empty
        detections_iou = [-1]
      max_iou = np.max(detections_iou)

      if max_iou >= self.__iou_threshold:
        max_iou_idx = np.argmax(detections_iou)
        act_track.add_detection(detections[max_iou_idx][:tidIdx])
        detections[max_iou_idx][tidIdx] = act_track.tid
      else:
        if act_track.highest_score >= self.__track_min_conf and \
          act_track.larger_than_min_t()[0]:
          self.finished_tracks.append(act_track)
        act_track.active = False

    # remove the inactive tracks from the active_tracks list
    # keep the active tracks in the list
    self.active_tracks = [act_track for act_track in self.active_tracks if act_track.active]

    # start a new track with the remained detections
    if lenPred > 0:
      for detection in detections:
        if detection[tidIdx] == -99:
          new_track = Track(min_t=self.__min_t)
          new_track.tid = self.__tidIncrement
          self.__tidIncrement += 1
          new_track.add_detection(detection[:tidIdx])
          self.active_tracks.append(new_track)
          detection[tidIdx] = new_track.tid

      detectionChecks = detections[:, tidIdx] == -99
      unprocessedDetections = detectionChecks.astype('int').sum()
      assert unprocessedDetections == 0, \
        "At least {} detections were not correctly processed.".format(unprocessedDetections)

      # type transformation
      detections = detections.tolist()
      for detection in detections:
        detection[-1] = int(detection[-1])

    return detections

  def get_active_tracks(self):
    """get_active_tracks gets the current active tracks.

    Args: None

    Returns:
      active_tracks: a list of active tracks that each is a Track object
    """
    return self.active_tracks

  def get_finished_tracks(self):
    """get_finished_tracks gets the finished tracks.

    Args: None

    Returns:
      finished_tracks: a list of finished tracks that each is a Track object
    """
    return self.finished_tracks

  def clear_finished_tracks(self):
    """clear_finished_tracks cleans finished tracks."""
    del self.finished_tracks
    self.finished_tracks = []

  def __releaseUsr(self):
    """__releaseUsr released all resources used in this object."""
    del self.__detection_conf
    del self.__iou_threshold
    del self.__min_t
    del self.__track_min_conf
    del self.active_tracks
    del self.finished_tracks
    del self.__assignedTID
    del self.__tidIncrement

  def __del__(self):
    """Delete this object."""
    self.__releaseUsr()

  def __previous__(self, detections, returnFinishedTrackers=False):
    """Runs the IOU tracker algorithm across the consecutive frames.

    Args:
      detections: a list contains multiple detections per frame, each detection
                  keeps [[bX, bY, bWidth, bHeight, visible], [], []]

      returnFinishedTrackers: a bool for returning finished trackers

    Returns:
      detectionMapping: a list contains multiple dictionary-structure objects
                        representing each detection, the order of those objects
                        is the same to the detection, the prototype is like

                        [{"tid": value, "numFrames": value, "largerThanMinT": Bool}]

                        in which numFrames is the number of objects in the history
                        , and largerThanMinT is a bool value for the numFrames larger
                        than min_t, if tid == -1, it represents this detection is
                        unassigned
      finishedTrackers: (optional)
                        a list contains multiple dictionary-structure objects
                        representing each finished tracks, the prototype is like

                        [{"ftid": value, "numFrames": value, "largerThanMinT": Bool}]

                        in which the finishedTrackers is similar to the detectionMapping
                        , the difference are that in finishedTrackers ftid is
                        finished tid and finishedTrackers keeps all finished
                        trackers from the beginning
    """
    detectionsCopy = detections.copy()

    # run the IOU trackers
    self.read_detections_per_frame(detections)
    active_tracks = self.get_active_tracks()
    addedDetections = [activeTrack.previous_detections() for activeTrack in active_tracks]

    # assignment between the all detections and added ones
    # assignmentTable whose row is addedDetection and whose col is detectionsCopy
    detectionMapping = []

    if len(detectionsCopy) > 0:
      # if detection is empty, no need to parse active trackers
      _, assignmentTable = self.__matchingMethod(detectionsCopy, addedDetections)
      matching = assignmentTable.apply(lambda col: np.where(col), axis=0)

      for detectionsIdx in list(matching.index):
        matchingRes = list(matching[detectionsIdx][0])
        if len(matchingRes) < 1:
          # this is an unassigned detection, e.g. probability too low
          detectionRes = {"tid": -1, "numFrames": 0, "largerThanMinT": False}
        else:
          matchingIdx = matchingRes[0]
          assignedDetection = active_tracks[matchingIdx]
          larger_than_t, total_t = assignedDetection.larger_than_min_t()
          detectionRes = {"tid": assignedDetection.tid,
                          "numFrames": total_t,
                          "largerThanMinT": larger_than_t}
        detectionMapping.append(detectionRes)

    finishedTrackers = []
    if returnFinishedTrackers:
      for finished in self.get_finished_tracks():
        larger_than_t, total_t = finished.larger_than_min_t()
        finishedRes = {"ftid": finished.tid,
                       "numFrames": total_t,
                       "largerThanMinT": larger_than_t}
        finishedTrackers.append(finishedRes)

    return detectionMapping, finishedTrackers

  def __call__(self, detections, returnFinishedTrackers=False, runPreviousVersion=False):
    """Runs the IOU tracker algorithm across the consecutive frames.

    Args:
      detections: a list contains multiple detections per frame, each detection
                  keeps [[bX, bY, bWidth, bHeight, visible], [], []]

      returnFinishedTrackers: a bool for returning finished trackers

      runPreviousVersion: whether to run the previous version of IOUTracker algorithm.

    Returns:
      detectionMapping: a list contains multiple dictionary-structure objects
                        representing each detection, the order of those objects
                        is the same to the detection, the prototype is like

                        [{"tid": value, "numFrames": value, "largerThanMinT": Bool}]

                        in which numFrames is the number of objects in the history
                        , and largerThanMinT is a bool value for the numFrames larger
                        than min_t, if tid == -1, it represents this detection is
                        unassigned
      finishedTrackers: (optional)
                        a list contains multiple dictionary-structure objects
                        representing each finished tracks, the prototype is like

                        [{"ftid": value, "numFrames": value, "largerThanMinT": Bool}]

                        in which the finishedTrackers is similar to the detectionMapping
                        , the difference are that in finishedTrackers ftid is
                        finished tid and finishedTrackers keeps all finished
                        trackers from the beginning
    """

    if runPreviousVersion:
      logging.info("You selected running the previous version of IOUTracker algorithm.")
      return self.__previous__(detections, returnFinishedTrackers)

    # run the IOU trackers
    detections = self.read_detections_per_frame_v2(detections)
    active_tracks = self.get_active_tracks()
    detectionMapping = []

    lenPred = 0 if np.array_equal(detections, []) or np.array_equal(detections, [[]]) else len(detections)
    if lenPred > 0:
      for detection in detections:
        tid = detection[-1]
        if tid != -1:
          for active_track in active_tracks:
            if tid == active_track.tid:
              larger_than_t, total_t = active_track.larger_than_min_t()
              detectionMapping.append({"tid": tid,
                                       "numFrames": total_t,
                                       "largerThanMinT": larger_than_t})
        else:
          # tid == -1
          detectionMapping.append({"tid": tid,
                                   "numFrames": 0,
                                   "largerThanMinT": False})


    finishedTrackers = []
    if returnFinishedTrackers:
      for finished in self.get_finished_tracks():
        larger_than_t, total_t = finished.larger_than_min_t()
        finishedRes = {"ftid": finished.tid,
                       "numFrames": total_t,
                       "largerThanMinT": larger_than_t}
        finishedTrackers.append(finishedRes)

    return detectionMapping, finishedTrackers

# In[]:

if __name__ == "__main__":
  pass
