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

  def __init__(self, detection_conf=0.5, iou_threshold=0.5, min_t = 1,
               track_min_conf=0.5):
    """Constructor.

    Args:
      detection_conf (sigma_l): the detection was removed when its confident score
                                is lower than detection_conf
      iou_threshold (sigma_IOU): the min IOU threshold between a detection and
                                 active tracks
      min_t: the track is filtered out when its length is shorter than min_t
      track_min_conf (sigma_h): the track is filtered out when all of its detections'
                                confident scores are less than the track_min_conf
    """
    self.__detection_conf = detection_conf
    self.__iou_threshold = iou_threshold
    self.__min_t = min_t
    self.__track_min_conf = track_min_conf
    self.active_tracks = []
    self.finished_tracks = []

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

  @staticmethod
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
      act_track_last_obj = IOUTracker.detections_transform(act_track_last_obj)

      detections_iou = [BBoxIOU(act_track_last_obj,
                                IOUTracker.detections_transform(detection)) \
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
      new_track.add_detection(detection)
      self.active_tracks.append(new_track)

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

# In[]:

if __name__ == "__main__":
  pass
