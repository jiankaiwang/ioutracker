#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: 0.0.1
@date: 2020/03
@desc: An Entry Example of IOU Tracker on MOT Datasets
@note:
  Style: pylint_2015
@reference:
  Article: http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf
"""

import os
import time
import cv2
import numpy as np
import logging
import subprocess
import tqdm

try:
  from ioutracker import IOUTracker, loadLabel
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

  from IOUTracker import IOUTracker
  from MOTDataLoader import loadLabel

# In[]

def colors(num=300):
  np.random.seed(10)
  cp = []
  for _ in range(num):
    r = np.random.randint(low=0, high=255)
    g = np.random.randint(low=0, high=255)
    b = np.random.randint(low=0, high=255)
    cp.append((r, g, b))
  return cp

# In[]

def outputAsFramesToVideo(detection_conf, iou_threshold, min_t, track_min_conf,
                          labelFilePath, frameFilePath, trackingOutput, fps,
                          outputFileName, plotting=True):
  """outputAsFramesToVideo generates the outputs by frames and generates the video
     by these frames.

  Args:
    detection_conf: the hyperparameter defined in IOUTracker
    iou_threshold: the hyperparameter defined in IOUTracker
    min_t: the hyperparameter defined in IOUTracker
    track_min_conf: the hyperparameter defined in IOUTracker
    labelFilePath: the path pointing to `gt.txt`
    frameFilePath: the path pointing to `img1` folder
    trackingOutput: the target folder for the output images
    fps: the frame rate of the video,
         suggesting this parameter would be the same to min_t or its fold
    outputFileName: the name of the output video file
    plotting: plots the frame and outputs the video file

  Returns:
    None
  """

  labels, df = loadLabel(src=labelFilePath, format_style="onlybbox_dict")

  # generates the color list for plotting the box on the frames
  COLOR_LIST = colors()

  iouTracks = IOUTracker(detection_conf=detection_conf,
                         iou_threshold=iou_threshold,
                         min_t=min_t,
                         track_min_conf=track_min_conf)

  tid_count = 1

  if plotting:
    # remove all images existing
    subprocess.call("rm -f {output}/*.jpg {output}/*.mp4".format(\
      output=trackingOutput), shell=True)

  start = time.time()
  for label in tqdm.trange(1, len(labels), 1):
    logging.debug("\n")
    logging.debug("Frame: {}".format(label))

    # iou tracker
    iouTracks.read_detections_per_frame(detections=labels[label])

    active_tacks = iouTracks.get_active_tracks()
    finished_tracks = iouTracks.get_finished_tracks()

    logging.debug("Active tracks: {}".format(len(active_tacks)))
    logging.debug("Finished tracks: {}".format(len(finished_tracks)))

    if plotting:
      # image
      img_name = "{:>6s}.jpg".format(str(label)).replace(' ', '0')
      img_path = os.path.join(frameFilePath, img_name)
      assert os.path.exists(img_path)

      img = cv2.imread(filename=img_path)
      for act_track in active_tacks:
        if not act_track.tid:
          # assign track id to use the color
          act_track.tid = tid_count
          tid_count += 1

          if tid_count >= len(COLOR_LIST) - 20:
            COLOR_LIST = COLOR_LIST + colors()
        # act_track_ped: [bX1, bY1, bW, bH, Visible]
        act_track_ped = act_track.previous_detections()
        # [bX1, bY1, bW, bH, Visible] -> [bX1, bY1, bX2, bY2]
        act_track_ped_coord = IOUTracker.detections_transform(act_track_ped)
        x1, y1, x2, y2 = np.array(act_track_ped_coord, dtype=int)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_LIST[act_track.tid], 2)
        text_x = x1
        text_y = int(y1*1.01)
        cv2.putText(img, "TID:{}".format(str(act_track.tid)), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LIST[act_track.tid], 1, cv2.LINE_AA)

      # add additional info about the video
      parainfo = ["Detection Conf: {:>4s}".format(str(detection_conf)), \
      "IOU Threshold:  {:>4s}".format(str(iou_threshold)), \
      "MIN Time/Frame: {:>4s}".format(str(min_t)), \
      "Track Min Conf: {:>4s}".format(str(track_min_conf)), \
      "FRAMERATE(fps): {:>4s}".format(str(fps))]
      for midx in range(len(parainfo)):
        cv2.putText(img, parainfo[midx], (5, 14*(midx+1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

      tracking_output_file = "{:>6s}.jpg".format(str(label)).replace(" ", "0")
      cv2.imwrite(os.path.join(trackingOutput, tracking_output_file), img)

  if plotting:
    # *.jpg to .mp4
    target_video_path = os.path.join(trackingOutput, "tracking_{}.mp4".format(outputFileName))
    subprocess.call("cat $(find {} | grep 'jpg' | sort) | ffmpeg -f image2pipe -r {} -i - -vcodec libx264 {}".format(
      trackingOutput, fps, target_video_path), shell=True)

  peroid = time.time() - start
  print("Total time cost: {}".format(peroid))

# In[]

if __name__ == "__main__":

  logging.basicConfig(level=logging.info)

  pass
