# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
@version: 0.0.1
@date: 2020/03
@desc: The script implements the data loader of the MOT challenge.
@note:
  Style: pylint_2015
@reference:
"""

import os
import logging
import pandas as pd
import requests
import tqdm
import zipfile
import argparse

# In[]

MOT_ID_LABEL = {1: "Pedestrian", 7: "Static_Person"}
MOT_LABEL_ID = {"Pedestrian": 1, "Static_Person": 7}

# In[]:

def formatBBoxAndVis(dataframe, is_dict=False):
  """formatBBoxAndVis keeps the bbox information and its visibility per frames.

  Args:
    dataframe: the pandas data frame
    is_dict: using the frame id as the key in the dictionary

  Returns:
    frameBBoxes: a list conserves person detection results that each one is a
                 list in which contains [x1, y1, width, height, visible],
                 visible also represents the probability or confident score of
                 the object
  """
  frameBBoxes = []
  fids = list(dataframe["fid"].unique())
  for fid in fids:
    tmp = dataframe[dataframe["fid"] == fid]
    frameBBoxes.append(tmp[["bX", "bY", "bW", "bH", "visible"]].values.tolist())
  if is_dict:
    return dict(zip(fids, frameBBoxes))
  return frameBBoxes

# In[]

def formatForMetrics(dataframe, is_dict=False):
  """formatForMetrics keeps the bbox information, its visibility and uid per frames.

  Args:
    dataframe: the pandas data frame
    is_dict: using the frame id as the key in the dictionary

  Returns:
    frameBBoxes: a list conserves person detection results that each one is a
                 list in which contains [x1, y1, width, height, visible, uid],
                 visible also represents the probability or confident score of
                 the object
  """
  frameBBoxes = []
  fids = list(dataframe["fid"].unique())
  for fid in fids:
    tmp = dataframe[dataframe["fid"] == fid]
    frameBBoxes.append(tmp[["bX", "bY", "bW", "bH", "visible", "uid"]].values.tolist())
  if is_dict:
    return dict(zip(fids, frameBBoxes))
  return frameBBoxes

# In[]

def maybeDownload(name="mot17det", src=None, target=os.path.join("/","tmp"),
                  uncompressed=True):
  """maybeDownload: Maybe download the MOT17Det dataset from the official datasets
                    to the local.

  Args:
    name (primary): the dataset name
    src: the source URL, select one of the name and src
    target: the local directory
    uncompressed: whether to compress the downloaded file

  Return:
    status: 0 (success) or Exception (failed)
  """
  assert os.path.exists(target), "No such folder exists."

  if name or (not src):
    availableURL = {"mot17det":
                    ["https://motchallenge.net/data/MOT17DetLabels.zip",
                     "https://motchallenge.net/data/MOT17Det.zip"]}
    if name not in list(availableURL.keys()):
      raise ValueError("Available datasets: {}".format(list(availableURL.keys())))
    src = availableURL["mot17det"]
  logging.info("Download source: {}".format(src))

  if type(src) == str: src = [src]

  for urlIdx in tqdm.trange(len(src)):
    url = src[urlIdx]
    fname = os.path.basename(url)
    folderName, fileType = fname.split('.')

    # the compressed file path
    filePath = os.path.join(target, fname)

    # download the compressed first
    if os.path.exists(filePath):
      logging.warning("{} existed.".format(filePath))
    else:
      logging.warning("Downloading {} ...".format(url))

      # change to wget tool on the shell
      res = requests.get(url, allow_redirects=True)
      if res.status_code != 200:
        logging.error("Download {} failed.".format(url))
        continue
      with open(filePath, "wb") as fout:
        fout.write(res.content)

    # uncompress the file
    if uncompressed:
      uncompPath = os.path.join(target, folderName)
      assert not os.path.exists(uncompPath), \
        "The folder {} exists. Please delete it first.".format(uncompPath)
      try:
        os.mkdir(uncompPath)
        logging.warning("Created a folder {}.".format(uncompPath))
      except Exception as e:
        raise Exception("Can't create the folder {}. ({})".format(uncompPath, e))

      allowedCompressedType = ["zip"]
      if fileType not in allowedCompressedType:
        raise ValueError("Available compressed type: {}".format(allowedCompressedType))
      if fileType == "zip":
        with zipfile.ZipFile(filePath, 'r') as fin:
          fin.extractall(uncompPath)
        logging.warning("Compressed to folder {}.".format(uncompPath))

  return 0

# In[]:

def loadLabel(src, is_path=True, load_Pedestrian=True, load_Static_Person=True,
              visible_thresholde=0, format_style="onlybbox"):
  """LoadLabel: Load a label file in the csv format.

  Args:
    src: the MOT label file path (available when is_path is True)
    is_path: True or False for whether the src is the file path or not
    load_Pedestrian: whether to load the pedestrian data or not
    load_Static_Person: whether to load the statuc person data or not
    visible_thresholde: the threshold for filtering the invisible person data
    format_style: provides different styles in the lists,
                  "onlybbox" (func: formatBBoxAndVis), "onlybbox_dict" (func: formatBBoxAndVis),
                  "metrics" (func: formatForMetrics), "metrics_dict" (func: formatForMetrics)

  Returns:
    objects_in_frames: a list contains the person detection information per frames
  """
  df = src
  if is_path:
    df = pd.read_csv(src, header=None)
  df.columns = ["fid", "uid", "bX", "bY", "bW", "bH", "conf", "class", "visible"]
  df_persons = df[((df["class"] == MOT_LABEL_ID["Pedestrian"]) & load_Pedestrian) | \
                  ((df["class"] == MOT_LABEL_ID["Static_Person"]) & load_Static_Person)]
  if visible_thresholde:
    df_persons = df_persons[df_persons["visible"] >= visible_thresholde]

  if format_style[:8] == "onlybbox":
    if format_style[-4:] == "dict":
      return formatBBoxAndVis(df_persons, is_dict=True), df_persons
    else:
      # format_style == "onlybbox"
      return formatBBoxAndVis(df_persons), df_persons
  elif format_style[:7] == "metrics":
    if format_style[-4:] == "dict":
      return formatForMetrics(df_persons, is_dict=True), df_persons
    else:
      # format_style == "onlybbox"
      return formatForMetrics(df_persons), df_persons

# In[]

if __name__ == "__main__":

  logging.basicConfig(level=logging.INFO)

  # parsing args for maybeDownload
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, default="mot17det")
  parser.add_argument("--src", type=str, default=None)
  parser.add_argument("--target", type=str, default="/tmp")
  parser.add_argument("--uncompressed", type=int, default=1)
  args = parser.parse_args()

  maybeDownload(name=args.name)