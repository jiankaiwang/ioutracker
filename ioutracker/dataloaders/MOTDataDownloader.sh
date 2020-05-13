#!/bin/bash

# OS: ubuntu-like, debian-like
# Maintainer: JianKai Wang (https://jiankaiwang.no-ip.biz/)
# Version: 1.0
# Desc: Download the MOTDet dataset to the local path.
# Example: 
#   sh MOTDataDownloader.sh <localPath>

helps()
{
  # show help message
  echo "Usage:"
  echo "  sh MOTDataDownloader.sh <targetPath>"
  echo "  sh MOTDataDownloader.sh /tmp/MOT"
}

if [ -z "$1" ]; then
  helps
  exit 0
fi

DATAURL=https://motchallenge.net/data/MOT17Det.zip
LABELURL=https://motchallenge.net/data/MOT17DetLabels.zip
DATAFNAME=MOT17Det.zip
LABELFNAME=MOT17DetLabels.zip
UNCOMPRESSEDDATA=MOT17Det
UNCOMPRESSEDLABEL=MOT17DetLabels

localPath=$1

if [ -d $localPath ]; then
  echo "Error: The target path exists."
  
  while true; do
    read -p "Delete $1 first? (Yes|No) " yn
    case $yn in
      [Yy]* ) rm -rf $localPath; break;;
      [Nn]* ) echo "Info: Please assign another path."; exit 1;;
      * ) echo "Error: Please answer yes (y) or no (n)";;
    esac
  done
fi

mkdir -p $localPath
echo "Info: Created the target path."

download()
{
  echo "Downloading $1 to $2 ..."
  curl -o $2 $1
  echo "Downloading was complete."
}

localDataPath="$localPath/$DATAFNAME"
localUncompDataPath="$localPath/$UNCOMPRESSEDDATA"
localLabelPath="$localPath/$LABELFNAME"
localUncompLabelPath="$localPath/$UNCOMPRESSEDLABEL"

download $DATAURL $localDataPath
unzip $localDataPath -d $localUncompDataPath

download $LABELURL $localLabelPath
unzip $localLabelPath -d $localUncompLabelPath

echo "Downloading process was finished."
exit 0
