# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:11:21 2020
@author: acer4755g
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

# In[]

dirPath = path.abspath(path.dirname(__file__))

# In[]

long_desc = """"IOUTracker implements a tracking algorithm or method to track
  # objects based on their Intersection-Over-Union (IOU) information across the
  # consecutive frames."""
with open(path.join(dirPath, "README.md"), encoding='utf-8') as fin:
  long_desc = fin.read()

# In[]

pkgs = []
with open(path.join(dirPath, "environ", "requirements.txt"), encoding='utf-8') as fin:
  for line in fin:
    pkgs.append(line.strip().replace("==", ">="))

# In[]

setup(
  name="ioutracker",
  version="1.1.1",
  description=long_desc,
  author="JianKai Wang",
  author_email="gljankai@gmail.com",
  license='MIT',
  url="https://github.com/jiankaiwang/ioutracker",
  classifiers=[
    # refer to https://pypi.org/pypi?%3Aaction=list_classifiers
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
  keywords='''tracking,ioutracker''',
  install_requires=pkgs,
  packages=find_packages(exclude=[])
)

