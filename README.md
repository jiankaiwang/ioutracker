# IOU Tracker

[![](https://img.shields.io/badge/Python-3.x-blue)](README.md)

IOUTracker implements a tracking algorithm or method to track objects based on their Intersection-Over-Union (IOU) information across the consecutive frames. The core concept of this algorithm refers to the article (http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf). The idea or the assumption is based on an existing and powerful detector and the high frame rate across the consecutive frames. Under this assumption, you can conduct the object tracking with only the localization and the IOU information. The algorithm conducts under a super-high frame rate and provides a foundation for more complicated calculations upon it. 

On the other hand, such an algorithm requires an evaluation. The evaluation of this implement also refers to two articles, the MOT16 benchmark (https://arxiv.org/abs/1603.00831) and Multi-Target Tracker (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.309.8335&rep=rep1&type=pdf).

**Notice this implement only uses a part of the MOT17 dataset. It only uses the `pedestrian` and `static_person` these two types of data.**

* This implementation uses MOT17Det dataset (https://motchallenge.net/data/MOT17Det/) as an example.
* More information please refer to https://github.com/jiankaiwang/ioutracker.
* The example videos:
[![](https://img.youtube.com/vi/k_5BvwrhNLw/0.jpg)](https://www.youtube.com/watch?v=k_5BvwrhNLw).

## Install

You can easily use the pip command to install this `ioutracker` package.

```sh
pip install git+https://github.com/jiankaiwang/ioutracker
```

You can also clone the repository and then install it on the local path.

```sh
git clone https://github.com/jiankaiwang/ioutracker
cd ioutracker
pip install -q --no-cache-dir -e .
```

## Contribution

[![](https://img.shields.io/badge/Contributor_Convenant-v1.4_adopted-ff69b4)](CODE_OF_CONDUCT.md)

If you want to contribute to IOUTracker, be sure to review the [contribution guidelines](CONTRIBUTING.md). This project adheres to IOUTracker's [code of conduct](CODE_OF_CONDUCT.md). 

We also use [Github issues](https://github.com/jiankaiwang/ioutracker/issues) for tracking requests and bugs.

## Continuous build status

| Types | Status |
|--|--|
| Package | [![](https://travis-ci.org/jiankaiwang/ioutracker.svg?branch=master)](https://travis-ci.org/github/jiankaiwang/ioutracker) |
| Docker | [![](https://img.shields.io/docker/cloud/build/jiankaiwang/ioutracker)](https://hub.docker.com/r/jiankaiwang/ioutracker) |

The continuous build includes several parts, the unit test, and the docker image build. The unit tests basically consist of the algorithm part and the metric part. The docker image build is through the official dockerhub.

## License

[The MIT License (MIT)](LICENSE)