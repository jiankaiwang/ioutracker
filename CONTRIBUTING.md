# Contributing guidelines

## Pull Request Checklist

Before sending your pull request, make sure you followed this list.

* Read the [contribution guidelines](CONTRIBUTING.md).
* Read the [code of conduct](CODE_OF_CONDUCT.md).
* Make sure you own the intellectual property or understand the [CLA](https://github.com/jiankaiwang/ioutracker/blob/master/CONTRIBUTING.md#contributor-license-agreements) concept.
* Check if your changes are consists with the [guidelines](https://github.com/jiankaiwang/ioutracker/blob/master/CONTRIBUTING.md#general-guidelines).
* Your changes are consistent with the [coding style](https://github.com/jiankaiwang/ioutracker/blob/master/CONTRIBUTING.md#coding-style).
* Run [Unit Tests](https://github.com/jiankaiwang/ioutracker/blob/master/CONTRIBUTING.md#running-unit-tests).


## How to become a contributor and submit your own code

### Contributor License Agreements

This repository was published for public or research good. If you are going to contribute to this project, please make sure to own the intellectual property, especially while you work for a corporate. 

### General guidelines

If you are interesting to IOUTracker, send us your request! You can refer to the [how to](https://help.github.com/articles/using-pull-requests/) from Github, if you are just getting started.

* Running the unit tests is necessary, no matter in contributing new features or fixing bugs.
* Keep API compatibility or consistency in mind, especially those related to functionalities.
* While merging your pull requests, the burden is transferred to our team. Please follow the coding style. We will also trade-off the cost of the maintenance and the benefit of the contributed features.

### Coding Style

In IOUTracker, Python is the default implemented programming language. You can use `pylint` to check your Python changes using our coding style.

```sh
pip install pylint
pylint --rcfile=.pylintrc yourfile.py
```

### Running unit tests

In IOUTracker, there are two main unit tests you can run. One is for the main algorithm, and the other is for evaluating metrics. Before you 

```sh
# unittests on the main algorithm, IOUTracker
python -m ioutracker.src.IOUTrackerTests

# unittests on the metrics, MOTMetrics
python -m ioutracker.metrics.MOTMetricsTests
```