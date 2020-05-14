# Change Log

## Release 1.1.0

IOU Tracker 1.1.0 improves easy-to-use functionality, including auto tracker ID increment and redesign metric and helper APIs.

### Major Features and Improvements

* Move the matching algorithm, Hungarian, from the metrics.MOTmetrics to the src.helpers, and its unit tests as well.

* Move the BBoxIOU and a static method of the IOUTracker class, detections_transform, to the src.Helpers class.

* Add the auto tracker ID increment as the default feature.

* Add an implementation to call IOUTracker for returning the corresponding track information, including ID, total time period on the unit of the frame, and a flag for considering the validation.

### Bug Fixes and Other Changes

* Update the tutorial.

* Add more functionalities or APIs to the package ioutracker.

## Release 1.0.0

The major algorithm features and metrics were designed and implemented.