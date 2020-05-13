# IOU Tracker Demo

There are two main approaches to run IOUTracker. The first is to load it as a package, this way acts as the usual way you import a module. The second is to run inside a container.

## Run as a package

You can start a tutorial using the [notebook](demo.ipynb). Otherwise, you can install it from scratch and use it in your script.

* Install the IOUTracker first.

```sh
git clone https://github.com/jiankaiwang/ioutracker
cd ioutracker
pip install -q --no-cache-dir -e .
```

* Import the package.

```python
import ioutracker
print(ioutracker.__version__)
```

## Run inside a container

You can run a jupyter notebook-ready environment and also start a tutorial using the [notebook](demo.ipynb). You might require to mount the data volume containing the MOT dataset to the `/tmp/MOT` (default downloading path) inside the container.

```sh
docker pull ioutracker
docker run --rm --name ioutracker -p 8888:8888 -v /tmp/MOT:/tmp/MOT ioutracker
```