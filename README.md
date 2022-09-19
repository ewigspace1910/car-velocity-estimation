# Vehicle Speed Estimation

## Getting Started

### Installation

1. Install Bot-SORT
- Do Install following [the instruction](estimation/README.md)

2. Prepare Datasets
* `_data` folder is constructed as below:

```
_data
├── image
│   └── IMG for check homography,....
│   └...
├── model
│   └── yolov7.pt
│   └── yolov7-d6.pt
│   └── yolov7-e6e.pt
│   └── #pretreined model 4 detection, tracking,...
│   └...
├── run
│   └── saved testing video
│   └...
├── video
│   └── video.mp4....
└── cfg.yaml
```
### Running
* In this stage, firstly we need setting some parameters for the config including:
	- bounding_box(A,B,C,D) : 4 points(x,y) in video to create a bounding box for detection and estimation
	- real_distances		: real-wolrd parameter for converting
	- in _2sd-_solution:
		- lseg-points-up, lseg-points-dw: two way to finding the vanishing point
		- interval : the number of frames. Each 1 interval, trigger an estimation

	
* Then, run this script:

```python
bash estimation/run.sh
```