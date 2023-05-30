# Fire Detection Model
[![python](https://img.shields.io/badge/python-3.7~3.9-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/pytorch-1.10~2.0-orange)](https://pytorch.org/get-started/previous-versions/)
[![cuda](https://img.shields.io/badge/cuda-11.0~11.7-green)](https://developer.nvidia.com/cuda-downloads)
## Model: FireDet
Based model: [YOLOv8m](https://docs.ultralytics.com/models/yolov8/#supported-tasks)

Support two pretrained weights
* [**FireDet640**](): trained with input size (640, 640)
* [**FireDet1280**](): trained with large input size (1280, 1280)

| Model         | input size<br><sup>(pixels) | mAP<sup>0.5 |
| ------------- | --------------------------- | ----------- |
| FireDet640    | 640                         | 0.77        |
| FireDet1280   | 1280                        | 0.86        |

## Requirements
* Python >= 3.7
* CUDA >= 11.0 (Mine: 11.3)
* [**PyTorch**](https://pytorch.org/get-started/previous-versions/) (Mine: 1.11.0)
* [**ultralytics YOLOv8**](https://github.com/ultralytics/ultralytics/)
    ```bash
    pip install ultralytics
    ```
* Conda env to run if using server: yolov8

### Training

Assume that you have installed [**ultralytics**](https://github.com/ultralytics/ultralytics/), other dependencies and prepared training dataset in YOLO format. You can train the model either in two ways:
1. Ultralytics CLI usage (recommended):

    From scratch
    ```bash
    yolo detect train data='train_data.yaml' model='yolov8m.pt' epochs=100 imgsz=640 batch=32 device=0,1 workers=8
    ```

    Resume an interrupted training
    ```bash
    yolo detect train resume model='weights/FireDet1280-last.pt'
    ```
    See [**train docs**](https://docs.ultralytics.com/usage/cli/#train) for more details.

2. Python script [train.py](train.py)

### Inference

#### Python

Run inference either in two ways:
1. Ultralytics CLI

    ```bash
    yolo detect predict model='weights/FireDet1280.pt' source='assets/fire-test.mp4' show=True
    ```
    See [**predict docs**](https://docs.ultralytics.com/usage/cli/#predict) for more details.

2. Python script [inference.py](inference.py)

#### C++

