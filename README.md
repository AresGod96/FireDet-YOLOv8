# Real-Time Fire Detection from Surveillance Camera CCTV
[![python](https://img.shields.io/badge/python-3.7~3.9-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/pytorch-1.10~2.0-orange)](https://pytorch.org/get-started/previous-versions/)
[![cuda](https://img.shields.io/badge/cuda-11.0~11.7-green)](https://developer.nvidia.com/cuda-downloads)

Fire is widely recognized for its destructive power, making fire prevention crucial. In this repository, we present a highly compatible fire detection model designed specifically for surveillance camera view.

## Model: FireDet
Based model: [YOLOv8m](https://docs.ultralytics.com/models/yolov8/#supported-tasks)

Support two pretrained weights
* [**FireDet640**](): trained with input size (640, 640)
* [**FireDet1280**](): trained with large input size (1280, 1280)

| Model         | input size<br><sup>(pixels) | mAP<sup>0.5 |
| ------------- | --------------------------- | ----------- |
| FireDet640    | 640                         | 0.77        |
| FireDet1280   | 1280                        | **0.86**    |

## Requirements
* Python >= 3.7
* CUDA >= 11.0 (Mine: 11.3)
* [**PyTorch**](https://pytorch.org/get-started/previous-versions/) (Mine: 1.11.0)
* [**ultralytics YOLOv8**](https://github.com/ultralytics/ultralytics/)
    ```bash
    pip install ultralytics
    ```
* Conda env to run if using server: yolov8

## Dataset preparation

Download and prepare the dataset in YOLO format. Tools such as [**Roboflow**](https://app.roboflow.com/) are highly recommmended if you want to prepare your own fire dataset. The generated dataset should contain a YAML file, for example, `train_data.yaml`.

## Training

Suppose you have installed [**ultralytics**](https://github.com/ultralytics/ultralytics/), other dependencies and prepared training dataset in YOLO format. You can train the model either in two ways:
1. Ultralytics CLI (recommended)

    From scratch
    ```bash
    yolo detect train data='train_data.yaml' model='yolov8m.pt' epochs=100 imgsz=640 batch=32 device=0,1 workers=8
    ```

    Resume an interrupted training
    ```bash
    yolo detect train resume model='weights/FireDet1280-last.pt'
    ```
    See [**train docs**](https://docs.ultralytics.com/usage/cli/#train) for more details.

2. Python script [`train.py`](train.py)

## Validation

For validation, simply use the command-line usage provided by Ultralytics. First, change the `val` path in your YAML file to the folder used for validation, for example, `../benchmark/images` and run the following command:

```bash
yolo detect val data='data.yaml' model='weights/FireDet1280.pt' device=0,1
```

## Inference in Python

Run inference either in two ways:
1. Ultralytics CLI (videos)

    ```bash
    yolo detect predict model='weights/FireDet1280.pt' source='assets/case2_house.mp4' show=True
    ```
    See [**predict docs**](https://docs.ultralytics.com/usage/cli/#predict) for more details.

2. Python script [`inference.py`](inference.py) (both images and videos)

## Inference in C++

First, you need to convert the [**ONNX**](weights/FireDet1280.onnx) file of the model to [**TensorRT Engine**](https://github.com/NVIDIA/TensorRT) in order to run inference. Follow [**this repo**](https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/README.md), from the step **Build End2End Engine from ONNX** using `build.py`, you should get the converted engine file.

After installing TensorRT and OpenCV libraries, navigate to [`cpp/inference.cpp`](cpp/inference.cpp), modify the engine path in line 27 [`const std::string engine_file_path`](cpp/inference.cpp#L27) along with the input size in line 78 [`cv::Size size = cv::Size{640, 640}`](cpp/inference.cpp#L78), and everything shall be ready for inference using TensorRT.


