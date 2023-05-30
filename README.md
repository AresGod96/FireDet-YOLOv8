# Fire Detection Task
[![python](https://img.shields.io/badge/python-3.7~3.9-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/pytorch-1.10~2.0-orange)](https://pytorch.org/get-started/previous-versions/)
[![cuda](https://img.shields.io/badge/cuda-11.0~11.7-green)](https://developer.nvidia.com/cuda-downloads)
## Model: FireDet-v6 (Based: YOLOv8m)

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
Run training code either in two ways:
1. Ultralytics CLI usage (recommended):
    From scratch
    ```bash
        yolo detect train data='train_data.yaml' model='yolov8m.pt' epochs=100 imgsz=640 batch=32 device=0,1 workers=8
    ```

    Resume an interrupted training
    ```bash
        yolo detect train resume model=last.pt
    ```
    See this [**docs**](https://docs.ultralytics.com/usage/cli/#train) for more details.

2. Python script [train.py](blob://train.py)

### Inference