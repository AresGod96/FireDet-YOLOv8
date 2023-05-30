import os

from ultralytics import YOLO
import cv2
import torch

# Test if gpu is detected
print(torch.cuda.is_available())

# Load the model
model = YOLO("weights/FireDet1280.pt")	

model.train(
	data="../dataset/train/data.yaml",
	epochs=100,
	imgsz=640,
	device=0,
	verbose=True,
	batch=32,
	workers=8
)
