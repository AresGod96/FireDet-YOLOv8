"""
This script is used for the performance evaluation of FireDet models on videos
Author: Minh Tu
Copyright (c) PINTEL, Inc.

Usage:
```
	python evaluate_vid.py --path path/to/videos
```
	Required arguments
		-p, --path: path to the testing videos (default: ../test_video/KISA/test)
		-m, --model: path to the model weight (default: runs/detect/fire-detection.pt)

	Optional arguments

"""
from datetime import timedelta
from ultralytics import YOLO
import os
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import argparse

import cv2


if __name__ == "__main__":
	# CLI mode
	parser = argparse.ArgumentParser(description="Performance evaluation on FireDet models")
	parser.add_argument("-p", "--path", help="path to videos", default="../test_video/KISA/train/C050205_001_0340.mp4", dest="src_path")
	parser.add_argument("-m", "--model", help="path to the model weight", default="runs/detect/fire-detection.pt")
	args = parser.parse_args()

	model_path = args.model
	models = [
		[YOLO(model_path), model_path]
		# [YOLO("runs/detect/fire-detection.pt"), "baseline"],
		# [YOLO("runs/detect/v6_yolo8m/weights/best.pt"), "v6_yolo8m"],
	]
	src_path = args.src_path
	cap = cv2.VideoCapture(src_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_duration = 1 / fps

	# Process video
	for (model, model_name) in models:
		print("==========================================Performance Evaluation===========================================")
		print(f"Model: {model_name}")
		print(f"Video source: {src_path} (fps={fps}, total frames={total_frames})")
		results = model.predict(source=src_path)

		frames = []
		for frame_id, result in enumerate(results):
			# assess if fire is detected in this frame
			if len(result) > 0:
				detected = False
				for idx in range(len(result)):
					if result.boxes[idx].conf > 0.3:
						detected = True
						break
				if detected:
					frames.append(frame_id)

		if len(frames) == 0:
			print(f"NO FIRE IS DETECTED..")
			continue
		
		# Post-process: only keep frames that are at most 10 frames to their neighbors
		gap = 10
		final_frames = []
		segment = []
		for idx in range(len(frames)):
			segment.append(frames[idx])
			if idx + 1 < len(frames) and frames[idx + 1] - frames[idx] > gap:
				if len(segment) > 3:
					final_frames.append([segment[0], segment[-1]])
				segment = []
		if len(segment) > 3:
			final_frames.append([segment[0], segment[-1]])

		# print(f"Post-process: fire is detected in {final_frames}")

		# Convert to timestamp in format of hh:mm:ss
		offset = (os.path.splitext(src_path)[0])[-4:]
		mm = int(offset[:2])
		ss = int(offset[2:])
		st_segment = final_frames[0]
		en_segment = final_frames[-1]
		st_time = st_segment[0] * frame_duration
		en_time = en_segment[-1] * frame_duration
		st_time += mm * 60 + ss
		en_time += mm * 60 + ss

		st_stamp = str(timedelta(seconds=int(st_time)))
		en_stamp = str(timedelta(seconds=int(en_time)))
		print(f"Event triggered at {st_stamp}")
		print(f"Event ended at {en_stamp}")

