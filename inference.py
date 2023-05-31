"""
Script for running inference on custom images

"""
from ultralytics import YOLO
import cv2
import os
import argparse


if __name__ == "__main__":
	# CLI mode
	parser = argparse.ArgumentParser(description="Inference on images using ULTRALYTICS YOLOV8")
	parser.add_argument("-w", "--weight", help="path to model weight", default="runs/detect/v6-yolov8m/weights/best.pt")
	parser.add_argument("-d", "--dir", help="path to image dir")
	parser.add_argument("-s", "--show", help="display image during inference", default=False, action="store_true")
	parser.add_argument("-o", "--out-dir", help="dir to save inf image(s)", default=None)
	parser.add_argument("-sz", "--imgsz", help="input image resolution", default=640, type=int)
	parser.add_argument("-c", "--conf", help="confidence", default=0.25, type=float)
	parser.add_argument("-iou", "--iou-thres", help="iou threshold", default=0.5, type=float)
	args = parser.parse_args()

	# Load the model
	model_path = args.weight
	if not os.path.exists(model_path):
		print(f"Model weight {model_path} not found!")
		exit(0)
	model = YOLO(model_path)

	show_img = args.show
	img_dir = args.dir
	out_dir = args.out_dir
	iou_thres = args.iou_thres
	imgsz = args.imgsz
	conf = args.conf
	if not os.path.exists(img_dir):
		print(f"Image dir {img_dir} not found!")
		exit(0)

	print(f"================================= Inference ======================================")
	print(f"Model: {model_path}")
	if out_dir is not None:
		if not os.path.exists(out_dir):
			print(f"Saved dir does not exist. Attempting to create dir {out_dir}")
			os.makedirs(out_dir)

	TP = 0
	total = 0
	for img_path in sorted(os.listdir(img_dir)):
		if img_path.endswith(".jpg") or img_path.endswith(".png"):
			total += 1
			img = cv2.imread(os.path.join(img_dir, img_path))
			result = model.predict(img, imgsz=imgsz, conf=conf, iou=iou_thres)
			result = result[0]
			if len(result.boxes) > 0:
				TP += 1
				print(f"{img_path} - Fire")
				res_plotted = result[0].plot()
				if show_img:
					cv2.imshow("result", res_plotted)
					cv2.waitKey(0)
				if out_dir is not None:
					cv2.imwrite(os.path.join(out_dir, img_path) + "_infer.jpg", res_plotted)
			else:
				print(f"{img_path} - No Fire")
				if out_dir is not None:
					cv2.imwrite(os.path.join(out_dir, img_path) + "_infer.jpg", img)

	print(f"TP/Total = {TP}/{total}")
