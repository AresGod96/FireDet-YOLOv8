"""
This script generates a video by synthesizing all the input images

"""

import cv2
import os
import argparse

# Set up video parameters
fps = 1
width = 1920
height = 1080
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

img_dir = "../dataset/benchmark/tmp"
video_dir = "../dataset/MinhTu"
video_name = "Duy_S3-N0810MF.mp4"

if not os.path.exists(video_dir):
	print(f"Folder {video_dir} not found! Create {video_dir}!")
	os.mkdir(video_dir)
out = cv2.VideoWriter(os.path.join(video_dir, video_name), fourcc, fps, (width, height))

idx = 0
for img_path in sorted(os.listdir(img_dir)):
	if img_path.endswith(".jpg"):
		print(f"Frame {idx}: {img_path}")
		idx += 1
		img = cv2.imread(os.path.join(img_dir, img_path))
		img = cv2.resize(img, (width, height))
		# cv2.imshow(f"{img_path}", img)
		# cv2.waitKey(0)
		out.write(img)

cv2.destroyAllWindows()
out.release()

