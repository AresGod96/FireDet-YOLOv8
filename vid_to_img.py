"""
This script extracts image frames from a specific video

"""

import cv2
import os
import argparse


if __name__ == "__main__":
	# CLI mode
	parser = argparse.ArgumentParser(description="Tool to extract image frames from a video")
	parser.add_argument("-p", "--path", help="path/to/video/", default="../test_video/")
	parser.add_argument("-fs", "--frame_skip", help="no. of frame to skip", type=int, default=30)
	parser.add_argument("-d", "--dest", help="path/to/extract/dir", default="../dataset/MinhTu")
	args = parser.parse_args()
	
	video_path = args.path
	fs = args.frame_skip
	img_dir = args.dest
	if not os.path.exists(img_dir):
		print(f"Folder {img_dir} not found! Create {img_dir}!")
		os.mkdir(img_dir)
	if not os.path.exists(video_path):
		print(f"Video {video_path} not found!")
		exit(0)

	vid_name = os.path.split(os.path.basename(video_path))[1]
	cap = cv2.VideoCapture(video_path)
	frame_idx = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			if frame_idx % fs == 0:
				output_path = img_dir + "/" + vid_name + "-frame_" + str(frame_idx) + ".jpg"
				cv2.imwrite(output_path, frame)
			frame_idx += 1
		else:
			break

	cap.release()
	cv2.destroyAllWindows()

