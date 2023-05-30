#define _CRT_SECURE_NO_WARNINGS
#include "chrono"
#include "yolov8.hpp"
#include "opencv2/opencv.hpp"

const std::vector<std::string> CLASS_NAMES = {
	"fire"
};

const std::vector<std::vector<unsigned int>> COLORS = {
	{0, 0, 255}
};

std::string getBaseFileName(const std::string& filePath) {
	size_t lastSeparator = filePath.find_last_of("/\\");
	size_t lastDot = filePath.find_last_of(".");
	return filePath.substr(lastSeparator + 1, lastDot - lastSeparator - 1);
}

int main(int argc, char** argv)
{
	
	// cuda:0
	cudaSetDevice(0);
	int show = 1;

	const std::string engine_file_path = "D:/MinhTu/Project/FireDetection/yolov8/runs/detect/v6-yolov8m/weights/v6-yolov8m.8517.engine";

	// single img / multiple imgs inference
	//const std::string path = "D:/MinhTu/Project/FireDetection/dataset/High_Resolution/Images/S3-N0814MF02598.jpg";
	
	// video inference
	const std::string path = "D:/MinhTu/Project/FireDetection/test_video/fire/C055305_002_trimmed.mp4";

	std::cout << "TensorRT Engine version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;

	std::vector<std::string> imagePathList;
	bool isVideo{ false };


	auto yolov8 = new YOLOv8(engine_file_path);
	yolov8->make_pipe(true);

	if (IsFile(path))
	{
		std::string suffix = path.substr(path.find_last_of('.') + 1);
		if (
			suffix == "jpg" ||
			suffix == "jpeg" ||
			suffix == "png"
			)
		{
			imagePathList.push_back(path);
		}
		else if (
			suffix == "mp4" ||
			suffix == "avi" ||
			suffix == "m4v" ||
			suffix == "mpeg" ||
			suffix == "mov" ||
			suffix == "mkv"
			)
		{
			isVideo = true;
		}
		else
		{
			printf("suffix %s is wrong !!!\n", suffix.c_str());
			std::abort();
		}
	}
	else if (IsFolder(path))
	{
		cv::glob(path + "/*.jpg", imagePathList);
	}

	cv::Mat res, image;
	cv::Size size = cv::Size{640, 640 };

	std::vector<Object> objs;
	std::vector<Object> total_objs;

	//cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

	if (isVideo)
	{
		cv::VideoCapture cap(path);

		if (!cap.isOpened())
		{
			printf("can not open %s\n", path.c_str());
			return -1;
		}
		while (cap.read(image))
		{
			objs.clear();
			auto start = std::chrono::system_clock::now();
			yolov8->copy_from_Mat(image, size);
			yolov8->infer();
			yolov8->postprocess(objs, total_objs);
			yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
			auto end = std::chrono::system_clock::now();
			auto tc = (double)
				std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
			printf("cost %2.4lf ms\n", tc);
			if (show) {
				cv::imshow("result", res);
				if (cv::waitKey(10) == 'q')
				{
					break;
				}
			}
		}
	}
	else
	{
		for (auto& path : imagePathList)
		{
			objs.clear();
			image = cv::imread(path);
			std::string fileName = getBaseFileName(path);
			printf("Processing image %s\n", fileName.c_str());
			auto start = std::chrono::system_clock::now();
			yolov8->copy_from_Mat(image, size);
			yolov8->infer();
			yolov8->postprocess(objs, total_objs);
			yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
			auto end = std::chrono::system_clock::now();
			auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
			printf("cost %2.4lf ms\n", tc);
			cv::imshow("result", res);
			cv::waitKey(0);
			//fileName = "D:/MinhTu/Project/FireDetection/cpp/inf-cpp/" + fileName + ".jpg";
			//cv::imwrite(fileName, res);
			//printf("Saved inferenced image at %s\n", fileName.c_str());
		}	
		auto end = std::chrono::system_clock::now();
	}
	cv::destroyAllWindows();
	delete yolov8;

	return 0;
}