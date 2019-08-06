#include "FaceRecognizer.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


int main(int argc, char ** argv)
{
	TFaceRecognizer rec;
	rec.Init("resFR/");
	
	cv::Mat img = cv::imread("image.jpg");
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	std::vector<cv::Point2f> landmarks;
	std::vector<cv::Point3d> shape;
	cv::Vec6d global;
	rec.DetectLandmarks(gray, landmarks, shape, global);

	for (size_t i = 0; i < landmarks.size(); i++) {
		cv::circle(img, landmarks[i], 3, cv::Scalar(255, 0, 0), -1);
	}

	cv::imwrite("dst.png", img);

	return 0;
}