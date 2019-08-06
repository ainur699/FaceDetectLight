#pragma once
#include "ShapeModel.h"

///dlib
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"

/// OpenFace
#include "LandmarkCoreIncludes.h"



//-----------------------------------------------------------------------------
class TFaceRecognizer
{
public:
	TFaceRecognizer() {}
	TFaceRecognizer(std::string path);
	~TFaceRecognizer() {}

	bool Init(const std::string &path);
	bool DetectLandmarks(const cv::Mat &img, std::vector<cv::Point2f> &fshape, std::vector<cv::Point3d> &shape3d, cv::Vec6d &global);

protected:
	std::vector<dlib::rectangle> FaceDetectFast(const cv::Mat &img, bool fast = false);
	std::vector<cv::Point2f> GetPoints(const std::vector<cv::Point3d> &shape, const cv::Vec6d &global);

	std::vector<cv::Point3d> addExtraPoints(cv::Mat &shape);
	std::vector<cv::Point3d> addExtraPointsGenesis(cv::Mat &shape);

	void Alignment(const cv::Mat &img, dlib::rectangle rect, std::vector<cv::Point2f> &fshape, std::vector<cv::Point3d> &shape3d, cv::Vec6d &global);
	void Euler2Rot(cv::Mat &R, double pitch, double yaw, double roll);

private:
	Estimate3D								shape_estimate;
	dlib::frontal_face_detector				dlib_detector;
	dlib::shape_predictor					pose_model;
	LandmarkDetector::FaceModelParameters	parameters;
	LandmarkDetector::CLNF					clnf_model;
};


//-----------------------------------------------------------------------------