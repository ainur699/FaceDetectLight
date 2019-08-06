#include "FaceRecognizer.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

//-----------------------------------------------------------------------------
#define db(i, j) at<double>(i, j)
#define array_size(arr) (sizeof(arr) / sizeof(arr[0]))

//-----------------------------------------------------------------------------





//-----------------------------------------------------------------------------
TFaceRecognizer::TFaceRecognizer(std::string path)
{
	if (!Init(path)) throw std::runtime_error("Face detection error");
}


//-----------------------------------------------------------------------------
bool TFaceRecognizer::Init(const std::string &path)
{
	/// dlib detector
	dlib_detector = dlib::get_frontal_face_detector();

	/// dlib alignment
	try {
		dlib::deserialize(path +  "shape_predictor_68_face_landmarks.dat") >> pose_model;
	}
	catch (...) {
		return false;
	}

	/// OpenFace
	clnf_model.Read(path + "openface/main_clnf_general.txt");

	/// Shape estimate
	shape_estimate.m_pdm.Init(clnf_model.pdm.mean_shape, clnf_model.pdm.princ_comp, clnf_model.pdm.eigen_values);

	return true;
}


//-----------------------------------------------------------------------------
bool TFaceRecognizer::DetectLandmarks(const cv::Mat &img, std::vector<cv::Point2f> &fshape, std::vector<cv::Point3d> &shape3d, cv::Vec6d &global)
{
	bool face_found = false;
	if (img.type() != CV_8UC1) throw "Error: Incomplate imagetype";
		
	std::vector<dlib::rectangle> rects = FaceDetectFast(img, true);

	if (!rects.empty()) 
	{
		Alignment(img, rects[0], fshape, shape3d, global);
		face_found = true;
	}

	return face_found;
}


//-----------------------------------------------------------------------------
std::vector<dlib::rectangle> TFaceRecognizer::FaceDetectFast(const cv::Mat &in_img, bool fast)
{
	if (fast) {
		cv::Mat      rotImg;
		const int    iQuant = 128;
		const int    iSteps = 4;
		const int    iStart = iQuant * 2;
		const int    iStop = iQuant * iSteps;
		const double minS = cv::min(in_img.rows, in_img.cols);
		for (int i = iStart; i <= iStop; i += iQuant) if (minS > i)
		{
			const double k = i / minS;
			cv::resize(in_img, rotImg, cv::Size(in_img.cols * k, in_img.rows * k), 0.0, 0.0, CV_INTER_AREA);

			std::vector<dlib::rectangle> faces_dlib = dlib_detector(dlib::cv_image<uchar>(rotImg));
			if (faces_dlib.size() > 0)
			{
				faces_dlib[0].set_left(faces_dlib[0].left() / k);
				faces_dlib[0].set_right(faces_dlib[0].right() / k);
				faces_dlib[0].set_top(faces_dlib[0].top() / k);
				faces_dlib[0].set_bottom(faces_dlib[0].bottom() / k);
				return faces_dlib;
			}
		}
	}

	std::vector<dlib::rectangle> faces_dlib = dlib_detector(dlib::cv_image<uchar>(in_img));
	return faces_dlib;
}


//-----------------------------------------------------------------------------
void TFaceRecognizer::Alignment(const cv::Mat &img, dlib::rectangle rect, std::vector<cv::Point2f> &fshape, std::vector<cv::Point3d> &shape3d, cv::Vec6d &global)
{
	dlib::full_object_detection	shape_dlib = pose_model(dlib::cv_image<uchar>(img), rect);
	std::vector<cv::Point2d> shape;
	for (unsigned j = 0; j < shape_dlib.num_parts(); j++) shape.push_back(cv::Point(shape_dlib.part(j).x(), shape_dlib.part(j).y()));

	cv::Mat landmarks(2 * 68, 1, CV_64F);
	for (int i = 0; i < 68; i++)
	{
		landmarks.at<double>(i, 0)		= shape[i].x;
		landmarks.at<double>(i + 68, 0) = shape[i].y;
	}

	/// get local and global params
	cv::Mat local, global_mat, shape3d68;
	shape_estimate.estimate(landmarks, global_mat, local, shape3d68);
	///clnf_model.pdm.CalcParams(clnf_model.params_global, clnf_model.params_local, landmarks);

	clnf_model.params_local.setTo(0);
	clnf_model.params_global = global_mat;

	bool detection_success = clnf_model.DetectLandmarks(img, parameters);

	/// get points
	const cv::Mat_<double>& shape2D = clnf_model.detected_landmarks;
	shape_estimate.estimate(shape2D, global_mat, local, shape3d68);

	cv::Mat shape3d66(66 * 3, 1, CV_64F);
	for (int i = 0, j = 0; i < 68; i++) {
		if (i == 60 || i == 64) continue;
		shape3d66.at<double>(j) = shape3d68.at<double>(i);
		shape3d66.at<double>(j + 66) = shape3d68.at<double>(i + 68);
		shape3d66.at<double>(j + 2 * 66) = shape3d68.at<double>(i + 2 * 68);
		j++;
	}

	global = global_mat;
	shape3d = addExtraPoints(shape3d66);
	fshape = GetPoints(shape3d, global);
}


//-----------------------------------------------------------------------------

	



//-----------------------------------------------------------------------------
std::vector<cv::Point3d> TFaceRecognizer::addExtraPoints(cv::Mat &shape)
{
	int in_size = shape.rows / 3;
	std::vector<cv::Point3d> points(in_size);


	/// convert input to vector
	for (int i = 0; i < in_size; i++) {
		points[i].x = shape.db(i, 0);
		points[i].y = shape.db(i + in_size, 0);
		points[i].z = shape.db(i + 2 * in_size, 0);
	}


	/// add nose points
	const double k_1 = 0.2;
	const double k_2 = 1.5;
	const double k_3 = 0.65;
	const double k_4 = 0.65;
	cv::Point3d pt;

	/// #1
	pt.x = points[31].x + k_1 * (points[32].x - points[32].x);
	pt.y = (points[28].y + points[29].y) / 2;
	pt.z = points[31].z;
	points.push_back(pt);
	
	/// #2
	pt.x = points[31].x;
	pt.y = (points[29].y + points[30].y) / 2;
	pt.z = points[31].z;
	points.push_back(pt);
	
	/// #3
	pt.x = points[31].x - k_3 * (points[32].x - points[31].x);
	pt.y = k_4 * points[30].y + (1 - k_4) * points[33].y;
	pt.z = points[31].z;
	points.push_back(pt);

	/// #4
	pt.x = points[35].x + k_3 * (points[35].x - points[34].x);
	pt.y = k_4 * points[30].y + (1 - k_4) * points[33].y;
	pt.z = points[35].z;
	points.push_back(pt);

	/// #5
	pt.x = points[35].x - k_1 * (points[35].x - points[34].x);
	pt.y = (points[28].y + points[29].y) / 2;
	pt.z = points[35].z;
	points.push_back(pt);

	/// #6
	pt.x = points[35].x;
	pt.y = (points[29].y + points[30].y) / 2;
	pt.z = points[35].z;
	points.push_back(pt);


	/// add forehead points
	std::vector<uint> idx = { 2, 1, 0 };	/// indexs for calculationg coarse head contour points
	const double cb = 2.0 * (points[30].y - points[27].y);

	cv::Point3d forehead_top;
	forehead_top.x = points[27].x;
	forehead_top.y = points[27].y - (points[28].y - points[27].y) - cb;
	forehead_top.z = points[27].z + 2.5 * (points[27].z - points[30].z);

	cv::Point3d init_left = points[0];
	cv::Point3d init_right = points[16];
	cv::Point3d range_left = forehead_top - points[0];
	cv::Point3d range_right = forehead_top - points[16];

	const double      factors_x[] = { 0.3, 0.4, 0.5, 0.6, 0.8, 0.9 };
	const double      factors_y[] = { 1.5, 1.55, 1.5, 1.35, 1.27, 1.15 };
	const double      factors_z[] = { 1.5, 1.55, 1.5, 1.35, 1.27, 1.15 };
	const double      k = 7.0;
	if (array_size(factors_x) != (k - 1) || array_size(factors_y) != (k - 1) || array_size(factors_z) != (k - 1)) std::length_error("Bad factor count");

	for (int i = 1; i < k; i++) {
		cv::Point3d pt;
		pt.x = init_left.x + (i / k) * range_left.x * factors_x[i - 1];
		pt.y = init_left.y + (i / k) * range_left.y * factors_y[i - 1];
		pt.z = init_left.z + (i / k) * range_left.z * factors_z[i - 1];
		points.push_back(pt);
		idx.push_back(points.size() - 1);
	}
	points.push_back(forehead_top);
	idx.push_back(points.size() - 1);
	for (int i = k - 1; i > 0; i--) {
		cv::Point3d pt;
		pt.x = init_right.x + (i / k) * range_right.x * factors_x[i - 1];
		pt.y = init_right.y + (i / k) * range_right.y * factors_y[i - 1];
		pt.z = init_right.z + (i / k) * range_right.z * factors_z[i - 1];
		points.push_back(pt);
		idx.push_back(points.size() - 1);
	}
	idx.push_back(16);
	idx.push_back(15);
	idx.push_back(14);


	/// add head coarse contour
	double coeff[] = { 1.2, 1.3, 1.3, 1.45, 1.55, 1.6, 1.6, 1.6, 1.6,/**/1.6/**/, 1.6, 1.6, 1.6, 1.6, 1.55, 1.45, 1.3, 1.3, 1.2 };
	if (array_size(coeff) != 19) throw std::length_error("Bad coeff count");

	for (uint i = 0; i < idx.size(); i++){
		cv::Point3d pt = points[27] + coeff[i] * (points[idx[i]] - points[27]);
		points.push_back(pt);
	}

	/// adjust forehead poits
	forehead_top.x = points[27].x;
	forehead_top.y = points[27].y - (points[28].y - points[27].y) - cb;
	forehead_top.z = points[27].z + 1.8 * (points[27].z - points[30].z);

	range_left = forehead_top - points[0];
	range_right = forehead_top - points[16];

	for (int j =3, i = 1; j <= 8; i++, j++) {
		points[idx[j]].x = init_left.x + (i / k) * range_left.x * factors_x[i - 1];
		points[idx[j]].y = init_left.y + (i / k) * range_left.y * factors_y[i - 1];
		points[idx[j]].z = init_left.z + (i / k) * range_left.z * factors_z[i - 1];
	}
	points[idx[9]] = forehead_top;
	for (int j = 10, i = k - 1; j <= 15; i--, j++) {
		points[idx[j]].x = init_right.x + (i / k) * range_right.x * factors_x[i - 1];
		points[idx[j]].y = init_right.y + (i / k) * range_right.y * factors_y[i - 1];
		points[idx[j]].z = init_right.z + (i / k) * range_right.z * factors_z[i - 1];
	}

	return points;
}


//-----------------------------------------------------------------------------
std::vector<cv::Point3d> TFaceRecognizer::addExtraPointsGenesis(cv::Mat &shape)
{
	int in_size = shape.rows / 3;
	std::vector<cv::Point3d> points(66);

	/// convert input to vector
	for (int i = 0; i < 66; i++) {
		points[i].x = shape.db(i, 0);
		points[i].y = shape.db(i + in_size, 0);
		points[i].z = shape.db(i + 2 * in_size, 0);
	}


	/// add nose points
	const double k_1 = 0.2;
	const double k_2 = 1.5;
	const double k_3 = 0.65;
	const double k_4 = 0.65;
	cv::Point3d pt;

	/// #1
	pt.x = points[31].x + k_1 * (points[32].x - points[32].x);
	pt.y = (points[28].y + points[29].y) / 2;
	pt.z = points[31].z;
	points.push_back(pt);

	/// #2
	pt.x = points[31].x;
	pt.y = (points[29].y + points[30].y) / 2;
	pt.z = points[31].z;
	points.push_back(pt);

	/// #3
	pt.x = points[31].x - k_3 * (points[32].x - points[31].x);
	pt.y = k_4 * points[30].y + (1 - k_4) * points[33].y;
	pt.z = points[31].z;
	points.push_back(pt);

	/// #4
	pt.x = points[35].x + k_3 * (points[35].x - points[34].x);
	pt.y = k_4 * points[30].y + (1 - k_4) * points[33].y;
	pt.z = points[35].z;
	points.push_back(pt);

	/// #5
	pt.x = points[35].x - k_1 * (points[35].x - points[34].x);
	pt.y = (points[28].y + points[29].y) / 2;
	pt.z = points[35].z;
	points.push_back(pt);

	/// #6
	pt.x = points[35].x;
	pt.y = (points[29].y + points[30].y) / 2;
	pt.z = points[35].z;
	points.push_back(pt);


	/// add forehead points
	std::vector<uint> idx = { 2, 1, 0 };	/// indexs for calculationg coarse head contour points
	const double cb = 60.0;

	cv::Point3d forehead_top;
	forehead_top.x = points[27].x;
	forehead_top.y = points[27].y - 60.0;
	forehead_top.z = points[27].z + 35.0;

	cv::Point3d init_left = points[0];
	cv::Point3d init_right = points[16];
	cv::Point3d range_left = forehead_top - init_left;
	cv::Point3d range_right = forehead_top - init_right;

	const std::vector<double> factors_x = { 0.3, 0.4, 0.5, 0.6, 0.8, 0.9 };
	const std::vector<double> factors_y = { 1.5, 1.55, 1.5, 1.35, 1.27, 1.15 };
	const std::vector<double> factors_z = { 1.5, 1.55, 1.5, 1.35, 1.27, 1.15 };
	const double k = 7.0;
	if (factors_x.size() != (k - 1) || factors_y.size() != (k - 1) || factors_z.size() != (k - 1)) throw std::length_error("Bad factor count");

	for (int i = 1; i < k; i++) {
		cv::Point3d pt;
		pt.x = init_left.x + (i / k) * range_left.x * factors_x[i - 1];
		pt.y = init_left.y + (i / k) * range_left.y * factors_y[i - 1];
		pt.z = init_left.z + (i / k) * range_left.z * factors_z[i - 1];
		points.push_back(pt);
		idx.push_back(points.size() - 1);
	}
	points.push_back(forehead_top);
	idx.push_back(points.size() - 1);
	for (int i = k - 1; i > 0; i--) {
		cv::Point3d pt;
		pt.x = init_right.x + (i / k) * range_right.x * factors_x[i - 1];
		pt.y = init_right.y + (i / k) * range_right.y * factors_y[i - 1];
		pt.z = init_right.z + (i / k) * range_right.z * factors_z[i - 1];
		points.push_back(pt);
		idx.push_back(points.size() - 1);
	}
	idx.push_back(16);
	idx.push_back(15);
	idx.push_back(14);


	/// add head coarse contour
	const std::vector<double> coeff = { 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.6,/**/1.6/**/, 1.6, 1.6, 1.5, 1.4, 1.35, 1.3, 1.25, 1.2, 1.15 };
	if (coeff.size() != 19) throw std::length_error("Bad coeff count");

	for (uint i = 0; i < idx.size(); i++) {
		cv::Point3d pt = points[27] + coeff[i] * (points[idx[i]] - points[27]);
		points.push_back(pt);
	}

	/// adjust forehead poits
	forehead_top.x = points[27].x;
	forehead_top.y = points[27].y - cb;
	forehead_top.z = points[27].z + 20.0;

	range_left = forehead_top - points[0];
	range_right = forehead_top - points[16];

	for (int j = 3, i = 1; j <= 8; i++, j++) {
		points[idx[j]].x = init_left.x + (i / k) * range_left.x * factors_x[i - 1];
		points[idx[j]].y = init_left.y + (i / k) * range_left.y * factors_y[i - 1];
		points[idx[j]].z = init_left.z + (i / k) * range_left.z * factors_z[i - 1];
	}
	points[idx[9]] = forehead_top;
	for (int j = 10, i = k - 1; j <= 15; i--, j++) {
		points[idx[j]].x = init_right.x + (i / k) * range_right.x * factors_x[i - 1];
		points[idx[j]].y = init_right.y + (i / k) * range_right.y * factors_y[i - 1];
		points[idx[j]].z = init_right.z + (i / k) * range_right.z * factors_z[i - 1];
	}

	return points;
}


//-----------------------------------------------------------------------------
std::vector<cv::Point2f> TFaceRecognizer::GetPoints(const std::vector<cv::Point3d> &shape, const cv::Vec6d &global)
{
	cv::Mat R;
	Euler2Rot(R, global[1], global[2], global[3]);

	std::vector<cv::Point2f> vertex(shape.size());
	const double a = global[0];
	const double x = global[4];
	const double y = global[5];
	for (int i = 0; i < shape.size(); i++)
	{
		vertex[i].x = a * (R.db(0, 0) * shape[i].x + R.db(0, 1) * shape[i].y + R.db(0, 2) * shape[i].z) + x;
		vertex[i].y = a * (R.db(1, 0) * shape[i].x + R.db(1, 1) * shape[i].y + R.db(1, 2) * shape[i].z) + y;
	}

	return vertex;
}


//-----------------------------------------------------------------------------
void TFaceRecognizer::Euler2Rot(cv::Mat &R, double pitch, double yaw, double roll)
{
	R.create(3, 3, CV_64F);

	const double sina = sin(pitch), sinb = sin(yaw), sinc = sin(roll);
	const double cosa = cos(pitch), cosb = cos(yaw), cosc = cos(roll);

	R.db(0, 0) =  cosb * cosc;
	R.db(0, 1) = -cosb * sinc;
	R.db(0, 2) =  sinb;
	R.db(1, 0) =  cosa * sinc + sina * sinb * cosc;
	R.db(1, 1) =  cosa * cosc - sina * sinb * sinc;
	R.db(1, 2) = -sina * cosb;

	R.db(2, 0) = R.db(0, 1) * R.db(1, 2) - R.db(0, 2) * R.db(1, 1);
	R.db(2, 1) = R.db(0, 2) * R.db(1, 0) - R.db(0, 0) * R.db(1, 2);
	R.db(2, 2) = R.db(0, 0) * R.db(1, 1) - R.db(0, 1) * R.db(1, 0);
}


//-----------------------------------------------------------------------------