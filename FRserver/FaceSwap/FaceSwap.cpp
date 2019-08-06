#include "FaceSwap.h"

namespace FaceTools {

	void FaceSwap::Swap(
		const cv::Mat donor, std::vector<cv::Point2d> donor_shape,
		cv::Mat &dst, std::vector<cv::Point2d> dst_shape,
		ShapeType shape_type, BlendType blend_type, bool filter)
	{
		cv::Mat img_text(dst.size(), CV_8UC3);

		switch (shape_type) {
		case Dlib_extra_85: {
			Render(img_text, donor, dst_shape, donor_shape, face_triangles_DlibExtra85);
			break;
		}
		case Dlib_Stasm_97: {
			Render(img_text, donor, dst_shape, donor_shape, face_triangles_DlibStasm97);
			break;
		}
		case Dlib_68: {
			Render(img_text, donor, dst_shape, donor_shape, face_triangles_DlibExtra68);
			break;
		}
		default: {throw "Error: wrong shape type"; }
		}

		// create mask
		int R = RadiusByPoints(dst_shape, 0.03);

		cv::Mat temp = FaceMask(dst.size(), dst_shape, shape_type);
		cv::Mat mask;
		if (filter) mask = filter_mask(temp, R);
		else mask = temp;

		// blendind
		switch (blend_type) {
		case BLEND: {
			cv::Point2d p1_m = dst_shape[64], p2_m = dst_shape[61];
			cv::Point2d p3_m = dst_shape[57], p4_m = dst_shape[51];
			double len1 = hypot(p1_m.x - p2_m.x, p1_m.y - p2_m.y);
			double len2 = hypot(p3_m.x - p4_m.x, p3_m.y - p4_m.y);

			cv::Rect bound;
			std::vector<std::vector<cv::Point> > contour(1);
			std::vector<cv::Point> &v = contour[0];

			if (len1 > len2 * 0.05) // open mouth
			{
				int idx[] = { 60, 61, 62, 54, 63, 64, 65, 48 };
				v.clear();
				for (int i = 0; i < _countof(idx); i++)
				{
					cv::Point p = dst_shape[idx[i]];
					v.push_back(cv::Point((int)p.x, (int)p.y));
				}
				cv::drawContours(mask, contour, 0, cv::Scalar::all(0), -1);
				bound = cv::boundingRect(v);
				cv::Mat temp_mask = mask(bound & cv::Rect(0, 0, mask.cols, mask.rows));
				cv::blur(temp_mask, temp_mask, cv::Size(0.5 * R, 0.5 * R));

				//cv::drawContours(dst, contour, 0, cv::Scalar::all(255), -1);
			}
			blendInPlace(dst, img_text, mask);
			break;
		}
		case SEAMLESS: {

			cv::Mat1b bounds_mask(mask.size());
			unsigned char *p0 = mask.ptr();
			unsigned char *p1 = bounds_mask.ptr();

			for (int i = 0; i < bounds_mask.rows; i++)
			{
				for (int j = 0; j < bounds_mask.cols; j++) {
					*p1 = *p0 > 0 ? Seamless::INTERIOR : Seamless::EDGE_FIXED;
					p0++;
					p1++;
				}
			}

			std::vector<std::vector<cv::Point> > contour(1);
			std::vector<cv::Point> &v = contour[0];
			cv::Rect bound;

			seamless.Init(dst, seamless.LaplacianByMixGrad(dst, img_text, mask), bounds_mask);

			cv::Point2d p1_m = dst_shape[64], p2_m = dst_shape[61];
			cv::Point2d p3_m = dst_shape[57], p4_m = dst_shape[51];
			double len1 = hypot(p1_m.x - p2_m.x, p1_m.y - p2_m.y);
			double len2 = hypot(p3_m.x - p4_m.x, p3_m.y - p4_m.y);

			mask.setTo(255);
			if (len1 > len2 * 0.1) // open mouth
			{
				int idx[] = { 60, 61, 62, 54, 63, 64, 65, 48 };
				v.clear();
				for (int i = 0; i < _countof(idx); i++)
				{
					cv::Point p = dst_shape[idx[i]];
					v.push_back(cv::Point((int)p.x, (int)p.y));
				}
				cv::drawContours(mask, contour, 0, cv::Scalar::all(0), -1);
				bound = cv::boundingRect(v);
				cv::Mat temp_mask = mask(bound & cv::Rect(0, 0, mask.cols, mask.rows));
				cv::blur(temp_mask, temp_mask, cv::Size(0.5 * R, 0.5 * R));
			}

			dst = seamless.Proceed(mask);
			break;
		}
		}
	}

	void FaceSwap::DrawTriangles(
		cv::Mat img, std::vector<cv::Point2d> points,
		ShapeType type)
	{
		const std::vector<int> *triangles = nullptr;
		switch (type) {
		case Dlib_extra_85: {
			triangles = &face_triangles_DlibExtra85;
			break;
		}
		case Dlib_Stasm_97: {
			triangles = &face_triangles_DlibStasm97;
			break;
		}
		}

		int n = triangles->size() / 3;
		for (int t = 0; t < n; t++)
		{
			int ix_a = (*triangles)[3 * t];
			int ix_b = (*triangles)[3 * t + 1];
			int ix_c = (*triangles)[3 * t + 2];
			cv::Point2d p_a = points[ix_a];
			cv::Point2d p_b = points[ix_b];
			cv::Point2d p_c = points[ix_c];
			cv::line(img, cv::Point(p_a.x, p_a.y), cv::Point(p_b.x, p_b.y), { 0, 255, 0 }, 1);
			cv::line(img, cv::Point(p_a.x, p_a.y), cv::Point(p_c.x, p_c.y), { 0, 255, 0 }, 1);
			cv::line(img, cv::Point(p_b.x, p_b.y), cv::Point(p_c.x, p_c.y), { 0, 255, 0 }, 1);
		}
	}

	void FaceSwap::blendInPlace(cv::Mat &dst, const cv::Mat &src, const cv::Mat &mask)
	{
		assert(dst.size() == src.size() && dst.channels() == 3 && src.channels() == 3 && mask.size() == dst.size() && mask.channels() == 1);

		for (int i = 0; i < mask.rows; i++)
		{
			unsigned char *p_dst = dst.ptr(i);
			const unsigned char *p_src = src.ptr(i);
			const unsigned char *p_mask = mask.ptr(i);
			for (int j = 0; j < mask.cols; j++)
			{
				int b = *p_mask;
				for (int k = 0; k < 3; k++, p_src++, p_dst++)
				{
					*p_dst = (int)*p_dst + ((((int)*p_src - (int)*p_dst) * b) >> 8);
				}
				p_mask++;
			}
		}
	}

	int FaceSwap::RadiusByPoints(const std::vector<cv::Point2d> &points, double R_perc)
	{
		cv::Mat1d p(points.size(), 2, (double*)&points[0]);

		double min_x = 0, min_y = 0, max_x = 0, max_y = 0;
		cv::minMaxLoc(p.col(0), &min_x, &max_x);
		cv::minMaxLoc(p.col(1), &min_y, &max_y);

		int R = std::max(1, (int)(R_perc * std::max(max_x - min_x, max_y - min_y)));

		return R;
	}

	bool FaceSwap::SameSide(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3)
	{
		double x = (x3 - x2)*(y0 - y2) - (x0 - x2)*(y3 - y2);
		double y = (x3 - x2)*(y1 - y2) - (x1 - x2)*(y3 - y2);
		return x*y >= 0;
	}

	void FaceSwap::bilinInterp(const cv::Mat &I, double x, double y, unsigned char *dst)
	{
		int x1 = (int)std::floor(x);
		int y1 = (int)std::floor(y);
		int x2 = (int)std::ceil(x);
		int y2 = (int)std::ceil(y);

		if (x1 < 0 || x2 >= I.cols || y1 < 0 || y2 >= I.rows)
			return;

		const unsigned char *p1 = I.ptr(y1, x1);
		const unsigned char *p2 = I.ptr(y1, x2);
		const unsigned char *p3 = I.ptr(y2, x1);
		const unsigned char *p4 = I.ptr(y2, x2);
		for (int i = 0; i < 3; i++)
		{
			double c1 = p1[i] + ((double)p2[i] - p1[i]) * (x - x1);
			double c2 = p3[i] + ((double)p4[i] - p3[i]) * (x - x1);
			*dst++ = (unsigned char)(c1 + (c2 - c1) * (y - y1));
		}
	}

	
	cv::Mat FaceSwap::FaceMask(cv::Size img_size, const std::vector<cv::Point2d> &points, ShapeType shape_type)
	{
		cv::Mat mask(img_size, CV_8UC1);
		mask = 0;
		std::vector<std::vector<cv::Point> > contour(1);
		std::vector<cv::Point> &v = contour[0];

		// контур лица
		switch (shape_type) {
		case Dlib_68: 
		{
			for (int i = 0; i <= 16; i++)
			{
				v.push_back(cv::Point((int)points[i].x, (int)points[i].y));
			}

			for (int i = 26; i >= 17; i--)
			{
				v.push_back(cv::Point((int)points[i].x, (int)points[i].y));
			}

			break;
		}
		case Dlib_Stasm_97:
		case Dlib_extra_85: 
		{
			for (int i = 0; i <= 16; i++)
			{
				v.push_back(cv::Point((int)points[i].x, (int)points[i].y));
			}

			for (int i = 70; i >= 66; i--)
			{
				v.push_back(cv::Point((int)points[i].x, (int)points[i].y));
			}
			break;
		}
		}

		cv::drawContours(mask, contour, 0, cv::Scalar::all(255), -1);

		return mask;
	}

	cv::Mat FaceSwap::filter_mask(const cv::Mat &mask, int R)
	{
		cv::Size sz(2 * R + 1, 2 * R + 1);
		cv::Mat f;
		cv::erode(mask, f, cv::getStructuringElement(cv::MORPH_RECT, sz));
		cv::boxFilter(f, f, CV_8UC1, sz);

		return f;
	}

	const std::vector<int> FaceSwap::face_triangles_DlibExtra85 = {
		///contour of face
		0, 36, 1,
		1, 36, 41,
		1, 41, 2,
		2, 41, 40,
		2, 40, 79,
		2, 79, 80,
		2, 80, 3,
		3, 80, 81,
		3, 81, 4,
		4, 81, 48,
		4, 48, 5,
		5, 48, 6,
		6, 48, 59,
		6, 59, 7,
		7, 59, 58,
		7, 58, 8,
		8, 58, 57,
		8, 57, 56,
		8, 56, 9,
		9, 56, 55,
		9, 55, 10,
		10, 55, 54,
		10, 54, 11,
		11, 54, 12,
		12, 54, 82,
		12, 82, 13,
		13, 82, 83,
		13, 83, 14,
		14, 83, 84,
		14, 84, 47,
		14, 47, 46,
		14, 46, 15,
		15, 46, 45,
		15, 45, 16,

		/// eye and eyebrow right
		16, 45, 78,
		16, 78, 26,
		26, 78, 77,
		26, 77, 25,
		25, 77, 76,
		25, 76, 24,
		24, 76, 23,
		23, 76, 75,
		23, 75, 22,
		78, 45, 44,
		78, 44, 77,
		77, 44, 43,
		77, 43, 76,
		76, 43, 42,
		76, 42, 75,

		/// eye and eyebrow left
		20, 21, 74,
		20, 74, 73,
		20, 73, 19,
		19, 73, 18,
		18, 73, 72,
		18, 72, 17,
		17, 72, 71,
		17, 71, 0,
		71, 36, 0,
		73, 74, 39,
		73, 39, 38,
		72, 73, 38,
		72, 38, 37,
		71, 72, 37,
		71, 37, 36,

		///	midl nose
		74, 21, 27,
		39, 74, 27,
		27, 21, 22,
		27, 22, 75,
		27, 75, 42,

		28, 39, 27,
		28, 27, 42,

		/// nose
		39, 27, 28,
		40, 39, 79,
		39, 28, 79,
		79, 29, 80,
		28, 29, 79,
		80, 29, 30,
		80, 30, 81,
		30, 32, 81,
		30, 33, 32,

		30, 34, 33,
		30, 82, 34,
		83, 82, 30,
		29, 83, 30,
		84, 83, 29,
		28, 84, 29,
		42, 47, 84,
		42, 84, 28,
		28, 42, 27,

		///between nose and mouth
		81, 49, 48,
		81, 32, 49,
		32, 50, 49,
		32, 33, 50,
		33, 51, 50,

		33, 52, 51,
		33, 34, 52,
		34, 53, 52,
		34, 82, 53,
		82, 54, 53,

		/// mouth
		48, 49, 60,
		60, 49, 50,
		60, 50, 61,
		61, 50, 51,
		61, 51, 52,
		61, 52, 62,
		62, 52, 53,
		62, 53, 54,

		48, 65, 59,
		59, 65, 58,
		58, 65, 57,
		57, 65, 64,
		57, 64, 56,
		56, 64, 63,
		56, 63, 55,
		55, 63, 54,

		/// forehead
		0, 66, 17,
		66, 18, 17,
		66, 19, 18,
		19, 66, 67,
		67, 20, 19,
		67, 21, 20,
		21, 67, 68,
		21, 68, 22,
		22, 68, 69,
		22, 69, 23,
		23, 69, 24,
		24, 69, 70,
		24, 70, 25,
		25, 70, 26,
		26, 70, 16,
		36, 37, 41,
		41, 37, 38,
		41, 38, 40,
		40, 38, 39,
		42, 43, 47,
		47, 43, 44,
		47, 44, 46,
		46, 44, 45,
		48, 60, 65,
		65, 60, 64,
		64, 60, 61,
		64, 61, 62,
		64, 62, 63,
		63, 62, 54
	};

	const std::vector<int> FaceSwap::mouth_triangles = {
		48, 60, 65,
		65, 60, 64,
		64, 60, 61,
		64, 61, 62,
		64, 62, 63,
		63, 62, 54
	};

	const std::vector<int> FaceSwap::face_triangles_DlibStasm97 = {
		///contour of face
		0, 36, 1,
		1, 36, 41,
		1, 41, 2,
		2, 41, 40,
		2, 40, 79,
		2, 79, 80,
		2, 80, 3,
		3, 80, 81,
		3, 81, 4,
		4, 81, 48,
		4, 48, 5,
		5, 48, 6,
		6, 48, 59,
		6, 59, 7,
		7, 59, 58,
		7, 58, 8,
		8, 58, 57,
		8, 57, 56,
		8, 56, 9,
		9, 56, 55,
		9, 55, 10,
		10, 55, 54,
		10, 54, 11,
		11, 54, 12,
		12, 54, 82,
		12, 82, 13,
		13, 82, 83,
		13, 83, 14,
		14, 83, 84,
		14, 84, 47,
		14, 47, 46,
		14, 46, 15,
		15, 46, 45,
		15, 45, 16,

		/// eyebrow right
		92, 96, 91,
		96, 92, 93,
		93, 95, 96,
		95, 93, 94,

		/// berween right eye and eyebrow
		96, 43, 91,
		43, 96, 44,
		44, 96, 95,
		95, 45, 44,
		45, 95, 94,
		45, 94, 16,

		/// eyebrow left
		86, 88, 87,
		88, 86, 89,
		85, 89, 86,
		89, 85, 90,

		/// between left eye and eyebrow
		0, 87, 36,
		88, 36, 87,
		36, 88, 37,
		37, 88, 89,
		37, 89, 38,
		38, 89, 90,

		///	circle area around 27
		27, 28, 39,
		39, 38, 27,
		38, 90, 27,
		27,90, 91,
		91, 43, 27,
		43, 42, 27,
		28, 27, 42,

		/// nose
		39, 27, 28,
		40, 39, 79,
		39, 28, 79,
		79, 29, 80,
		28, 29, 79,
		80, 29, 30,
		80, 30, 81,
		30, 32, 81,
		30, 33, 32,

		30, 34, 33,
		30, 82, 34,
		83, 82, 30,
		29, 83, 30,
		84, 83, 29,
		28, 84, 29,
		42, 47, 84,
		42, 84, 28,
		28, 42, 27,

		///between nose and mouth
		81, 49, 48,
		81, 32, 49,
		32, 50, 49,
		32, 33, 50,
		33, 51, 50,

		33, 52, 51,
		33, 34, 52,
		34, 53, 52,
		34, 82, 53,
		82, 54, 53,

		/// mouth
		48, 49, 60,
		60, 49, 50,
		60, 50, 61,
		61, 50, 51,
		61, 51, 52,
		61, 52, 62,
		62, 52, 53,
		62, 53, 54,

		48, 65, 59,
		59, 65, 58,
		58, 65, 57,
		57, 65, 64,
		57, 64, 56,
		56, 64, 63,
		56, 63, 55,
		55, 63, 54,

		/// forehead
		66, 87, 0,
		87, 66, 86,
		86, 66, 67,
		67, 85, 86,
		85, 67, 68,
		68, 90, 85,
		68, 91, 90,
		91, 68, 92,
		69, 92, 68,
		92, 69, 93,
		93, 69, 70,
		70, 94, 93,
		70, 16, 94,

		/// inside eyes
		36, 37, 41,
		41, 37, 38,
		41, 38, 40,
		40, 38, 39,
		42, 43, 47,
		47, 43, 44,
		47, 44, 46,
		46, 44, 45,
	};

	const std::vector<int> FaceSwap::face_triangles_DlibExtra68 = {
		0, 36, 1,
		1, 36, 41,
		1, 41, 2,
		2, 41, 40,
		2, 40, 29,
		2, 29, 3,
		3, 29, 31,
		3, 31, 4,
		4, 31, 48,
		4, 48, 5,
		5, 48, 6,
		6, 48, 59,
		6, 59, 7,
		7, 59, 58,
		7, 58, 8,
		8, 58, 57,
		8, 57, 56,
		8, 56, 9,
		9, 56, 55,
		9, 55, 10,
		10, 55, 54,
		10, 54, 11,
		11, 54, 12,
		12, 54, 35,
		12, 35, 13,
		13, 35, 29,
		13, 29, 14,
		14, 29, 47,
		14, 47, 46,
		14, 46, 15,
		15, 46, 45,
		15, 45, 16,
		16, 45, 26,
		26, 45, 44,
		26, 44, 25,
		25, 44, 43,
		25, 43, 24,
		24, 43, 23,
		23, 43, 42,
		23, 42, 22,
		//	23, 22, 21,
		//	23, 21, 20,
		20, 21, 39,
		20, 39, 38,
		20, 38, 19,
		19, 38, 18,
		18, 38, 37,
		18, 37, 17,
		17, 37, 36,
		17, 36, 0,

		39, 21, 27,
		27, 21, 22,
		27, 22, 42,
		40, 39, 28,
		28, 39, 27,
		28, 27, 42,
		28, 42, 47,
		29, 40, 28,
		29, 28, 47,

		31, 29, 30,
		30, 29, 35,
		31, 30, 32,
		32, 30, 33,
		33, 30, 34,
		34, 30, 35,
		48, 31, 49,
		49, 31, 50,
		50, 31, 32,
		50, 32, 33,
		50, 33, 51,
		51, 33, 52,
		52, 33, 34,
		52, 34, 35,
		52, 35, 53,
		53, 35, 54,

		48, 49, 60,
		60, 49, 50,
		60, 50, 61,
		61, 50, 51,
		61, 51, 52,
		61, 52, 62,
		62, 52, 53,
		62, 53, 54,

		48, 65, 59,
		59, 65, 58,
		58, 65, 57,
		57, 65, 64,
		57, 64, 56,
		56, 64, 63,
		56, 63, 55,
		55, 63, 54,

		36, 37, 41,
		41, 37, 38,
		41, 38, 40,
		40, 38, 39,

		42, 43, 47,
		47, 43, 44,
		47, 44, 46,
		46, 44, 45,

		48, 60, 65,
		65, 60, 64,
		64, 60, 61,
		64, 61, 62,
		64, 62, 63,
		63, 62, 54
	};

	const std::vector<int> FaceSwap::face_rotation =
	{
		/// background
		90, 111, 91,
		91, 111, 92,
		92, 111, 112,
		92, 112, 93,
		93, 112, 94,
		94, 112, 95,
		95, 112, 96,
		96, 112, 113,
		97, 96, 113,
		98, 97, 113,
		98, 113, 114,
		99, 98, 114,
		100, 99, 114,
		101, 100, 114,
		102, 101, 114,
		115, 102, 114,
		115, 103, 102,
		115, 13, 103,
		115, 12, 13,
		110, 12, 115,
		116, 110, 115,
		109, 110, 116,
		117, 109, 116,
		117, 108, 109,
		117, 107, 108,
		117, 106, 107,
		117, 105, 106,
		118, 105, 117,
		118, 104, 105,
		118, 119, 104,
		119, 4, 104,
		119, 3, 4,
		119, 85, 3,
		119, 86, 85,
		119, 120, 86,
		120, 87, 86,
		120, 88, 87,
		88, 120, 89,
		120, 90, 89,
		90, 120, 111,

		/// neck
		104, 4, 5,
		104, 5, 6,
		105, 104, 6,
		105, 6, 7,
		106, 105, 7,
		106, 7, 8,
		106, 8, 107,
		107, 8, 108,
		108, 8, 9,
		108, 9, 109,
		109, 9, 10,
		109, 10, 110,
		110, 10, 11,
		110, 11, 12,

		/// coarse hair
		3, 85, 2,
		85, 86, 2,
		2, 86, 1,
		1, 86, 87,
		1, 87, 0,
		0, 87, 72,
		87, 88, 72,
		72, 88, 73,
		73, 88, 89,
		73, 88, 89,
		73, 89, 74,
		89, 90, 74,
		74, 90, 75,
		75, 90, 91,
		75, 91, 76,
		76, 91, 92,
		76, 92, 77,
		77, 92, 93,
		77, 93, 94,
		77, 94, 78,
		78, 94, 79,
		79, 94, 95,
		79, 95, 96,
		80, 79, 96,
		80, 96, 97,
		81, 80, 97,
		81, 97, 98,
		82, 81, 98,
		82, 98, 99,
		83, 82, 99,
		100, 83, 99,
		84, 83, 100,
		101, 84, 100,
		16, 84, 101,
		15, 16, 101,
		102, 15, 101,
		14, 15, 102,
		103, 14, 102,
		13, 14, 103,

		/// face contour
		0, 18, 36,
		1, 0, 36,
		1, 36, 41,
		2, 1, 41,
		2, 41, 68,
		3, 2, 68,
		3, 68, 48,
		4, 3, 48,
		5, 4, 48,
		5, 48, 59,
		6, 5, 59,
		6, 59, 58,
		7, 6, 58,
		7, 58, 57,
		7, 57, 8,
		8, 57, 9,
		57, 56, 9,
		9, 56, 10,
		10, 56, 55,
		10, 55, 11,
		11, 55, 54,
		11, 54, 12,
		12, 54, 13,
		54, 69, 13,
		13, 69, 14,
		69, 46, 14,
		14, 46, 15,
		15, 46, 45,
		15, 45, 16,
		16, 45, 25,

		/// forehead
		16, 25, 84,
		25, 83, 84,
		25, 24, 83,
		24, 82, 83,
		24, 81, 82,
		24, 80, 81,
		23, 80, 24,
		23, 79, 80,
		22, 79, 23,
		22, 78, 79,
		21, 78, 22,
		21, 77, 78,
		20, 77, 21,
		20, 76, 77,
		19, 76, 20,
		19, 75, 76,
		19, 74, 75,
		19, 73, 74,
		18, 73, 19,
		72, 73, 18,
		0, 72, 18,

		/// left eyebrow and eye
		36, 18, 37,
		37, 18, 19,
		37, 19, 20,
		37, 20, 38,
		38, 20, 21,
		39, 38, 21,
		40, 38, 39,
		40, 41, 38,
		41, 37, 38,
		41, 36, 37,

		///right eyebrow and eye
		42, 22, 43,
		43, 22, 23,
		43, 23, 44,
		44, 23, 24,
		44, 24, 25,
		45, 44, 25,
		46, 44, 45,
		46, 43, 44,
		47, 43, 46,
		47, 42, 43,

		/// between left eye and nose
		68, 41, 67,
		67, 41, 40,
		67, 40, 66,
		66, 40, 39,

		/// between right eye and nose
		69, 71, 46,
		71, 47, 46,
		71, 70, 47,
		70, 42, 47,

		///space between nose and mouth

		48, 68, 31,
		48, 31, 49,
		49, 31, 50,
		50, 31, 32,
		50, 32, 33,
		51, 50, 33,
		52, 51, 33,
		52, 33, 34,
		52, 34, 35,
		53, 52, 35,
		54, 53, 35,
		54, 35, 69,

		/// lips
		60, 48, 49,
		60, 49, 50,
		60, 50, 51,
		51, 61, 60,
		62, 61, 51,
		62, 51, 52,
		53, 62, 52,
		54, 62, 53,
		54, 55, 63,
		55, 56, 63,
		56, 57, 63,
		57, 64, 63,
		57, 65, 64,
		57, 58, 65,
		65, 58, 59,
		65, 59, 48,

		/// mouth interior
		48, 60, 65,
		65, 60, 64,
		64, 60, 61,
		64, 61, 62,
		64, 62, 63,
		63, 62, 54,

		/// nose
		39, 21, 27,
		27, 21, 22,
		27, 22, 42,
		28, 39, 27,
		28, 27, 42,
		66, 39, 28,
		70, 28, 42,
		29, 66, 28,
		29, 28, 70,
		67, 66, 29,
		71, 29, 70,
		30, 67, 29,
		30, 29, 71,
		68, 67, 30,
		69, 30, 71,
		31, 68, 30,
		32, 31, 30,
		33, 32, 30,
		34, 33, 30,
		35, 34, 30,
		69, 35, 30,
	};
}