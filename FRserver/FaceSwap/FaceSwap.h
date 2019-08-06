#pragma once

#include <opencv2/opencv.hpp>
#include "seamless/seamless.h"

namespace FaceTools {
	class FaceSwap {
	public:
		enum ShapeType {
			Dlib_68,
			Dlib_extra_85,
			Dlib_Stasm_97
		};

		enum BlendType {
			BLEND,
			SEAMLESS
		};

	public:
		FaceSwap() {}
		~FaceSwap() {}

		void Swap(
			const cv::Mat donor, std::vector<cv::Point2d> donor_shape,
			cv::Mat &dst, std::vector<cv::Point2d> dst_shape,
			ShapeType shape_type = Dlib_extra_85, BlendType blend_type = SEAMLESS,
			bool filter = true);

		void DrawTriangles(
			cv::Mat img, std::vector<cv::Point2d> points,
			ShapeType type = Dlib_extra_85);

	public:
		int RadiusByPoints(const std::vector<cv::Point2d> &points, double R_perc);
		template<class Point> void Render(cv::Mat &img, const cv::Mat &tex, const std::vector<Point> &points, const std::vector<Point> &tex_coord, std::vector<int> triangles = face_triangles_DlibExtra85);
		bool SameSide(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3);
		void bilinInterp(const cv::Mat &I, double x, double y, unsigned char *dst);
		cv::Mat FaceMask(cv::Size img_size, const std::vector<cv::Point2d> &points, ShapeType shape_type = Dlib_extra_85);
		cv::Mat filter_mask(const cv::Mat &mask, int R);
		void blendInPlace(cv::Mat &dst, const cv::Mat &src, const cv::Mat &mask);

	private:
		Seamless seamless;

	public:
		const static std::vector<int> face_triangles_DlibExtra85;
		const static std::vector<int> face_triangles_DlibStasm97;
		const static std::vector<int> mouth_triangles;
		const static std::vector<int> face_triangles_DlibExtra68;
		const static std::vector<int> face_rotation;
	};


	template<class Point>
	void FaceSwap::Render(cv::Mat &img, const cv::Mat &tex, const std::vector<Point> &points, const std::vector<Point> &tex_coord, std::vector<int> triangles)
	{
		int n = triangles.size() / 3;
		for (int t = 0; t < n; t++)
		{
			int ix_a = triangles[3 * t];
			int ix_b = triangles[3 * t + 1];
			int ix_c = triangles[3 * t + 2];

			Point p_a = points[ix_a];
			Point p_b = points[ix_b];
			Point p_c = points[ix_c];
			double vz = (p_a.x - p_c.x) * (p_b.y - p_c.y) - (p_a.y - p_c.y) * (p_b.x - p_c.x);
			if (vz > 0)
			{
				Point t_a = tex_coord[ix_a];
				Point t_b = tex_coord[ix_b];
				Point t_c = tex_coord[ix_c];
				cv::Mat1d A(3, 3);
				A <<
					p_a.x, p_b.x, p_c.x,
					p_a.y, p_b.y, p_c.y,
					1, 1, 1;
				cv::Mat1d B(3, 3);
				B <<
					t_a.x, t_b.x, t_c.x,
					t_a.y, t_b.y, t_c.y,
					1, 1, 1;

				cv::Mat1d M = B * A.inv();
				double *affine = M.ptr<double>();

				int xmax = (int)std::ceil(std::max(std::max(p_a.x, p_b.x), p_c.x));
				int ymax = (int)std::ceil(std::max(std::max(p_a.y, p_b.y), p_c.y));
				int xmin = (int)std::floor(std::min(std::min(p_a.x, p_b.x), p_c.x));
				int ymin = (int)std::floor(std::min(std::min(p_a.y, p_b.y), p_c.y));
				if (xmax > img.cols - 1)
					xmax = img.cols - 1;
				if (ymax > img.rows - 1)
					ymax = img.rows - 1;
				if (xmin < 0)
					xmin = 0;
				if (ymin < 0)
					ymin = 0;
				for (int i = ymin; i <= ymax; i++)
				{
					unsigned char *dst_p = img.ptr(i, xmin);
					for (int j = xmin; j <= xmax; j++, dst_p += 3)
					{
						if (SameSide(j, i, p_a.x, p_a.y, p_b.x, p_b.y, p_c.x, p_c.y) &&
							SameSide(j, i, p_b.x, p_b.y, p_c.x, p_c.y, p_a.x, p_a.y) &&
							SameSide(j, i, p_c.x, p_c.y, p_a.x, p_a.y, p_b.x, p_b.y))
						{
							double x = affine[0] * j + affine[1] * i + affine[2];
							double y = affine[3] * j + affine[4] * i + affine[5];
							bilinInterp(tex, x, y, dst_p);
						}
					}
				}
			}
		}
	}
}