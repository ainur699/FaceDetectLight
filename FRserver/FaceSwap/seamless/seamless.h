#pragma once


#include <Sparse>


class Seamless
{
	int m_width;
	int m_height;

	cv::Mat1i m_index_buf;
	Eigen::SparseMatrix<double, Eigen::ColMajor> m_a;
	Eigen::VectorXd m_b[3], m_solution;

	cv::Mat m_dst_img;

public:
	enum {INTERIOR, EDGE_FIXED, EDGE_FLOAT};

	void Init(const cv::Mat3b &target, const cv::Mat3f &lapl, const cv::Mat1b &bounds_mask);
	cv::Mat3b Proceed(cv::Mat mask);

	static cv::Mat3f Laplacian(const cv::Mat3b &src);
	static cv::Mat3f LaplacianByMixGrad(const cv::Mat3b &src1, const cv::Mat3b &src2, const cv::Mat1b &mix_v);
};
