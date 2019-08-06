#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#pragma warning(disable:4018)

#include "seamless.h"

using namespace cv;

static Mat3f mix_filter(const Mat3b &img0, const Mat3b &img1, const Mat &kern, const Mat1f &mix)
{
	Mat3f img0_d, img1_d;
	cv::filter2D(img0, img0_d, CV_32FC3, kern, cv::Point(0, 0), 0, BORDER_REFLECT);
	cv::filter2D(img1, img1_d, CV_32FC3, kern, cv::Point(0, 0), 0, BORDER_REFLECT);
	float *p0 = img0_d.ptr<float>();
	float *p1 = img1_d.ptr<float>();
	const float *p_mix = mix.ptr<float>();
	for (int i = img0_d.rows * img0_d.cols; i--;)
	{
		for (int i = 0; i < 3; i++)
		{
			*p0 += (*p1 - *p0) * *p_mix;
			p0++;
			p1++;
		}
		p_mix++;
	}

	return img0_d;
}

static Mat3f mix_grad(const Mat3b &img0, const Mat3b &img1, const Mat1b &mix_v)
{
	Mat3f img_dx, img_dy;
	Mat1f mix(mix_v);
	mix *= 1.0 / 255.0;

	cv::Mat1f kern_x(1, 2);
		kern_x << -1, 1;
	cv::Mat1f kern_y(2, 1);
		kern_y << -1, 1;

	img_dx = mix_filter(img0, img1, kern_x, mix);
	img_dy = mix_filter(img0, img1, kern_y, mix);

	filter2D(img_dx, img_dx, CV_32FC3, kern_x, Point(1, 0), 0, BORDER_REFLECT);
	filter2D(img_dy, img_dy, CV_32FC3, kern_y, Point(0, 1), 0, BORDER_REFLECT);

	return img_dx + img_dy;
}

void Seamless::Init(const cv::Mat3b &target, const cv::Mat3f &lapl, const cv::Mat1b &bounds_mask)
{
	m_dst_img = target.clone();
	m_width  = bounds_mask.cols;
	m_height = bounds_mask.rows;

	cv::Mat bounds_mask_ext;
	cv::copyMakeBorder(bounds_mask, bounds_mask_ext, 1, 1, 1, 1, BORDER_CONSTANT, Seamless::EDGE_FLOAT);

	m_index_buf.create(m_height + 2, m_width + 2);
	m_index_buf = -1;

	int equation_cnt = 0;
	for (int i = 0; i < m_height; i++)
	{
		uchar *src = bounds_mask_ext.ptr(i+1, 1);
		int *dst   = m_index_buf.ptr<int>(i+1, 1);
		for (int j = 0; j < m_width; j++, src++, dst++)
		{
			if (*src==INTERIOR)
				*dst = equation_cnt++;
		}
	}

	int elements_cnt = equation_cnt;
	for (int i = 0; i < m_height; i++)
	{
		int *src   = m_index_buf.ptr<int>(i+1, 1);
		for (int j = 0; j < m_width; j++, src++)
		{
			if (src[1] >= 0)
				elements_cnt++;
			if (src[m_index_buf.cols] >= 0)
				elements_cnt++;
		}
	}
	m_a.resize(equation_cnt, equation_cnt);
	m_a.reserve(elements_cnt);
	m_b[0].resize(equation_cnt);
	m_b[1].resize(equation_cnt);
	m_b[2].resize(equation_cnt);
	m_solution.resize(equation_cnt);

	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(equation_cnt);

	int offs_enl[] = {-1, 1, -m_width - 2, m_width + 2};
	int offs[]     = {-3, 3, -(int)target.step, target.step};
	int p_i = 0;
	for (int i = 0; i < m_height; i++)
	{
		const int *p_index                 = m_index_buf.ptr<int>(i + 1, 1);
		const unsigned char *p_bounds_mask = bounds_mask_ext.ptr(i + 1, 1);
		const unsigned char *p_target      = target.ptr(i, 0);
		const float *p_lapl                = lapl.ptr<float>(i, 0);

		for (int j = 0; j < m_width; j++)
		{
			if (*p_bounds_mask==INTERIOR)
			{
				double s[3] = {-p_lapl[0], -p_lapl[1], -p_lapl[2]};
				int cnt_c = 0;
				for (int k = 0; k < 4; k++)
				{
					switch (p_bounds_mask[offs_enl[k]])
					{
					case EDGE_FIXED:
						{
							const unsigned char *p = p_target + offs[k];
							s[0] += p[0];
							s[1] += p[1];
							s[2] += p[2];
						}
					case INTERIOR:
						{
							cnt_c++;
							break;
						}
					case EDGE_FLOAT:
						break;
					}
				}
				m_b[0](p_i) = s[0];
				m_b[1](p_i) = s[1];
				m_b[2](p_i) = s[2];

				tripletList.push_back(T(p_i, p_i, cnt_c));
				int eq_number = p_index[1];
				if (eq_number >= 0)
				{
					tripletList.push_back(T(p_i, eq_number, -1.0));
				}
				eq_number = p_index[m_index_buf.cols];
				if (eq_number >= 0)
				{
					tripletList.push_back(T(p_i, eq_number, -1.0));
				}
				p_i++;
			}
			p_index++;
			p_bounds_mask++;
			p_target+= 3;
			p_lapl+= 3;
		}
	}
	m_a.setFromTriplets(tripletList.begin(), tripletList.end());
}

cv::Mat3b Seamless::Proceed(cv::Mat mask)
{
	//Eigen::SimplicialLDLT< Eigen::SparseMatrix<double>, Eigen::Upper > cg(m_a);
	Eigen::SimplicialLLT< Eigen::SparseMatrix<double>, Eigen::Upper > cg(m_a);

	for (int c = 0; c < 3; c++)
	{
		m_solution = cg.solve(m_b[c]);

		for (int i = 0; i < m_height; i++)
		{
			unsigned char *p_dst_img = m_dst_img.ptr(i) + c;
			int *p_index   = m_index_buf.ptr<int>(i+1, 1);
			for (int j = 0; j < m_width; j++)
			{
				int k = p_index[j];
				uchar weight = (uchar)mask.at<uchar>(i, j);
				if (k >= 0)
				{
					*p_dst_img = (255 - weight) / 255.f * (*p_dst_img)  + weight / 255.f * (unsigned char)min(255.0, max(0.0, m_solution(k)));
				}
				p_dst_img+= 3;
			}
		}
	}

	return m_dst_img;
}

cv::Mat3f Seamless::Laplacian(const cv::Mat3b &src)
{
	cv::Mat3f lapl;
	cv::Laplacian(src, lapl, CV_32F, 1, 1, 0, BORDER_REFLECT);

	return lapl;
}

cv::Mat3f Seamless::LaplacianByMixGrad(const cv::Mat3b &src1, const cv::Mat3b &src2, const cv::Mat1b &mix_v)
{
	return mix_grad(src1, src2, mix_v);
}
