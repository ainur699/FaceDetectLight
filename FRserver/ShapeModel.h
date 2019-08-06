#pragma once


//-----------------------------------------------------------------------------
#include <opencv2/imgproc/imgproc.hpp>


//-----------------------------------------------------------------------------
namespace FACETRACKER
{
	void AddOrthRow(cv::Mat &R);
	void Euler2Rot(cv::Mat &R, const double pitch, const double yaw, const double roll, bool full = true);
	void Euler2Rot(cv::Mat &R, cv::Mat &p, bool full = true);
	void Rot2Euler(cv::Mat &R, double &pitch, double &yaw, double &roll);
	void Rot2Euler(cv::Mat &R, cv::Mat &p);

	void Align3Dto2DShapes(double &scale, double &pitch, double &yaw, double &roll, double &x, double &y, cv::Mat &s2D, cv::Mat &s3D);

	void ReadMat(std::ifstream &s, cv::Mat &M);
	void WriteMat(std::ofstream &s, cv::Mat &M);


	//-----------------------------------------------------------------------------
	class ShapeModel
	{
	public:
		int _type;

		virtual void Write(std::ofstream &s) = 0;
		virtual void ReadBinary(std::ifstream &s) = 0;
	};


	//-----------------------------------------------------------------------------
	class LinearShapeModel : public ShapeModel
	{
	public:
		int     _n;
		cv::Mat _V;
		cv::Mat _E;
		cv::Mat _M;

		inline int nPoints() { return _n; }
		inline int nModes() { return _V.cols; }

		virtual void Init(cv::Mat &M, cv::Mat &V, cv::Mat &E) = 0;
		virtual void Identity(cv::Mat &plocal, cv::Mat &pglobl) = 0;
	};


	//-----------------------------------------------------------------------------
	class PDM3D : public LinearShapeModel
	{
	public:
		enum { TP_PDM3D_BIN = 35 };

		PDM3D() { _type = TP_PDM3D_BIN; }
		void Init(cv::Mat &M, cv::Mat &V, cv::Mat &E);

		void Write(std::ofstream &s);
		void ReadBinary(std::ifstream &s);

		void CalcShape3D(cv::Mat &s, cv::Mat &plocal);
		void CalcParams(cv::Mat &s, cv::Mat &plocal, cv::Mat &pglobl, cv::Mat S_);
		void Identity(cv::Mat &plocal, cv::Mat &pglobl);
	};
}


//-----------------------------------------------------------------------------
class Estimate3D
{
public:
	enum { N = 66 };

	bool Init(const std::string &pdm_filename);
	void estimate(const cv::Mat &points2d, cv::Mat &global, cv::Mat &local, cv::Mat &shape);

public:
	FACETRACKER::PDM3D m_pdm;
};


//-----------------------------------------------------------------------------
