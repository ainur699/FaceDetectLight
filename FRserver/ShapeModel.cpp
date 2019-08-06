

//-----------------------------------------------------------------------------
#include <fstream>
#include "ShapeModel.h"


//-----------------------------------------------------------------------------
#define db at<double>


//-----------------------------------------------------------------------------





//-----------------------------------------------------------------------------
void FACETRACKER::PDM3D::Init(cv::Mat &M, cv::Mat &V, cv::Mat &E)
{
	_M = M.clone();
	_V = V.clone();
	_E = E.clone();
	_n = _M.rows / 3;
}


//-----------------------------------------------------------------------------
void FACETRACKER::PDM3D::Identity(cv::Mat &plocal, cv::Mat &pglobl)
{
	plocal = cv::Mat::zeros(_V.cols, 1, CV_64F);
	pglobl = (cv::Mat_<double>(6, 1) << 1, 0, 0, 0, 0, 0);
}


//-----------------------------------------------------------------------------
void FACETRACKER::PDM3D::CalcShape3D(cv::Mat &s, cv::Mat &plocal)
{
	s = _M + _V * plocal;

	return;
}


//-----------------------------------------------------------------------------
void FACETRACKER::PDM3D::CalcParams(cv::Mat &s, cv::Mat &plocal, cv::Mat &pglobl, cv::Mat S_)
{
	if ((pglobl.rows != 6) || (pglobl.cols != 1) || (pglobl.type() != CV_64F)) pglobl.create(6, 1, CV_64F);

	int j, n = _M.rows / 3;
	double si, scale, pitch, yaw, roll, tx, ty, Tx, Ty, Tz;

	cv::Mat R(3, 3, CV_64F);
	cv::Mat z(n, 1, CV_64F);
	cv::Mat t(3, 1, CV_64F);
	cv::Mat p(_V.cols, 1, CV_64F);
	cv::Mat S(this->nPoints(), 3, CV_64F);
	cv::Mat r = R.row(2);

	plocal = cv::Mat::zeros(_V.cols, 1, CV_64F);
	for (int iter = 0; iter < 100; iter++)
	{
		this->CalcShape3D(S_, plocal);
		Align3Dto2DShapes(scale, pitch, yaw, roll, tx, ty, s, S_);
		Euler2Rot(R, pitch, yaw, roll);

		S = (S_.reshape(1, 3)).t();
		z = scale * S * r.t();
		si = 1.0 / scale;
		Tx = -si * (R.db(0, 0) * tx + R.db(1, 0) * ty);
		Ty = -si * (R.db(0, 1) * tx + R.db(1, 1) * ty);
		Tz = -si * (R.db(0, 2) * tx + R.db(1, 2) * ty);
		for (j = 0; j < n; j++)
		{
			t.db(0, 0) = s.db(j, 0);
			t.db(1, 0) = s.db(j + n, 0);
			t.db(2, 0) = z.db(j, 0);
			S_.db(j, 0) = si * t.dot(R.col(0)) + Tx;
			S_.db(j + n, 0) = si * t.dot(R.col(1)) + Ty;
			S_.db(j + n * 2, 0) = si * t.dot(R.col(2)) + Tz;
		}

		plocal = _V.t() * (S_ - _M);
		if (iter > 0) if (cv::norm(plocal - p) < 1.0e-5) break;

		plocal.copyTo(p);
	}

	pglobl.db(0, 0) = scale;
	pglobl.db(1, 0) = pitch;
	pglobl.db(2, 0) = yaw;
	pglobl.db(3, 0) = roll;
	pglobl.db(4, 0) = tx;
	pglobl.db(5, 0) = ty;

	return;
}


//-----------------------------------------------------------------------------
void FACETRACKER::PDM3D::Write(std::ofstream &s)
{
	int t = TP_PDM3D_BIN;
	s.write(reinterpret_cast<char*>(&t), sizeof(t));
	WriteMat(s, _V);
	WriteMat(s, _E);
	WriteMat(s, _M);
}


//-----------------------------------------------------------------------------
void FACETRACKER::PDM3D::ReadBinary(std::ifstream &s)
{
	int type;
	s.read(reinterpret_cast<char*>(&type), sizeof(type));

	ReadMat(s, _V);
	ReadMat(s, _E);
	ReadMat(s, _M);

	_n = _M.rows / 3;
}


//-----------------------------------------------------------------------------
void FACETRACKER::Euler2Rot(cv::Mat &R, cv::Mat &p, bool full)
{
	Euler2Rot(R, p.db(1, 0), p.db(2, 0), p.db(3, 0), full);

	return;
}


//-----------------------------------------------------------------------------
void FACETRACKER::Euler2Rot(cv::Mat &R, const double pitch, const double yaw, const double roll, bool full)
{
	if (full)
	{
		if ((R.rows != 3) || (R.cols != 3)) R.create(3, 3, CV_64F);
	}
	else
	{
		if ((R.rows != 2) || (R.cols != 3)) R.create(2, 3, CV_64F);
	}

	double sina = sin(pitch), sinb = sin(yaw), sinc = sin(roll);
	double cosa = cos(pitch), cosb = cos(yaw), cosc = cos(roll);

	R.db(0, 0) = cosb * cosc;
	R.db(0, 1) = -cosb * sinc;
	R.db(0, 2) = sinb;
	R.db(1, 0) = cosa * sinc + sina * sinb * cosc;
	R.db(1, 1) = cosa * cosc - sina * sinb * sinc;
	R.db(1, 2) = -sina * cosb;
	if (full) AddOrthRow(R);

	return;
}


//-----------------------------------------------------------------------------
void FACETRACKER::Rot2Euler(cv::Mat &R, double &pitch, double &yaw, double &roll)
{
	double q[4];
	q[0] = sqrt(1 + R.db(0, 0) + R.db(1, 1) + R.db(2, 2)) / 2;
	q[1] = (R.db(2, 1) - R.db(1, 2)) / (4 * q[0]);
	q[2] = (R.db(0, 2) - R.db(2, 0)) / (4 * q[0]);
	q[3] = (R.db(1, 0) - R.db(0, 1)) / (4 * q[0]);

	yaw = asin(2 * (q[0] * q[2] + q[1] * q[3]));
	pitch = atan2(2 * (q[0] * q[1] - q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]);
	roll = atan2(2 * (q[0] * q[3] - q[1] * q[2]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);

	return;
}


//-----------------------------------------------------------------------------
void FACETRACKER::Rot2Euler(cv::Mat &R, cv::Mat &p)
{
	Rot2Euler(R, p.db(1, 0), p.db(2, 0), p.db(3, 0));

	return;
}


//-----------------------------------------------------------------------------
void FACETRACKER::AddOrthRow(cv::Mat &R)
{
	R.db(2, 0) = R.db(0, 1) * R.db(1, 2) - R.db(0, 2) * R.db(1, 1);
	R.db(2, 1) = R.db(0, 2) * R.db(1, 0) - R.db(0, 0) * R.db(1, 2);
	R.db(2, 2) = R.db(0, 0) * R.db(1, 1) - R.db(0, 1) * R.db(1, 0);

	return;
}


//-----------------------------------------------------------------------------
void FACETRACKER::Align3Dto2DShapes(double &scale, double &pitch, double &yaw, double &roll, double &x, double &y, cv::Mat &s2D, cv::Mat &s3D)
{
	int    i, n = s2D.rows / 2;
	double t2[2], t3[3];

	cv::Mat s2D_cpy = s2D.clone();
	cv::Mat s3D_cpy = s3D.clone();
	cv::Mat X = (s2D_cpy.reshape(1, 2)).t();
	cv::Mat S = (s3D_cpy.reshape(1, 3)).t();
	for (i = 0; i < 2; i++) { cv::Mat v = X.col(i); t2[i] = sum(v)[0] / n; v -= t2[i]; }
	for (i = 0; i < 3; i++) { cv::Mat v = S.col(i); t3[i] = sum(v)[0] / n; v -= t3[i]; }

	cv::Mat M = ((S.t() * S).inv(cv::DECOMP_CHOLESKY)) * S.t() * X;
	cv::Mat MtM = M.t() * M;
	cv::SVD svd(MtM, cv::SVD::MODIFY_A);

	svd.w.db(0, 0) = 1.0 / sqrt(svd.w.db(0, 0));
	svd.w.db(1, 0) = 1.0 / sqrt(svd.w.db(1, 0));

	cv::Mat T(3, 3, CV_64F);
	T(cv::Rect(0, 0, 3, 2)) = svd.u * cv::Mat::diag(svd.w) * svd.vt * M.t();

	scale = 0.5 * sum(T(cv::Rect(0, 0, 3, 2)).mul(M.t()))[0];
	AddOrthRow(T);
	Rot2Euler(T, pitch, yaw, roll);

	T *= scale;
	x = t2[0] - (T.db(0, 0) * t3[0] + T.db(0, 1) * t3[1] + T.db(0, 2) * t3[2]);
	y = t2[1] - (T.db(1, 0) * t3[0] + T.db(1, 1) * t3[1] + T.db(1, 2) * t3[2]);

	return;
}


//-----------------------------------------------------------------------------
void FACETRACKER::ReadMat(std::ifstream &s, cv::Mat &M)
{
	int r, c, t;
	s.read((char*)&r, sizeof(int));
	s.read((char*)&c, sizeof(int));
	s.read((char*)&t, sizeof(int));

	M = cv::Mat(r, c, t);
	s.read(reinterpret_cast<char*>(const_cast<uchar*>(M.datastart)), M.total() * M.elemSize());
}


//-----------------------------------------------------------------------------
void FACETRACKER::WriteMat(std::ofstream &s, cv::Mat &M)
{
	int t = M.type();
	s.write(reinterpret_cast<char*>(&M.rows), sizeof(int));
	s.write(reinterpret_cast<char*>(&M.cols), sizeof(int));
	s.write(reinterpret_cast<char*>(&t), sizeof(int));

	s.write(reinterpret_cast<char*>(const_cast<uchar *>(M.datastart)), M.total() * M.elemSize());
}


//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
bool Estimate3D::Init(const std::string &pdm_filename)
{
	std::ifstream is(pdm_filename, std::ios::binary);
	if (!is.is_open()) return false;

	m_pdm.ReadBinary(is);

	return true;
}


//-----------------------------------------------------------------------------
void Estimate3D::estimate(const cv::Mat &points2d, cv::Mat &global, cv::Mat &local, cv::Mat &shape)
{
	global.create(6, 1, CV_64F);
	shape.create(m_pdm._M.rows, 1, CV_64F);

	m_pdm.CalcParams(const_cast<cv::Mat&>(points2d), local, global, shape);
}


//-----------------------------------------------------------------------------