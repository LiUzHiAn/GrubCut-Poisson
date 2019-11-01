#include "myGMM.h"

MyGMM::MyGMM(Mat& _params)
{
	if (_params.empty())  // �������Ϊ�գ��򴴽�
	{
		// ����ֻ��Ϊ1������ֻ��Ϊ13x5��������¼GMM��ÿ����˹����ģ���е�ϵ������ֵ��Э�����
		_params.create(1, 13 * componentNum, CV_64FC1);
		_params.setTo(Scalar(0));
	}
	else if ((_params.type() != CV_64FC1) || (_params.rows != 1) || (_params.cols != 13 * componentNum))
		CV_Error(CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount");

	this->params = _params;

	this->coefs = params.ptr<double>(0);		// һ����5*1������
	this->mean = this->coefs + componentNum;	// һ����5*3������
	this->cov = this->mean + 3 * componentNum;	// һ����5*9������

	for (int i = 0;i < componentNum;i++)
	{
		// ��ÿ����˹ģ�����Э���������ʽ����
		if (coefs[i]>0)
			calDetermAndInverse(i);
	}
	totalSamplePixelCount = 0;  // ׼����ʼ��GMM����ѧϰ�����������ƣ�

}

// �����ci����˹ģ�͵�Э�����������Э���������ʽ
void MyGMM::calDetermAndInverse(int ci)
{
	double* c = this->cov + ci * 9;
	// ��������ʽ
	double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
	this->covDeterms[ci] = dtrm;  // �ѽ�����浽���Ա����
								  // ���������Э�������
	double inv_dtrm = 1.0 / dtrm;
	this->inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) * inv_dtrm;
	this->inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) * inv_dtrm;
	this->inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) * inv_dtrm;
	this->inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) * inv_dtrm;
	this->inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) * inv_dtrm;
	this->inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) * inv_dtrm;
	this->inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) * inv_dtrm;
	this->inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) * inv_dtrm;
	this->inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) * inv_dtrm;
}

//GMM����ѧϰǰ�ĳ�ʼ������Ҫ�Ƕ�Ҫ��͵ı�������
void MyGMM::startLearning()
{
	for (int ci = 0;ci < componentNum;ci++)
	{
		sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;   // sum��R/G/B������ɫ����ֵ�ĺ�
		prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
		prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
		prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
		samplePixelCounts[ci] = 0;  // ��ci����˹ģ���а�����������
	}
	totalSamplePixelCount = 0;
}
// ���һ����������GMM�е�ci����˹ģ�͵ĸ���
double MyGMM::calGaussianItemPr(Vec3d pixel, int ci)
{
	// ������Ǹ��ݶ�ά�ĸ�˹ģ�ͣ���������
	double res = 0;
	// �����ci��ģ�͵�Ȩ��Ϊ0��ֱ�Ӳ����㣬Ȼ��0����
	if (coefs[ci] > 0)
	{
		Vec3d diff = pixel;
		double *m = this->mean + ci * 3;   //   ��ci����˹ģ�͵ľ�ֵ
										   // diff���Ƕ�ά��˹��������еģ�x-u��
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];

		double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] + diff[2] * inverseCovs[ci][2][0])
			+ diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] + diff[2] * inverseCovs[ci][2][1])
			+ diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] + diff[2] * inverseCovs[ci][2][2]);
		res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f*mult);
	}
	return res;
}

// ���һ���������������GMM���ĸ���˹ģ��
int MyGMM::whichComponent(Vec3d pixel)
{
	int k = 0;
	double max = 0;
	for (int ci = 0;ci < componentNum;ci++)
	{
		double pr = this->calGaussianItemPr(pixel, ci);
		if (pr > max)
		{
			k = ci;
			max = pr;
		}
	}
	return k;
}

// ������pixel�ӵ�GMM�ĵ�Ci����˹ģ����
void MyGMM::addSamplePixel(int ci, Vec3d pixel)
{
	sums[ci][0] += pixel[0];
	sums[ci][1] += pixel[1];
	sums[ci][2] += pixel[2];
	prods[ci][0][0] += pixel[0] * pixel[0]; prods[ci][0][1] += pixel[0] * pixel[1]; prods[ci][0][2] += pixel[0] * pixel[2];
	prods[ci][1][0] += pixel[1] * pixel[0]; prods[ci][1][1] += pixel[1] * pixel[1]; prods[ci][1][2] += pixel[1] * pixel[2];
	prods[ci][2][0] += pixel[2] * pixel[0]; prods[ci][2][1] += pixel[2] * pixel[1]; prods[ci][2][2] += pixel[2] * pixel[2];
	samplePixelCounts[ci]++;
	totalSamplePixelCount++;
}

// ����ĳ��GMM����ѧϰ
void MyGMM::endLearning()
{
	for (int ci = 0;ci < componentNum;ci++)
	{
		int n = samplePixelCounts[ci]++;
		if (n == 0)  // ���ĳ����˹ģ����û�е㣬Ҳ���Ǹø�˹ģ��Ȩ��Ϊ0����ʱ�������ʱ����������
			coefs[ci] = 0;
		else
		{
			CV_Assert(totalSamplePixelCount > 0);
			// �����ci����˹ģ�͵�Ȩֵϵ��
			double inv_n = 1.0 / n;
			coefs[ci] = (double)n / totalSamplePixelCount;  // ÿ����˹ģ�Ͷ�Ӧ��ϵ��

															// ����mean��Э����
			double* m = mean + 3 * ci;   // ���ҵ�Ci����˹ģ�Ͷ�Ӧ��mean������ʼָ��
			m[0] = sums[ci][0] * inv_n;
			m[1] = sums[ci][1] * inv_n;
			m[2] = sums[ci][2] * inv_n;

			//�����ci����˹ģ�͵�Э����
			double* c = cov + 9 * ci;  // ���ҵ�Ci����˹ģ�Ͷ�Ӧ��Э���������ʼָ��
			c[0] = prods[ci][0][0] * inv_n - m[0] * m[0];
			c[1] = prods[ci][0][1] * inv_n - m[0] * m[1];
			c[2] = prods[ci][0][2] * inv_n - m[0] * m[2];
			c[3] = prods[ci][1][0] * inv_n - m[1] * m[0];
			c[4] = prods[ci][1][1] * inv_n - m[1] * m[1];
			c[5] = prods[ci][1][2] * inv_n - m[1] * m[2];
			c[6] = prods[ci][2][0] * inv_n - m[2] * m[0];
			c[7] = prods[ci][2][1] * inv_n - m[2] * m[1];
			c[8] = prods[ci][2][2] * inv_n - m[2] * m[2];

			// ���ݸ��¹�������طֲ������������˹ģ�͵�Э���������ʽ���������
			calDetermAndInverse(ci);
		}
	}
}

double MyGMM::calGMMPr(Vec3d pixel)
{
	// ��������أ���color=��B,G,R����άdouble����������ʾ���������GMM��ϸ�˹ģ�͵ĸ���
	// Ҳ���ǰ����������������componentsNum����˹ģ�͵ĸ������Ӧ��Ȩֵ��������
	double res = 0;
	for (int ci = 0; ci < componentNum; ci++)
		res += coefs[ci] * this->calGaussianItemPr(pixel, ci);
	return res;
}