#include "myGMM.h"

MyGMM::MyGMM(Mat& _params)
{
	if (_params.empty())  // 如果参数为空，则创建
	{
		// 行数只能为1，列数只能为13x5，用来记录GMM中每个高斯分量模型中的系数，均值，协方差等
		_params.create(1, 13 * componentNum, CV_64FC1);
		_params.setTo(Scalar(0));
	}
	else if ((_params.type() != CV_64FC1) || (_params.rows != 1) || (_params.cols != 13 * componentNum))
		CV_Error(CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount");

	this->params = _params;

	this->coefs = params.ptr<double>(0);		// 一共有5*1个数字
	this->mean = this->coefs + componentNum;	// 一共有5*3个数字
	this->cov = this->mean + 3 * componentNum;	// 一共有5*9个数字

	for (int i = 0;i < componentNum;i++)
	{
		// 对每个高斯模型求解协方差的行列式和逆
		if (coefs[i]>0)
			calDetermAndInverse(i);
	}
	totalSamplePixelCount = 0;  // 准备开始对GMM进行学习（即参数估计）

}

// 计算第ci个高斯模型的协方差的逆矩阵和协方差的行列式
void MyGMM::calDetermAndInverse(int ci)
{
	double* c = this->cov + ci * 9;
	// 三阶行列式
	double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
	this->covDeterms[ci] = dtrm;  // 把结果保存到类成员变量
								  // 伴随矩阵求协方差的逆
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

//GMM参数学习前的初始化，主要是对要求和的变量置零
void MyGMM::startLearning()
{
	for (int ci = 0;ci < componentNum;ci++)
	{
		sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;   // sum是R/G/B三种颜色像素值的和
		prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
		prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
		prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
		samplePixelCounts[ci] = 0;  // 第ci个高斯模型中包含的像素量
	}
	totalSamplePixelCount = 0;
}
// 求解一个像素属于GMM中第ci个高斯模型的概率
double MyGMM::calGaussianItemPr(Vec3d pixel, int ci)
{
	// 这里就是根据多维的高斯模型，求解其概率
	double res = 0;
	// 如果第ci个模型的权重为0，直接不用算，然后0即可
	if (coefs[ci] > 0)
	{
		Vec3d diff = pixel;
		double *m = this->mean + ci * 3;   //   第ci个高斯模型的均值
										   // diff就是多维高斯概率求解中的（x-u）
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];

		double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] + diff[2] * inverseCovs[ci][2][0])
			+ diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] + diff[2] * inverseCovs[ci][2][1])
			+ diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] + diff[2] * inverseCovs[ci][2][2]);
		res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f*mult);
	}
	return res;
}

// 求解一个像素最可能属于GMM中哪个高斯模型
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

// 把像素pixel加到GMM的第Ci个高斯模型中
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

// 结束某次GMM参数学习
void MyGMM::endLearning()
{
	for (int ci = 0;ci < componentNum;ci++)
	{
		int n = samplePixelCounts[ci]++;
		if (n == 0)  // 如果某个高斯模型中没有点，也就是该高斯模型权重为0，到时候求概率时根本不考虑
			coefs[ci] = 0;
		else
		{
			CV_Assert(totalSamplePixelCount > 0);
			// 计算第ci个高斯模型的权值系数
			double inv_n = 1.0 / n;
			coefs[ci] = (double)n / totalSamplePixelCount;  // 每个高斯模型对应的系数

															// 更新mean和协方差
			double* m = mean + 3 * ci;   // 先找到Ci个高斯模型对应的mean参数起始指针
			m[0] = sums[ci][0] * inv_n;
			m[1] = sums[ci][1] * inv_n;
			m[2] = sums[ci][2] * inv_n;

			//计算第ci个高斯模型的协方差
			double* c = cov + 9 * ci;  // 先找到Ci个高斯模型对应的协方差参数起始指针
			c[0] = prods[ci][0][0] * inv_n - m[0] * m[0];
			c[1] = prods[ci][0][1] * inv_n - m[0] * m[1];
			c[2] = prods[ci][0][2] * inv_n - m[0] * m[2];
			c[3] = prods[ci][1][0] * inv_n - m[1] * m[0];
			c[4] = prods[ci][1][1] * inv_n - m[1] * m[1];
			c[5] = prods[ci][1][2] * inv_n - m[1] * m[2];
			c[6] = prods[ci][2][0] * inv_n - m[2] * m[0];
			c[7] = prods[ci][2][1] * inv_n - m[2] * m[1];
			c[8] = prods[ci][2][2] * inv_n - m[2] * m[2];

			// 根据更新过后的像素分布，计算各个高斯模型的协方差的行列式和其逆矩阵
			calDetermAndInverse(ci);
		}
	}
}

double MyGMM::calGMMPr(Vec3d pixel)
{
	// 计算个像素（由color=（B,G,R）三维double型向量来表示）属于这个GMM混合高斯模型的概率
	// 也就是把这个像素像素属于componentsNum个高斯模型的概率与对应的权值相乘再相加
	double res = 0;
	for (int ci = 0; ci < componentNum; ci++)
		res += coefs[ci] * this->calGaussianItemPr(pixel, ci);
	return res;
}