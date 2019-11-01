#ifndef __MY_GMM__
#define __MY_GMM__

#include <opencv2\imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// 高斯混合模型
class MyGMM
{
public:
	const static int componentNum = 5;					//  5个高斯模型的混合
	MyGMM(Mat& _params);

	void calDetermAndInverse(int ci);					// 求解第i个高斯模型的协方差的行列式和逆矩阵
	void startLearning();								// 开始一次GMM学习（即将开始分配像素）
	void addSamplePixel(int ci, Vec3d pixel);			// 将某个像素添加到某个高斯模型
	double calGaussianItemPr(Vec3d pixel, int ci);		// 计算某个像素属于某个高斯模型的概率
	int whichComponent(Vec3d pixel);					// 判断某个像素属于哪个高斯模型
	void endLearning();									// 结束一次GMM学习（即参数更新）
	double calGMMPr(Vec3d pixel);						// 计算某个像素点属于GMM的概率
private:

	Mat params;									//  GMM的参数（包括每个高斯模型的协方差，均值，权重）* 高斯模型数量

	double* coefs;								// GMM的每个像素的高斯模型的权值变量起始存储指针
	double* mean;								// 均值变量起始存储指针（每个模型有一个均值u，u是3维度的向量）
	double* cov;								// 协方差变量起始存储指针（每个模型有一个协方差，协方差是3*3维度的矩阵）

	double inverseCovs[componentNum][3][3];		// 协方差的逆矩阵
	double covDeterms[componentNum];			// 协方差矩阵的行列数
	double sums[componentNum][3];				// 属于某个GMM中高斯模型的所有像素的R、G、B像素之和
	double prods[componentNum][3][3];			// 用来求协方差矩阵的
	int samplePixelCounts[componentNum];		// 属于某个GMM中高斯模型的所有像素数
	int totalSamplePixelCount;					// 所有像素数
};

#endif