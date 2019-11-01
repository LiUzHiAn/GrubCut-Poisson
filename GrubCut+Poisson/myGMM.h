#ifndef __MY_GMM__
#define __MY_GMM__

#include <opencv2\imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// ��˹���ģ��
class MyGMM
{
public:
	const static int componentNum = 5;					//  5����˹ģ�͵Ļ��
	MyGMM(Mat& _params);

	void calDetermAndInverse(int ci);					// ����i����˹ģ�͵�Э���������ʽ�������
	void startLearning();								// ��ʼһ��GMMѧϰ��������ʼ�������أ�
	void addSamplePixel(int ci, Vec3d pixel);			// ��ĳ��������ӵ�ĳ����˹ģ��
	double calGaussianItemPr(Vec3d pixel, int ci);		// ����ĳ����������ĳ����˹ģ�͵ĸ���
	int whichComponent(Vec3d pixel);					// �ж�ĳ�����������ĸ���˹ģ��
	void endLearning();									// ����һ��GMMѧϰ�����������£�
	double calGMMPr(Vec3d pixel);						// ����ĳ�����ص�����GMM�ĸ���
private:

	Mat params;									//  GMM�Ĳ���������ÿ����˹ģ�͵�Э�����ֵ��Ȩ�أ�* ��˹ģ������

	double* coefs;								// GMM��ÿ�����صĸ�˹ģ�͵�Ȩֵ������ʼ�洢ָ��
	double* mean;								// ��ֵ������ʼ�洢ָ�루ÿ��ģ����һ����ֵu��u��3ά�ȵ�������
	double* cov;								// Э���������ʼ�洢ָ�루ÿ��ģ����һ��Э���Э������3*3ά�ȵľ���

	double inverseCovs[componentNum][3][3];		// Э����������
	double covDeterms[componentNum];			// Э��������������
	double sums[componentNum][3];				// ����ĳ��GMM�и�˹ģ�͵��������ص�R��G��B����֮��
	double prods[componentNum][3][3];			// ������Э��������
	int samplePixelCounts[componentNum];		// ����ĳ��GMM�и�˹ģ�͵�����������
	int totalSamplePixelCount;					// ����������
};

#endif