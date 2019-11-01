#ifndef __MY_GC_APPLICATION__
#define __MY_GC_APPLICATION__

#include <opencv2\imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class MyGCApplication
{
public:
	enum { NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
	static const int radius = 2;
	static const int thickness = -1;
	void reset();
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void showImage() const;
	void mouseClick(int event, int x, int y, int flags, void* param);
	int nextIter();
	int getIterCount() const { return iterCount; }
	void saveBinaryMaskImg(string maskFileName);  // ���������ɵ�mask����ͼƬ������poisson�ں�
	void help();  
private:
	void setRectInMask();
	void setLblsInMask(int flags, Point p, bool isPr);
	const string* winName;
	const Mat* image;
	Mat mask;
	Mat bgdModel, fgdModel;
	uchar rectState, lblsState, prLblsState;
	bool isInitialized;
	Rect rect;  // ���ο�
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;  // ǰ�����أ��������أ����ܵ�ǰ�����أ����ܵı�������
	int iterCount;
};

#endif