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
	void saveBinaryMaskImg(string maskFileName);  // 保存结果生成的mask遮罩图片，用于poisson融合
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
	Rect rect;  // 矩形框
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;  // 前景像素，背景像素，可能的前景像素，可能的背景像素
	int iterCount;
};

#endif