#include "myGCApplication.h"
#include "grabCutUtils.h"
#include <opencv2\highgui.hpp>

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);
const int BGD_KEY = EVENT_FLAG_CTRLKEY;  //Ctrl键
const int FGD_KEY = EVENT_FLAG_SHIFTKEY; //Shift键

void MyGCApplication::reset()
{
	if (!mask.empty())  // 清空mask全为背景
		mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();
	isInitialized = false;
	rectState = NOT_SET;
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}
void MyGCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
	if (_image.empty() || _winName.empty())
		return;
	// 把图片从参数中读进来，设置为类成员变量，以后的操作均在image中
	image = &_image;
	winName = &_winName;
	// 创建一个mask和图片一样大小的mask
	mask.create(image->size(), CV_8UC1);
	reset();   // 把一些状态清除
}
void MyGCApplication::showImage() const
{
	if (image->empty() || winName->empty())
		return;
	// 最后显示结果的图片（初始化为全白）
	Mat res=Mat(image->rows, image->cols, CV_8UC3, Scalar(255, 255, 255));
	// res.create(image->rows, image->cols, CV_8UC3, Scalar(255, 255, 255));
	Mat binMask;
	if (!isInitialized) // 如果没有初始化，直接将图片作为结果
		image->copyTo(res);
	else
	{
		// 根据分割结果的mask（有4种取值），得到最后在图片上的遮罩binMask
		getBinMask(mask, binMask);
		// binMask作为一个掩模板，如果在某个像素点(i, j)其值为1，则把srcImage.at(i, j)处的值直接赋给dstImage.at(i, j)，
		// 如果其值为0则dstImage.at(i, j)处保留其原始像素值。
		image->copyTo(res, binMask);
	}
	vector<Point>::const_iterator it;
	/*下面4句代码是将选用户选择的4种点用不同的颜色显示出来*/
	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)
		circle(res, *it, radius, BLUE, thickness);  // 背景是蓝色
	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)
		circle(res, *it, radius, RED, thickness); // 前景是红色
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness); // 可能的背景是淡蓝色
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);  // 可能的前景是粉色

	// 如果是在画或者已经画好了矩形框，就把矩形框用绿色笔画出来
	if (rectState == IN_PROCESS || rectState == SET)
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
	imshow(*winName, res);
}
void MyGCApplication::setRectInMask()
{
	CV_Assert(!mask.empty());
	mask.setTo(GC_BGD);
	// 将矩形框内全部设置为GC_PR_FGD，框外圈设置为GC_BGD
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	// 设置mask的RIO
	(mask(rect)).setTo(Scalar(GC_PR_FGD));
}
void MyGCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if (!isPr)  // 确定的点（鼠标左键+ctrl/shift）

	// ctrl 背景  shift  前景
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;
		fvalue = GC_FGD;
	}
	else  // 可能的点 （鼠标右键+ctrl/shift）
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD;
		fvalue = GC_PR_FGD;
	}
	if (flags & BGD_KEY) // ctrl键
	{
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);  // 改变mask的值
	}
	if (flags & FGD_KEY) // shift键
	{
		fpxls->push_back(p);
		circle(mask, p, radius, fvalue, thickness);
	}
}
void MyGCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	// TODO add bad args check
	switch (event)
	{
		// 鼠标左键按下
	case EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,  // 且按下了ctrl键，表明在框内选背景
			isf = (flags & FGD_KEY) != 0;
		// 如果ctrl或shift都没按下，且矩形框也还没选，那肯定是即将要开始选矩形框
		if (rectState == NOT_SET && !isb && !isf)
		{
			rectState = IN_PROCESS;
			rect = Rect(x, y, 1, 1);
		}
		// 如果矩形框选好了，且按下了ctrl或shift键，那肯定是在选好的矩形框内用户再加mask
		// 这说明是在进行lblsState过程
		if ((isb || isf) && rectState == SET)
			lblsState = IN_PROCESS;
	}
	break;
	// 鼠标右键按下
	case EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET)
			prLblsState = IN_PROCESS;
	}
	break;
	// 鼠标左键抬起
	case EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)  // 如果是在画矩形框时抬起
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			rectState = SET; // 矩形框画好了
			setRectInMask(); // 将矩形框外全设置为背景，框内设置为GC_PR_FGD
			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		if (lblsState == IN_PROCESS)
		{
			// 把矩形框中设置为背景的点加入到相应的mask中去
			setLblsInMask(flags, Point(x, y), false);
			lblsState = SET;
			showImage();
		}
		break;
		// 鼠标右键抬起
	case EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			// 把矩形框中设置为背景的点加入到相应的mask中去
			setLblsInMask(flags, Point(x, y), true);
			prLblsState = SET;
			showImage();
		}
		break;
		// 鼠标移动事件
	case EVENT_MOUSEMOVE: // 鼠标移动时，必须将矩形框先画好
		if (rectState == IN_PROCESS)  // 如果是矩形框还处于在画的状态
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		else if (lblsState == IN_PROCESS)  //如果是左键移动
		{
			setLblsInMask(flags, Point(x, y), false);
			showImage();
		}
		else if (prLblsState == IN_PROCESS)  //如果是右键移动
		{
			setLblsInMask(flags, Point(x, y), true);
			showImage();
		}
		break;
	}
}
int MyGCApplication::nextIter()   // 返回值是总的迭代次数
{
	if (isInitialized)
		myGrabCut(*image, mask, rect, bgdModel, fgdModel, 3, GC_EVAL);  // 默认是GC_EVAL
	else
	{
		if (rectState != SET)  // 如果还没设置矩形框
			return iterCount;
		if (lblsState == SET || prLblsState == SET)  // 如果在矩形框内左键或右键标注
			myGrabCut(*image, mask, rect, bgdModel, fgdModel, 3, GC_INIT_WITH_MASK);
		else  // 如果只是矩形框标注好了，没有进行相关标注
			myGrabCut(*image, mask, rect, bgdModel, fgdModel, 3, GC_INIT_WITH_RECT);
		isInitialized = true;
	}
	iterCount++;
	// 这里是清空掉每次绘图的项
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();
	return iterCount;
}

// 保存结果生成的mask遮罩图片，用于poisson融合
void MyGCApplication::saveBinaryMaskImg(string maskFileName)
{
	Mat resMask = Mat(image->rows, image->cols, CV_8UC3, Scalar(0, 0, 0));
	
	Point p;
	for (p.y = 0; p.y < mask.rows; p.y++)
	{
		for (p.x = 0; p.x < mask.cols; p.x++)
		{
			// 如果属于前景或者是可能的前景，就置为红色
			if (mask.at<uchar>(p) == GC_FGD || mask.at<uchar>(p) == GC_PR_FGD)
			{
				resMask.at<Vec3b>(p)[0] = 0;
				resMask.at<Vec3b>(p)[1] = 0 ;
				resMask.at<Vec3b>(p)[2] = 255;
			}
		}
	}
	// 保存resMask为png图片
	cv::imwrite(maskFileName, resMask);
	cout<<"保存成功！\n";
}

void MyGCApplication::help()
{
	cout << "\n这是对GrabCut论文的复现程序\n"
		"\n参考:\n\n"
		"1.OpenCV中的grabcut的源代码\n"
		"2.GraphCut论文中Max-flow/min-cut现成库函数\n"
		"\n请勾选一个矩形框，使得前景目标在该矩形框内\n\n"
		"相关操作:\n\n"
		"\t ESC-退出程序\n"
		"\t n---执行下一次分割\n"
		"\t m---保存当前分割对应的mask图片\n"
		"\t r---重载勾选图片\n"
		"\t ctrl+鼠标左键---确定在背景中的像素\t\tGC_BGD\n"
		"\t shift+鼠标左键---确定在前景（目标）中的像素\tGC_FGD\n";
		/*"\t ctrl+鼠标右键---可能在背景中的像素\t\tGC_PR_BGD\n"
		"\t shift+鼠标右键---可能在前景（目标）中的像素\tGC_PR_FGD\n\n";*/
}
