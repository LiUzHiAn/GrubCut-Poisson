#include "myGCApplication.h"
#include "grabCutUtils.h"
#include <opencv2\highgui.hpp>

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);
const int BGD_KEY = EVENT_FLAG_CTRLKEY;  //Ctrl��
const int FGD_KEY = EVENT_FLAG_SHIFTKEY; //Shift��

void MyGCApplication::reset()
{
	if (!mask.empty())  // ���maskȫΪ����
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
	// ��ͼƬ�Ӳ����ж�����������Ϊ���Ա�������Ժ�Ĳ�������image��
	image = &_image;
	winName = &_winName;
	// ����һ��mask��ͼƬһ����С��mask
	mask.create(image->size(), CV_8UC1);
	reset();   // ��һЩ״̬���
}
void MyGCApplication::showImage() const
{
	if (image->empty() || winName->empty())
		return;
	// �����ʾ�����ͼƬ����ʼ��Ϊȫ�ף�
	Mat res=Mat(image->rows, image->cols, CV_8UC3, Scalar(255, 255, 255));
	// res.create(image->rows, image->cols, CV_8UC3, Scalar(255, 255, 255));
	Mat binMask;
	if (!isInitialized) // ���û�г�ʼ����ֱ�ӽ�ͼƬ��Ϊ���
		image->copyTo(res);
	else
	{
		// ���ݷָ�����mask����4��ȡֵ�����õ������ͼƬ�ϵ�����binMask
		getBinMask(mask, binMask);
		// binMask��Ϊһ����ģ�壬�����ĳ�����ص�(i, j)��ֵΪ1�����srcImage.at(i, j)����ֱֵ�Ӹ���dstImage.at(i, j)��
		// �����ֵΪ0��dstImage.at(i, j)��������ԭʼ����ֵ��
		image->copyTo(res, binMask);
	}
	vector<Point>::const_iterator it;
	/*����4������ǽ�ѡ�û�ѡ���4�ֵ��ò�ͬ����ɫ��ʾ����*/
	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)
		circle(res, *it, radius, BLUE, thickness);  // ��������ɫ
	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)
		circle(res, *it, radius, RED, thickness); // ǰ���Ǻ�ɫ
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness); // ���ܵı����ǵ���ɫ
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);  // ���ܵ�ǰ���Ƿ�ɫ

	// ������ڻ������Ѿ������˾��ο򣬾ͰѾ��ο�����ɫ�ʻ�����
	if (rectState == IN_PROCESS || rectState == SET)
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
	imshow(*winName, res);
}
void MyGCApplication::setRectInMask()
{
	CV_Assert(!mask.empty());
	mask.setTo(GC_BGD);
	// �����ο���ȫ������ΪGC_PR_FGD������Ȧ����ΪGC_BGD
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	// ����mask��RIO
	(mask(rect)).setTo(Scalar(GC_PR_FGD));
}
void MyGCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if (!isPr)  // ȷ���ĵ㣨������+ctrl/shift��

	// ctrl ����  shift  ǰ��
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;
		fvalue = GC_FGD;
	}
	else  // ���ܵĵ� ������Ҽ�+ctrl/shift��
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD;
		fvalue = GC_PR_FGD;
	}
	if (flags & BGD_KEY) // ctrl��
	{
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);  // �ı�mask��ֵ
	}
	if (flags & FGD_KEY) // shift��
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
		// ����������
	case EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,  // �Ұ�����ctrl���������ڿ���ѡ����
			isf = (flags & FGD_KEY) != 0;
		// ���ctrl��shift��û���£��Ҿ��ο�Ҳ��ûѡ���ǿ϶��Ǽ���Ҫ��ʼѡ���ο�
		if (rectState == NOT_SET && !isb && !isf)
		{
			rectState = IN_PROCESS;
			rect = Rect(x, y, 1, 1);
		}
		// ������ο�ѡ���ˣ��Ұ�����ctrl��shift�����ǿ϶�����ѡ�õľ��ο����û��ټ�mask
		// ��˵�����ڽ���lblsState����
		if ((isb || isf) && rectState == SET)
			lblsState = IN_PROCESS;
	}
	break;
	// ����Ҽ�����
	case EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET)
			prLblsState = IN_PROCESS;
	}
	break;
	// ������̧��
	case EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)  // ������ڻ����ο�ʱ̧��
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			rectState = SET; // ���ο򻭺���
			setRectInMask(); // �����ο���ȫ����Ϊ��������������ΪGC_PR_FGD
			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		if (lblsState == IN_PROCESS)
		{
			// �Ѿ��ο�������Ϊ�����ĵ���뵽��Ӧ��mask��ȥ
			setLblsInMask(flags, Point(x, y), false);
			lblsState = SET;
			showImage();
		}
		break;
		// ����Ҽ�̧��
	case EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			// �Ѿ��ο�������Ϊ�����ĵ���뵽��Ӧ��mask��ȥ
			setLblsInMask(flags, Point(x, y), true);
			prLblsState = SET;
			showImage();
		}
		break;
		// ����ƶ��¼�
	case EVENT_MOUSEMOVE: // ����ƶ�ʱ�����뽫���ο��Ȼ���
		if (rectState == IN_PROCESS)  // ����Ǿ��ο򻹴����ڻ���״̬
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		else if (lblsState == IN_PROCESS)  //���������ƶ�
		{
			setLblsInMask(flags, Point(x, y), false);
			showImage();
		}
		else if (prLblsState == IN_PROCESS)  //������Ҽ��ƶ�
		{
			setLblsInMask(flags, Point(x, y), true);
			showImage();
		}
		break;
	}
}
int MyGCApplication::nextIter()   // ����ֵ���ܵĵ�������
{
	if (isInitialized)
		myGrabCut(*image, mask, rect, bgdModel, fgdModel, 3, GC_EVAL);  // Ĭ����GC_EVAL
	else
	{
		if (rectState != SET)  // �����û���þ��ο�
			return iterCount;
		if (lblsState == SET || prLblsState == SET)  // ����ھ��ο���������Ҽ���ע
			myGrabCut(*image, mask, rect, bgdModel, fgdModel, 3, GC_INIT_WITH_MASK);
		else  // ���ֻ�Ǿ��ο��ע���ˣ�û�н�����ر�ע
			myGrabCut(*image, mask, rect, bgdModel, fgdModel, 3, GC_INIT_WITH_RECT);
		isInitialized = true;
	}
	iterCount++;
	// ��������յ�ÿ�λ�ͼ����
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();
	return iterCount;
}

// ���������ɵ�mask����ͼƬ������poisson�ں�
void MyGCApplication::saveBinaryMaskImg(string maskFileName)
{
	Mat resMask = Mat(image->rows, image->cols, CV_8UC3, Scalar(0, 0, 0));
	
	Point p;
	for (p.y = 0; p.y < mask.rows; p.y++)
	{
		for (p.x = 0; p.x < mask.cols; p.x++)
		{
			// �������ǰ�������ǿ��ܵ�ǰ��������Ϊ��ɫ
			if (mask.at<uchar>(p) == GC_FGD || mask.at<uchar>(p) == GC_PR_FGD)
			{
				resMask.at<Vec3b>(p)[0] = 0;
				resMask.at<Vec3b>(p)[1] = 0 ;
				resMask.at<Vec3b>(p)[2] = 255;
			}
		}
	}
	// ����resMaskΪpngͼƬ
	cv::imwrite(maskFileName, resMask);
	cout<<"����ɹ���\n";
}

void MyGCApplication::help()
{
	cout << "\n���Ƕ�GrabCut���ĵĸ��ֳ���\n"
		"\n�ο�:\n\n"
		"1.OpenCV�е�grabcut��Դ����\n"
		"2.GraphCut������Max-flow/min-cut�ֳɿ⺯��\n"
		"\n�빴ѡһ�����ο�ʹ��ǰ��Ŀ���ڸþ��ο���\n\n"
		"��ز���:\n\n"
		"\t ESC-�˳�����\n"
		"\t n---ִ����һ�ηָ�\n"
		"\t m---���浱ǰ�ָ��Ӧ��maskͼƬ\n"
		"\t r---���ع�ѡͼƬ\n"
		"\t ctrl+������---ȷ���ڱ����е�����\t\tGC_BGD\n"
		"\t shift+������---ȷ����ǰ����Ŀ�꣩�е�����\tGC_FGD\n";
		/*"\t ctrl+����Ҽ�---�����ڱ����е�����\t\tGC_PR_BGD\n"
		"\t shift+����Ҽ�---������ǰ����Ŀ�꣩�е�����\tGC_PR_FGD\n\n";*/
}
