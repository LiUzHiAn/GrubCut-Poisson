#ifndef __UTILS__
#define __UTILS__

#include "myGMM.h"
#include "graph.h"
#include <time.h>  
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

//GC_BGD = 0,			//����
//GC_FGD = 1,			//ǰ�� 
//GC_PR_BGD = 2,		//���ܱ���
//GC_PR_FGD = 3			//����ǰ�� 

// ���û���ѡ���ο�������mask���֣�mask��Ŀ��Ϊ1������Ϊ0
static void initMaskWithRect(Mat& mask, Size imgSize, Rect rect)
{
	mask.create(imgSize, CV_8UC1);
	mask.setTo(GC_BGD);		// ��ȫ������Ϊ����

	// ��rect���б߽��Խ������
	rect.x = std::max(rect.x, 0);   // ���Ͻ�Խ��
	rect.y = std::max(rect.y, 0);
	rect.width = std::min(rect.width, imgSize.width - rect.x);   // ���½�Խ��
	rect.height = std::min(rect.height, imgSize.height - rect.y);

	// ��rect��Ӧ��maskȡֵ����Ϊ���ܵ�ǰ��
	(mask(rect)).setTo(Scalar(GC_PR_FGD));
}

// ���mask����ȷ�ԡ�maskΪͨ���û��������߳����趨�ģ����Ǻ�ͼ���Сһ���ĵ�ͨ���Ҷ�ͼ��
// ÿ������ֻ��ȡGC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD ����ö��ֵ���ֱ��ʾ������
//���û����߳���ָ�������ڱ�����ǰ��������Ϊ�������߿���Ϊǰ������.
static void checkMask(const Mat& img, const Mat& mask)
{
	if (mask.empty())
		CV_Error(CV_StsBadArg, "mask is empty");
	if (mask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "mask must have CV_8UC1 type");
	if (mask.cols != img.cols || mask.rows != img.rows)
		CV_Error(CV_StsBadArg, "mask must have as many rows and cols as img");
	for (int y = 0; y < mask.rows; y++)
	{
		for (int x = 0; x < mask.cols; x++)
		{
			uchar val = mask.at<uchar>(y, x);
			if (val != GC_BGD && val != GC_FGD && val != GC_PR_BGD && val != GC_PR_FGD)
				CV_Error(CV_StsBadArg, "mask element value must be equal "
					"GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD");
		}
	}
}


// ƽ�����е�beta����
/*
beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta(const Mat& img)
{
	double beta = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			Vec3d color = img.at<Vec3b>(y, x);
			if (x>0) // left
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				beta += diff.dot(diff);
			}
			if (y>0 && x>0) // upleft
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
				beta += diff.dot(diff);
			}
			if (y>0) // up
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
				beta += diff.dot(diff);
			}
			if (y>0 && x<img.cols - 1) // upright
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
				beta += diff.dot(diff);
			}
		}
	}
	if (beta <= std::numeric_limits<double>::epsilon())
		beta = 0;
	else
		beta = 1.f / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2));

	return beta;
}

/*
Calculate weights of non-terminal vertices of graph.
beta and gamma - parameters of GrabCut algorithm.
*/
// ���������������ص�ƽ����
static void calcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma)
{
	const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
	leftW.create(img.rows, img.cols, CV_64FC1);
	upleftW.create(img.rows, img.cols, CV_64FC1);
	upW.create(img.rows, img.cols, CV_64FC1);
	uprightW.create(img.rows, img.cols, CV_64FC1);
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			Vec3d color = img.at<Vec3b>(y, x);
			if (x - 1 >= 0) // left
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				leftW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
			}
			else
				leftW.at<double>(y, x) = 0;
			if (x - 1 >= 0 && y - 1 >= 0) // upleft
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
				upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
			}
			else
				upleftW.at<double>(y, x) = 0;
			if (y - 1 >= 0) // up
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
				upW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
			}
			else
				upW.at<double>(y, x) = 0;
			if (x + 1<img.cols && y - 1 >= 0) // upright
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
				uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
			}
			else
				uprightW.at<double>(y, x) = 0;
		}
	}
}

/*
* �����еĳ�ʼ������
*/
// ��ʼ��һ��Ŀ��ͱ�����GMM����
static void initGMMS(const Mat& img, const Mat& mask, MyGMM& bgdGMM, MyGMM& fgdGMM)
{
	const int kMeansTimes = 10;					// k-means��������
	const int kMeansType = KMEANS_PP_CENTERS;	// k-means++�㷨

												// ��¼������ǰ����������������ÿ�����ض�ӦGMM���ĸ���˹ģ�ͣ������е�kn
	Mat fgdLabels, bgdLabels;

	std::vector<Vec3f>  bgdSamples, fgdSamples; //������ǰ��������������
	Point p;
	for (p.y = 0;p.y < img.rows;p.y++)
	{
		for (p.x = 0; p.x < img.cols; p.x++)
		{
			// mask�б��ΪGC_BGD��GC_PR_BGD�����ض���Ϊ��������������
			if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
			else // GC_FGD | GC_PR_FGD
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
		}
	}
	CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());

	// ����k-means�㷨�Ը����ؽ��г�ʼ�����࣬ȷ��������GMM���ĸ���˹ģ��
	// _bgdSamplesÿһ����һ�����أ�һ����3�У��ֱ��ʾRGB������ɫ�����һ��������Ĭ��ֵ
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	// ������������bgdLabels�����У����Ǻ�ͼƬ�ȴ��һ������
	kmeans(_bgdSamples, MyGMM::componentNum, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansTimes, 0.0), 0, kMeansType);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, MyGMM::componentNum, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansTimes, 0.0), 0, kMeansType);

	// ���е�һ��ѧϰ��������Ľ�����䵽GMM�е�ÿ��Ci��˹ģ�ͣ�
	bgdGMM.startLearning();  // �ѱ���GMM���е�һ��ѧϰ
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSamplePixel(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.endLearning();

	fgdGMM.startLearning();  // ��Ŀ��GMM���е�һ��ѧϰ
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSamplePixel(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.endLearning();
}





/*
*���¿�ʼ���ĵ�����������̣�
*	Step1��Ϊÿ�����ط���һ��GMM�еĸ�˹ģ�ͣ����±�k_n������Mat compIdxs��
*   Step2����ÿ����˹ģ�͵�������������ѧϰÿ����˹ģ�͵Ĳ���
*   Step3: ����һ��graph cut�����е�ͼ���õ���max-flow/ min-cut�ָ�
*/

// step 1
static void assignGMMsComponents(const Mat& img, const Mat& mask, MyGMM& bgdGMM, MyGMM& fgdGMM, Mat& compIdxs)
{
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++)
	{
		for (p.x = 0; p.x < img.cols; p.x++)
		{
			Vec3d pixel = img.at<Vec3b>(p);
			// ÿ�����ص㶼��һ����Ӧ�ģ�����ܵ�ci
			// ���mask�и����ǿ��ܵı�����ȷ���ı������Ǿ��ñ�����GMMȥ������kn
			compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
				bgdGMM.whichComponent(pixel) : fgdGMM.whichComponent(pixel);
		}
	}
}

// step 2
static void learnGMMs(const Mat& img, const Mat& mask, const Mat& compIdxs, MyGMM& bgdGMM, MyGMM& fgdGMM)
{
	// ÿ��ѧϰǰ����������
	bgdGMM.startLearning();
	fgdGMM.startLearning();
	Point p;
	for (int ci = 0; ci < MyGMM::componentNum; ci++)
	{
		for (p.y = 0; p.y < img.rows; p.y++)
		{
			for (p.x = 0; p.x < img.cols; p.x++)
			{
				if (compIdxs.at<int>(p) == ci)
				{
					if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
						bgdGMM.addSamplePixel(ci, img.at<Vec3b>(p));
					else
						fgdGMM.addSamplePixel(ci, img.at<Vec3b>(p));
				}
			}
		}
	}
	bgdGMM.endLearning();   // ����GMM�и�˹����
	fgdGMM.endLearning();
}

// step 3.1 ����һ��ͼ
static void buildGCGraph(const Mat& img, const Mat& mask, MyGMM& bgdGMM, MyGMM& fgdGMM, double lambda,
	const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
	Graph<double, double, double> *graph)
{
	//// ������
	//int nodeCount = img.cols*img.rows;
	//// ˫���
	//int edgeCount = 2 * (4 * img.cols*img.rows - 3 * (img.cols + img.rows) + 2);
	//
	//// graph=new Graph<double, double, double>(nodeCount,edgeCount);

	Point p;
	for (p.y = 0; p.y < img.rows; p.y++)
	{
		for (p.x = 0; p.x < img.cols; p.x++)
		{
			// ��Ӷ���
			int nodeIdx = graph->add_node();
			Vec3b pixel = img.at<Vec3b>(p);

			// ���� t-weights
			double fromSource, toSink;
			// ���p���ѿ���Ϊ������Ŀ�꣬Ҳ���������е�T_U����
			if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD)
			{
				fromSource = -log(bgdGMM.calGMMPr(pixel));  // ������ڱ���GMM�ĸ���
				toSink = -log(fgdGMM.calGMMPr(pixel));      // �������Ŀ��GMM�ĸ���
			}
			// ���ȷ���������Ա�������һ�����û�ָ����
			else if (mask.at<uchar>(p) == GC_BGD)
			{
				fromSource = 0;
				toSink = lambda;
			}
			else // GC_FGD
			{
				fromSource = lambda;
				toSink = 0;
			}
			graph->add_tweights(nodeIdx, fromSource, toSink);

			// set n-weights
			//�����������򶥵�֮�����ӵ�Ȩֵ��
			//Ҳ������Gibbs�����ĵڶ��������ƽ���
			if (p.x>0)  // ���������ڱ�
			{
				double w = leftW.at<double>(p);
				graph->add_edge(nodeIdx, nodeIdx - 1, w, w);
			}
			if (p.x>0 && p.y>0)   // �����ϵ����ڱ�
			{
				double w = upleftW.at<double>(p);
				graph->add_edge(nodeIdx, nodeIdx - img.cols - 1, w, w);
			}
			if (p.y>0)  // ����������ڱ�
			{
				double w = upW.at<double>(p);
				graph->add_edge(nodeIdx, nodeIdx - img.cols, w, w);
			}
			if (p.x<img.cols - 1 && p.y>0)  // �����ϵ����ڱ�
			{
				double w = uprightW.at<double>(p);
				graph->add_edge(nodeIdx, nodeIdx - img.cols + 1, w, w);
			}
		}
	}
}

// step 3.1 ��С�����������㷨�ָ�
static void estimateSeg(Graph<double, double, double> *graph, Mat& mask)
{
	graph->maxflow();

	// �ָ�󣬸���mask
	Point p;
	for (p.y = 0; p.y < mask.rows; p.y++)
	{
		for (p.x = 0; p.x < mask.cols; p.x++)
		{
			//ע����ǣ���Զ����������û�ָ��Ϊ��������ǰ��������
			if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD)
			{
				if (graph->what_segment(p.y*mask.cols + p.x /*���������*/) == Graph<double, double, double>::SOURCE)
					mask.at<uchar>(p) = GC_PR_FGD;
				else
					mask.at<uchar>(p) = GC_PR_BGD;
			}
		}
	}
}

static void myGrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
	InputOutputArray _bgdModel, InputOutputArray _fgdModel,
	int iterCount, int mode)
{
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();

	if (img.empty())
		CV_Error(CV_StsBadArg, "image is empty");
	if (img.type() != CV_8UC3)
		CV_Error(CV_StsBadArg, "image must have CV_8UC3 type");

	MyGMM bgdGMM(bgdModel), fgdGMM(fgdModel);  // ��������GMM�����в�����GMM�Ĳ�����������
	Mat compIdxs(img.size(), CV_32SC1); // ÿ����������ܶ�Ӧ��GMMģ���е�ci�±�

	if (mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK)
	{
		if (mode == GC_INIT_WITH_RECT)
			initMaskWithRect(mask, img.size(), rect);
		else // flag == GC_INIT_WITH_MASK
			checkMask(img, mask);
		// ��ʼ��������Ŀ���GMM (���õ�k-means��ʼ�������ص�kn)
		initGMMS(img, mask, bgdGMM, fgdGMM);
	}
	if (iterCount <= 0)
		return;
	if (mode == GC_EVAL_FREEZE_MODEL)  // ִֻ��һ��
		iterCount = 1;
	// GC_EVAL--ִ�зָ�(Ĭ��ֵ)  GC_EVAL_FREEZE_MODEL--ֻ����һ��GrabCut�㷨����
	if (mode == GC_EVAL || mode == GC_EVAL_FREEZE_MODEL)
		checkMask(img, mask);

	const double gamma = 50;
	const double lambda = 9 * gamma;
	const double beta = calcBeta(img);  // ����ͼƬ��ƽ�����е�beta����

	Mat leftW, upleftW, upW, uprightW;
	// ����n-weight��Ҳ������������Ԫ�ص�����֮�ͣ�����������������ĸ�������
	calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);

	for (int i = 0;i < iterCount;i++)
	{
		//  ������
		int nodeCount = img.cols*img.rows;
		// ˫���
		int edgeCount = 2 * (4 * img.cols*img.rows - 3 * (img.cols + img.rows) + 2);

		Graph<double, double, double> *graph = new Graph<double, double, double>(nodeCount, edgeCount);

		
		// ����ÿ����������GMM���Ǹ�Ciģ�ͣ���Kn
		// ע�⣬��������mask�����жϣ������ǰ������˵����ǰ�ڶ�ǰ����GMM���и��£�����ͬ��
		assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);

		// ������ǽ���һ�εķָ�
		if (mode != GC_EVAL_FREEZE_MODEL)
			// �ٶ�GMM�Ĳ������и���
			learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
		
		// ����һ��graph cutͼ
		buildGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
		// ��С�����������㷨����mask
		estimateSeg(graph, mask);

		//// �����м���
		//string cur_time= std::to_string(time(NULL));
		//// ����mask���ļ�
		//string maskFileName = "mask" + cur_time;
		//ofstream maskFile(maskFileName + ".csv");
		//maskFile << format(mask, Formatter::FMT_CSV);
		//maskFile.close();

		//// ����compoIdx���ļ�
		//string compoIdxFileName = "compoIdx" + cur_time;
		//ofstream compoIdxFile(compoIdxFileName + ".csv");
		//compoIdxFile << format(mask, Formatter::FMT_CSV);
		//compoIdxFile.close();

		delete graph;
	}


}

static void getBinMask(const Mat& comMask, Mat& binMask)
{
	// comMask�Ƿָ���ɺ�mask����Ҫ��ʾ���ķָ�ͼ�񣬽�mask��1����
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;  //�õ�mask�����λ,ʵ������ֻ����ȷ���Ļ����п��ܵ�ǰ���㵱��mask
}

#endif 
