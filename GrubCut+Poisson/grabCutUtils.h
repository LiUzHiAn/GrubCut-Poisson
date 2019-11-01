#ifndef __UTILS__
#define __UTILS__

#include "myGMM.h"
#include "graph.h"
#include <time.h>  
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

//GC_BGD = 0,			//背景
//GC_FGD = 1,			//前景 
//GC_PR_BGD = 2,		//可能背景
//GC_PR_FGD = 3			//可能前景 

// 用用户所选矩形框来设置mask遮罩，mask中目标为1，背景为0
static void initMaskWithRect(Mat& mask, Size imgSize, Rect rect)
{
	mask.create(imgSize, CV_8UC1);
	mask.setTo(GC_BGD);		// 先全部设置为背景

	// 对rect进行边界防越出处理
	rect.x = std::max(rect.x, 0);   // 左上角越界
	rect.y = std::max(rect.y, 0);
	rect.width = std::min(rect.width, imgSize.width - rect.x);   // 右下角越界
	rect.height = std::min(rect.height, imgSize.height - rect.y);

	// 把rect对应的mask取值设置为可能的前景
	(mask(rect)).setTo(Scalar(GC_PR_FGD));
}

// 检查mask的正确性。mask为通过用户交互或者程序设定的，它是和图像大小一样的单通道灰度图，
// 每个像素只能取GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD 四种枚举值，分别表示该像素
//（用户或者程序指定）属于背景、前景、可能为背景或者可能为前景像素.
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


// 平滑项中的beta参数
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
// 计算所有相邻像素的平滑项
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
* 论文中的初始化过程
*/
// 初始化一下目标和背景的GMM参数
static void initGMMS(const Mat& img, const Mat& mask, MyGMM& bgdGMM, MyGMM& fgdGMM)
{
	const int kMeansTimes = 10;					// k-means迭代次数
	const int kMeansType = KMEANS_PP_CENTERS;	// k-means++算法

												// 记录背景和前景的像素样本集中每个像素对应GMM的哪个高斯模型，论文中的kn
	Mat fgdLabels, bgdLabels;

	std::vector<Vec3f>  bgdSamples, fgdSamples; //背景和前景的像素样本集
	Point p;
	for (p.y = 0;p.y < img.rows;p.y++)
	{
		for (p.x = 0; p.x < img.cols; p.x++)
		{
			// mask中标记为GC_BGD和GC_PR_BGD的像素都作为背景的样本像素
			if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
			else // GC_FGD | GC_PR_FGD
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
		}
	}
	CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());

	// 运用k-means算法对各像素进行初始化聚类，确定其属于GMM中哪个高斯模型
	// _bgdSamples每一行是一个像素，一行有3列，分别表示RGB三个颜色，最后一个参数是默认值
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	// 分类结果保存在bgdLabels矩阵中，这是和图片等大的一个矩阵
	kmeans(_bgdSamples, MyGMM::componentNum, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansTimes, 0.0), 0, kMeansType);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, MyGMM::componentNum, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansTimes, 0.0), 0, kMeansType);

	// 进行第一次学习（将聚类的结果分配到GMM中的每个Ci高斯模型）
	bgdGMM.startLearning();  // 把背景GMM进行第一次学习
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSamplePixel(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.endLearning();

	fgdGMM.startLearning();  // 把目标GMM进行第一次学习
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSamplePixel(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.endLearning();
}





/*
*以下开始论文的整体迭代过程：
*	Step1：为每个像素分配一个GMM中的高斯模型，其下标k_n保存在Mat compIdxs中
*   Step2：从每个高斯模型的像素样本集中学习每个高斯模型的参数
*   Step3: 构造一个graph cut论文中的图，用的是max-flow/ min-cut分割
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
			// 每个像素点都有一个对应的，最可能的ci
			// 如果mask中给定是可能的背景或确定的背景，那就用背景的GMM去分配点的kn
			compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
				bgdGMM.whichComponent(pixel) : fgdGMM.whichComponent(pixel);
		}
	}
}

// step 2
static void learnGMMs(const Mat& img, const Mat& mask, const Mat& compIdxs, MyGMM& bgdGMM, MyGMM& fgdGMM)
{
	// 每次学习前将参数清零
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
	bgdGMM.endLearning();   // 更新GMM中高斯参数
	fgdGMM.endLearning();
}

// step 3.1 构造一个图
static void buildGCGraph(const Mat& img, const Mat& mask, MyGMM& bgdGMM, MyGMM& fgdGMM, double lambda,
	const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
	Graph<double, double, double> *graph)
{
	//// 顶点数
	//int nodeCount = img.cols*img.rows;
	//// 双向边
	//int edgeCount = 2 * (4 * img.cols*img.rows - 3 * (img.cols + img.rows) + 2);
	//
	//// graph=new Graph<double, double, double>(nodeCount,edgeCount);

	Point p;
	for (p.y = 0; p.y < img.rows; p.y++)
	{
		for (p.x = 0; p.x < img.cols; p.x++)
		{
			// 添加顶点
			int nodeIdx = graph->add_node();
			Vec3b pixel = img.at<Vec3b>(p);

			// 设置 t-weights
			double fromSource, toSink;
			// 如果p点已可能为背景或目标，也就是论文中的T_U区域
			if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD)
			{
				fromSource = -log(bgdGMM.calGMMPr(pixel));  // 求解属于背景GMM的概率
				toSink = -log(fgdGMM.calGMMPr(pixel));      // 求解属于目标GMM的概率
			}
			// 如果确定了是来自背景（这一般是用户指定）
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
			//计算两个邻域顶点之间连接的权值。
			//也即计算Gibbs能量的第二个能量项（平滑项）
			if (p.x>0)  // 有左侧的相邻边
			{
				double w = leftW.at<double>(p);
				graph->add_edge(nodeIdx, nodeIdx - 1, w, w);
			}
			if (p.x>0 && p.y>0)   // 有左上的相邻边
			{
				double w = upleftW.at<double>(p);
				graph->add_edge(nodeIdx, nodeIdx - img.cols - 1, w, w);
			}
			if (p.y>0)  // 有上面的相邻边
			{
				double w = upW.at<double>(p);
				graph->add_edge(nodeIdx, nodeIdx - img.cols, w, w);
			}
			if (p.x<img.cols - 1 && p.y>0)  // 有右上的相邻边
			{
				double w = uprightW.at<double>(p);
				graph->add_edge(nodeIdx, nodeIdx - img.cols + 1, w, w);
			}
		}
	}
}

// step 3.1 最小割或者最大流算法分割
static void estimateSeg(Graph<double, double, double> *graph, Mat& mask)
{
	graph->maxflow();

	// 分割后，更新mask
	Point p;
	for (p.y = 0; p.y < mask.rows; p.y++)
	{
		for (p.x = 0; p.x < mask.cols; p.x++)
		{
			//注意的是，永远都不会更新用户指定为背景或者前景的像素
			if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD)
			{
				if (graph->what_segment(p.y*mask.cols + p.x /*顶点的索引*/) == Graph<double, double, double>::SOURCE)
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

	MyGMM bgdGMM(bgdModel), fgdGMM(fgdModel);  // 创建两个GMM，其中参数是GMM的参数保存数据
	Mat compIdxs(img.size(), CV_32SC1); // 每个像素最可能对应的GMM模型中的ci下标

	if (mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK)
	{
		if (mode == GC_INIT_WITH_RECT)
			initMaskWithRect(mask, img.size(), rect);
		else // flag == GC_INIT_WITH_MASK
			checkMask(img, mask);
		// 初始化背景和目标的GMM (会用到k-means初始化各像素的kn)
		initGMMS(img, mask, bgdGMM, fgdGMM);
	}
	if (iterCount <= 0)
		return;
	if (mode == GC_EVAL_FREEZE_MODEL)  // 只执行一次
		iterCount = 1;
	// GC_EVAL--执行分割(默认值)  GC_EVAL_FREEZE_MODEL--只进行一次GrabCut算法迭代
	if (mode == GC_EVAL || mode == GC_EVAL_FREEZE_MODEL)
		checkMask(img, mask);

	const double gamma = 50;
	const double lambda = 9 * gamma;
	const double beta = calcBeta(img);  // 根据图片求平滑项中的beta参数

	Mat leftW, upleftW, upW, uprightW;
	// 计算n-weight，也就是所有相邻元素的能量之和，结果保存在上述的四个矩阵中
	calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);

	for (int i = 0;i < iterCount;i++)
	{
		//  顶点数
		int nodeCount = img.cols*img.rows;
		// 双向边
		int edgeCount = 2 * (4 * img.cols*img.rows - 3 * (img.cols + img.rows) + 2);

		Graph<double, double, double> *graph = new Graph<double, double, double>(nodeCount, edgeCount);

		
		// 更新每个像素属于GMM的那个Ci模型，即Kn
		// 注意，这里会根据mask进行判断，如果是前景，就说明当前在对前景的GMM进行更新，背景同理。
		assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);

		// 如果不是仅仅一次的分割
		if (mode != GC_EVAL_FREEZE_MODEL)
			// 再对GMM的参数进行更新
			learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
		
		// 构造一个graph cut图
		buildGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
		// 最小割或者最大流算法更新mask
		estimateSeg(graph, mask);

		//// 保存中间结果
		//string cur_time= std::to_string(time(NULL));
		//// 保存mask到文件
		//string maskFileName = "mask" + cur_time;
		//ofstream maskFile(maskFileName + ".csv");
		//maskFile << format(mask, Formatter::FMT_CSV);
		//maskFile.close();

		//// 保存compoIdx到文件
		//string compoIdxFileName = "compoIdx" + cur_time;
		//ofstream compoIdxFile(compoIdxFileName + ".csv");
		//compoIdxFile << format(mask, Formatter::FMT_CSV);
		//compoIdxFile.close();

		delete graph;
	}


}

static void getBinMask(const Mat& comMask, Mat& binMask)
{
	// comMask是分割完成后mask矩阵，要显示最后的分割图像，将mask和1相与
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;  //得到mask的最低位,实际上是只保留确定的或者有可能的前景点当做mask
}

#endif 
