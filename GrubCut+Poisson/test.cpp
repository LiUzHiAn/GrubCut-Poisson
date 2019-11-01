//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//int main()
//{
//	// 读取源图像并转化为灰度图像
//	cv::Mat srcImage = cv::imread("F:\\C++代码\\MyGrabcut\\123.jpg");
//	// 判断文件是否读入正确
//	if (!srcImage.data)
//		return 1;
//	// 图像显示
//	cv::imshow("srcImage", srcImage);
//	// 等待键盘键入
//	cv::waitKey(0);
//	return 0;
//}

#include <stdio.h>
#include <opencv2\imgproc.hpp>
#include "graph.h"

int main()
{
	//typedef Graph<int, int, int> GraphType;
	//GraphType *g = new GraphType(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);

	//g->add_node();
	//g->add_node();

	//g->add_tweights(0,   /* capacities */  1, 5);  // 点0到SOURCE（目标）边权值为1，点0到SINK（背景）边权值为5
	//g->add_tweights(1,   /* capacities */  2, 6);  
	//g->add_edge(0, 1,    /* capacities */  3, 4);  // 点i到点j的权值为3，反向边权值为4

	//int flow = g->maxflow();

	//printf("Flow = %d\n", flow);
	//printf("Minimum cut:\n");
	//if (g->what_segment(0) == GraphType::SOURCE)
	//	printf("node0 is in the SOURCE set\n");
	//else
	//	printf("node0 is in the SINK set\n");
	//if (g->what_segment(1) == GraphType::SOURCE)
	//	printf("node1 is in the SOURCE set\n");
	//else
	//	printf("node1 is in the SINK set\n");

	//delete g;
	

	return 0;
}
