//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//int main()
//{
//	// ��ȡԴͼ��ת��Ϊ�Ҷ�ͼ��
//	cv::Mat srcImage = cv::imread("F:\\C++����\\MyGrabcut\\123.jpg");
//	// �ж��ļ��Ƿ������ȷ
//	if (!srcImage.data)
//		return 1;
//	// ͼ����ʾ
//	cv::imshow("srcImage", srcImage);
//	// �ȴ����̼���
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

	//g->add_tweights(0,   /* capacities */  1, 5);  // ��0��SOURCE��Ŀ�꣩��ȨֵΪ1����0��SINK����������ȨֵΪ5
	//g->add_tweights(1,   /* capacities */  2, 6);  
	//g->add_edge(0, 1,    /* capacities */  3, 4);  // ��i����j��ȨֵΪ3�������ȨֵΪ4

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
