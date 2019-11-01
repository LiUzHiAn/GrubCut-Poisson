#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include "graph.h"
#include <fstream>
#include <iostream>
#include "myGCApplication.h"
#include "poissonEditer.h"

using namespace std;
using namespace cv;

MyGCApplication gcapp;
poissonEditer pe;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
	//  x和y是鼠标指针在图像坐标系的坐标，并不是整个窗口的坐标
	gcapp.mouseClick(event, x, y, flags, param);
}
void grabCutInteractive()
{
	string filename /*= "bear.png"*/;
	cout<<"请输入待分割的图片(包括后缀名)\n";
	cin>>filename;
	if (filename.empty())
	{
		cout << "\n文件名为空" << endl;
		return;
	}
	Mat image = imread(filename, 1);
	if (image.empty())
	{
		cout << "\n 不存在图片: " << filename << endl;
		return;
	}
	// 输出mask图片名
	string maskFileName;
	string::size_type pos = filename.find(".");
	maskFileName = filename.substr(0, pos) + "_mask.png";

	gcapp.help();  // 帮助函数
	const string winName = "GrabCut前后景自动分割";
	namedWindow(winName, WINDOW_AUTOSIZE);  // 命名窗口
											//  鼠标监听事件
	setMouseCallback(winName, on_mouse, 0);
	// 初始化图片矩阵到类成员变量，并给mask赋值为全背景
	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();
	for (;;)
	{
		char c = (char)waitKey(0);
		switch (c)
		{
		case '\x1b':
			cout << "退出 ..." << endl;
			destroyWindow(winName);
			return;
		case 'r':
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			break;
		case 'm':
			gcapp.saveBinaryMaskImg(maskFileName);
			break;
		case 'n':
			int iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			int newIterCount = gcapp.nextIter();
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}
			else
				cout << "rect must be determined>" << endl;
			break;

		}
	}
}


void poissonBlendingInteractive()
{
	// 处理用户输入参数
	pe.setInputParam();
	// 泊松融合
	pe.poisson_blending();
}
void printOperationHelp()
{
	cout<< "******************* GrabCut+PoissonCloning ********************\n";
	cout<< "******说明：此程序包含两大功能\t\t\t\t*******\n";
	cout<< "******\t\t1、图像前后景自动分割（基于GrabCut)\t*******\n";
	cout<< "******\t\t2、图像无缝融合(基于poisson方程求解)\t*******\n";
	cout << "***************************************************************\n";
	cout<< "\n操作提示：\n";
	cout<< "1-图像前后景自动分割\n";
	cout<< "2-图像无缝融合\n";
	cout <<"0-退出程序\n";
}

int main()
{
	char c;
	while(1)
	{
		printOperationHelp();	
		cin >> c;
		switch (c)
		{
		case '1':
			grabCutInteractive();
			break;
		case '2':
			poissonBlendingInteractive();
			break;
		case '0':
			return 0;
		default:
			break;
		}
	}
	return 0;
}





