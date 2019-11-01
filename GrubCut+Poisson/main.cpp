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
	//  x��y�����ָ����ͼ������ϵ�����꣬�������������ڵ�����
	gcapp.mouseClick(event, x, y, flags, param);
}
void grabCutInteractive()
{
	string filename /*= "bear.png"*/;
	cout<<"��������ָ��ͼƬ(������׺��)\n";
	cin>>filename;
	if (filename.empty())
	{
		cout << "\n�ļ���Ϊ��" << endl;
		return;
	}
	Mat image = imread(filename, 1);
	if (image.empty())
	{
		cout << "\n ������ͼƬ: " << filename << endl;
		return;
	}
	// ���maskͼƬ��
	string maskFileName;
	string::size_type pos = filename.find(".");
	maskFileName = filename.substr(0, pos) + "_mask.png";

	gcapp.help();  // ��������
	const string winName = "GrabCutǰ���Զ��ָ�";
	namedWindow(winName, WINDOW_AUTOSIZE);  // ��������
											//  �������¼�
	setMouseCallback(winName, on_mouse, 0);
	// ��ʼ��ͼƬ�������Ա����������mask��ֵΪȫ����
	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();
	for (;;)
	{
		char c = (char)waitKey(0);
		switch (c)
		{
		case '\x1b':
			cout << "�˳� ..." << endl;
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
	// �����û��������
	pe.setInputParam();
	// �����ں�
	pe.poisson_blending();
}
void printOperationHelp()
{
	cout<< "******************* GrabCut+PoissonCloning ********************\n";
	cout<< "******˵�����˳������������\t\t\t\t*******\n";
	cout<< "******\t\t1��ͼ��ǰ���Զ��ָ����GrabCut)\t*******\n";
	cout<< "******\t\t2��ͼ���޷��ں�(����poisson�������)\t*******\n";
	cout << "***************************************************************\n";
	cout<< "\n������ʾ��\n";
	cout<< "1-ͼ��ǰ���Զ��ָ�\n";
	cout<< "2-ͼ���޷��ں�\n";
	cout <<"0-�˳�����\n";
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





