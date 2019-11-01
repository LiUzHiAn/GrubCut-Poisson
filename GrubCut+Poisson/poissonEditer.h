#pragma once

#include "poissonUtils.h"
#include <Eigen/Sparse>



class poissonEditer
{

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::VectorXd Vec;
public:
	// 执行泊松融合操作
	void poisson_blending() {							
		// 载入3张图片
		loadImage(targetFile.c_str(), targetImage);
		loadImage(maskFile.c_str(), maskImage);
		loadImage(sourceFile.c_str(), sourceImage);

		// 对mx和my进行检查（防止出现在边界上的情况）  630 600 520 291
		{
			unsigned int xmin = mx;
			unsigned int ymin = my;

			unsigned int xmax = mx + maskImage.width;
			unsigned int ymax = my + maskImage.height;

			if (xmin > 0 && ymin > 0 && xmax < targetImage.width - 1 && ymax < targetImage.height - 1) {
				// 检查通过
			}
			else
			{ 
				printf("指定融合的位置(%d,%d)无法容纳下待融合的图片\n", mx, my);
				return;
			}
		}
		cout << "正在融合...\n";
		// key是像素在整个mask图片中的索引（行优先顺序），value是像素在Ω区域的索引
		std::map<unsigned int, unsigned int> varMap;
		{
			int i = 0;
			for (unsigned int y = 0; y < maskImage.height; ++y) {
				for (unsigned int x = 0; x < maskImage.width; ++x) {
					if (isMaskPixel(x, y)) {
						varMap[maskFlatten(x, y)] = i;
						++i;
					}
				}
			}
		}
		const unsigned int numUnknowns = (unsigned int)varMap.size();  //  Ω区域的像素个数

		 // 构造泊松线性方程组 Mx=b 的系数矩阵M（稀疏矩阵，用Eigen中的Triplet来构造）
		std::vector<Triplet> mt;
		{
			unsigned int irow = 0;
			for (unsigned int y = my; y < my + maskImage.height; ++y) {
				for (unsigned int x = mx; x < mx + maskImage.width; ++x) {
					// 如果改点在Ω区域内
					if (isMaskPixel(x - mx, y - my)) {
						/*
						详细求解泊松方程公式见论文（7）式，因为这里不允许融合在目标图片的边缘，所以|N_p| 恒等于 4
						*/
						mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - my)], 4)); // |N_p| = 4.

						// 判断相邻4个像素是否在Ω区域，在的话，就添加到方程左边，否则的话，必然是在方程组的右边（即已知量边界b）
						// 上方像素点
						if (isMaskPixel(x - mx, y - my - 1)) {
							mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - 1 - my)], -1));
						}
						// 右方像素点
						if (isMaskPixel(x - mx + 1, y - my)) {
							mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx + 1, y - my)], -1));
						}
						// 下方像素点
						if (isMaskPixel(x - mx, y - my + 1)) {
							mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - my + 1)], -1));
						}
						// 左方像素点
						if (isMaskPixel(x - mx - 1, y - my)) {
							mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx - 1, y - my)], -1));
						}

						++irow; // 下一个线性方程
					}
				}
			}
		}

		// 根据线性方程组的特性可知，系数矩阵M是一个对称正定矩阵，可以用Cholesky分解快速求解X
		// 因为是RGB颜色空间，会用到M矩阵求解3次线性方程组
		Eigen::SimplicialCholesky<SpMat> solver;
		{
			// 稀疏矩阵，方程个数为Ω区域的像素个数
			SpMat mat(numUnknowns, numUnknowns);
			mat.setFromTriplets(mt.begin(), mt.end());
			// Cholesky分解
			solver.compute(mat);
		}

		Vec solutionChannels[3];
		Vec b(numUnknowns);
		// 求解每个通道的颜色都需要解一次线性方程组，b在每次求解时会变化
		for (unsigned int ic = 0; ic < 3; ++ic)
		{
			unsigned int irow = 0;

			for (unsigned int y = my; y < my + maskImage.height; ++y) {
				for (unsigned int x = mx; x < mx + maskImage.width; ++x) {
					// 如果改点在Ω区域内
					if (isMaskPixel(x - mx, y - my)) {
						// we only ended up using v in the end.
						// 源图像当前像素的颜色
						vec3 v = sourceImage.data[maskFlatten(x - mx, y - my)];
						// 目标图像当前像素的颜色
						vec3 u = targetImage.data[targetFlatten(x, y)];

						/* 论文公式(7)的右边项 */

						// 每个像素和它相邻像素的梯度v_pq之和，这里求解梯度有两种方法。
						// 1. 直接用源图片g中相邻像素的梯度
						// 2. 原图片g和目标图片f*中相邻像素的梯度，取较大那个  论文公式(13)
						float grad =
							vpq(
								u[ic], targetImage.data[targetFlatten(x, y - 1)][ic], // unused
								v[ic], sourceImage.data[maskFlatten(x - mx, y - 1 - my)][ic]) // used
							+
							vpq(
								u[ic], targetImage.data[targetFlatten(x - 1, y)][ic], // unused
								v[ic], sourceImage.data[maskFlatten(x - 1 - mx, y - my)][ic]) // used
							+
							vpq(
								u[ic], targetImage.data[targetFlatten(x, y + 1)][ic], // unused
								v[ic], sourceImage.data[maskFlatten(x - mx, y + 1 - my)][ic] // used
							)
							+
							vpq(
								u[ic], targetImage.data[targetFlatten(x + 1, y)][ic], // unused
								v[ic], sourceImage.data[maskFlatten(x + 1 - mx, y - my)][ic]); // used

						b[irow] = grad;

						/*
						边界条件在公式（7）右边项中的f*_q
						*/
						if (!isMaskPixel(x - mx, y - my - 1)) {
							b[irow] += targetImage.data[targetFlatten(x, y - 1)][ic];
						}
						if (!isMaskPixel(x - mx + 1, y - my)) {
							b[irow] += targetImage.data[targetFlatten(x + 1, y)][ic];
						}
						if (!isMaskPixel(x - mx, y - my + 1)) {
							b[irow] += targetImage.data[targetFlatten(x, y + 1)][ic];
						}
						if (!isMaskPixel(x - mx - 1, y - my)) {
							b[irow] += targetImage.data[targetFlatten(x - 1, y)][ic];
						}

						++irow;
					}
				}
			}

			// 求解该通道上方程组的解
			solutionChannels[ic] = solver.solve(b);
		}

		// 输出结果图片
		{
			std::vector<unsigned char> outImage;


			// 先把目标图全输出
			for (unsigned int i = 0; i < targetImage.data.size(); ++i) {
				vec3 v = targetImage.data[i];
				// 注意，lodepng库中导出成图片时用的是RGB_a格式
				outImage.push_back((unsigned char)(pow(v[0], GAMMA) * 255.0f));
				outImage.push_back((unsigned char)(pow(v[1], GAMMA) * 255.0f));
				outImage.push_back((unsigned char)(pow(v[2], GAMMA) * 255.0f));
				outImage.push_back(255);  // 透明0 - 255，值越大越不透明
			}

			// 再把泊融合的结果替换到目标图上去
			for (unsigned int y = my; y < my + maskImage.height; ++y) {
				for (unsigned int x = mx; x < mx + maskImage.width; ++x) {
					// 像素在Ω区域
					if (isMaskPixel(x - mx, y - my)) {
						unsigned int i = varMap[maskFlatten(x - mx, y - my)];
						vec3 col = vec3((float)solutionChannels[0][i], (float)solutionChannels[1][i], (float)solutionChannels[2][i]);

						// 缩放到0-1范围内
						col[0] = clamp(col[0]);
						col[1] = clamp(col[1]);
						col[2] = clamp(col[2]);

						// outImage是RGBA格式，只要替换RGB3个通道即可，最后一个透明度保持默认255不变
						outImage[4 * targetFlatten(x, y) + 0] = (unsigned char)(pow(col[0], GAMMA) * 255.0f);
						outImage[4 * targetFlatten(x, y) + 1] = (unsigned char)(pow(col[1], GAMMA) * 255.0f);
						outImage[4 * targetFlatten(x, y) + 2] = (unsigned char)(pow(col[2], GAMMA) * 255.0f);
					}
				}
			}
			lodepng::encode(outputFile.c_str(), outImage, targetImage.width, targetImage.height);
			cout << "图片融合成功!请打开相应图片查看结果~\n";
		}
	}
	// 处理输入输出参数
	void setInputParam()
	{
		cout<<"源图像文件名（包括后缀名）：";
		cin>> sourceFile;
		cout << "\nmask图像文件名（包括后缀名）：";
		cin >> maskFile;
		cout << "\n目标图像保存文件名（包括后缀名）：";
		cin >> targetFile;
		cout << "\n输出图像保存文件名（包括后缀名）：";
		cin >> outputFile;

		cout << "\n在目标图x方向融合位置：";
		cin >> mx;
		cout << "\n在目标图y方向融合位置：";
		cin>>my;
		/*sourceFile = "img/bear.png";
		maskFile = "img/bear_mask.png";
		targetFile = "img/bear_bg.png";
		outputFile = "res_bear.png";*/
		/*mx = 350;
		my = 300;*/
	}

private:
	string targetFile, maskFile, sourceFile, outputFile; // 文件名
	unsigned int mx, my;								// 融合位置
	ImageData maskImage;
	ImageData sourceImage;
	ImageData targetImage;


	// 2维坐标转1维，行优先顺序
	int targetFlatten(unsigned int x, unsigned int y) {
		return  targetImage.width * y + x;
	}
	// 2维坐标转1维，行优先顺序
	unsigned int maskFlatten(unsigned int x, unsigned int y) {
		return  maskImage.width * y + x;
	}
	// 检查（x,y）处的像素是否是在mask中，阈值为0.99带有一定的margin，这里假设标注的mask为红色（255,0,0）
	bool isMaskPixel(unsigned int x, unsigned int y) {
		return maskImage.data[maskFlatten(x, y)][0] > 0.99;
	}

};

