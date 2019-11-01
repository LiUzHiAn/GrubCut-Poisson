#pragma once

#include "poissonUtils.h"
#include <Eigen/Sparse>



class poissonEditer
{

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::VectorXd Vec;
public:
	// ִ�в����ںϲ���
	void poisson_blending() {							
		// ����3��ͼƬ
		loadImage(targetFile.c_str(), targetImage);
		loadImage(maskFile.c_str(), maskImage);
		loadImage(sourceFile.c_str(), sourceImage);

		// ��mx��my���м�飨��ֹ�����ڱ߽��ϵ������  630 600 520 291
		{
			unsigned int xmin = mx;
			unsigned int ymin = my;

			unsigned int xmax = mx + maskImage.width;
			unsigned int ymax = my + maskImage.height;

			if (xmin > 0 && ymin > 0 && xmax < targetImage.width - 1 && ymax < targetImage.height - 1) {
				// ���ͨ��
			}
			else
			{ 
				printf("ָ���ںϵ�λ��(%d,%d)�޷������´��ںϵ�ͼƬ\n", mx, my);
				return;
			}
		}
		cout << "�����ں�...\n";
		// key������������maskͼƬ�е�������������˳�򣩣�value�������ڦ����������
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
		const unsigned int numUnknowns = (unsigned int)varMap.size();  //  ����������ظ���

		 // ���체�����Է����� Mx=b ��ϵ������M��ϡ�������Eigen�е�Triplet�����죩
		std::vector<Triplet> mt;
		{
			unsigned int irow = 0;
			for (unsigned int y = my; y < my + maskImage.height; ++y) {
				for (unsigned int x = mx; x < mx + maskImage.width; ++x) {
					// ����ĵ��ڦ�������
					if (isMaskPixel(x - mx, y - my)) {
						/*
						��ϸ��Ⲵ�ɷ��̹�ʽ�����ģ�7��ʽ����Ϊ���ﲻ�����ں���Ŀ��ͼƬ�ı�Ե������|N_p| ����� 4
						*/
						mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - my)], 4)); // |N_p| = 4.

						// �ж�����4�������Ƿ��ڦ������ڵĻ�������ӵ�������ߣ�����Ļ�����Ȼ���ڷ�������ұߣ�����֪���߽�b��
						// �Ϸ����ص�
						if (isMaskPixel(x - mx, y - my - 1)) {
							mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - 1 - my)], -1));
						}
						// �ҷ����ص�
						if (isMaskPixel(x - mx + 1, y - my)) {
							mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx + 1, y - my)], -1));
						}
						// �·����ص�
						if (isMaskPixel(x - mx, y - my + 1)) {
							mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx, y - my + 1)], -1));
						}
						// �����ص�
						if (isMaskPixel(x - mx - 1, y - my)) {
							mt.push_back(Triplet(irow, varMap[maskFlatten(x - mx - 1, y - my)], -1));
						}

						++irow; // ��һ�����Է���
					}
				}
			}
		}

		// �������Է���������Կ�֪��ϵ������M��һ���Գ��������󣬿�����Cholesky�ֽ�������X
		// ��Ϊ��RGB��ɫ�ռ䣬���õ�M�������3�����Է�����
		Eigen::SimplicialCholesky<SpMat> solver;
		{
			// ϡ����󣬷��̸���Ϊ����������ظ���
			SpMat mat(numUnknowns, numUnknowns);
			mat.setFromTriplets(mt.begin(), mt.end());
			// Cholesky�ֽ�
			solver.compute(mat);
		}

		Vec solutionChannels[3];
		Vec b(numUnknowns);
		// ���ÿ��ͨ������ɫ����Ҫ��һ�����Է����飬b��ÿ�����ʱ��仯
		for (unsigned int ic = 0; ic < 3; ++ic)
		{
			unsigned int irow = 0;

			for (unsigned int y = my; y < my + maskImage.height; ++y) {
				for (unsigned int x = mx; x < mx + maskImage.width; ++x) {
					// ����ĵ��ڦ�������
					if (isMaskPixel(x - mx, y - my)) {
						// we only ended up using v in the end.
						// Դͼ��ǰ���ص���ɫ
						vec3 v = sourceImage.data[maskFlatten(x - mx, y - my)];
						// Ŀ��ͼ��ǰ���ص���ɫ
						vec3 u = targetImage.data[targetFlatten(x, y)];

						/* ���Ĺ�ʽ(7)���ұ��� */

						// ÿ�����غ����������ص��ݶ�v_pq֮�ͣ���������ݶ������ַ�����
						// 1. ֱ����ԴͼƬg���������ص��ݶ�
						// 2. ԭͼƬg��Ŀ��ͼƬf*���������ص��ݶȣ�ȡ�ϴ��Ǹ�  ���Ĺ�ʽ(13)
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
						�߽������ڹ�ʽ��7���ұ����е�f*_q
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

			// ����ͨ���Ϸ�����Ľ�
			solutionChannels[ic] = solver.solve(b);
		}

		// ������ͼƬ
		{
			std::vector<unsigned char> outImage;


			// �Ȱ�Ŀ��ͼȫ���
			for (unsigned int i = 0; i < targetImage.data.size(); ++i) {
				vec3 v = targetImage.data[i];
				// ע�⣬lodepng���е�����ͼƬʱ�õ���RGB_a��ʽ
				outImage.push_back((unsigned char)(pow(v[0], GAMMA) * 255.0f));
				outImage.push_back((unsigned char)(pow(v[1], GAMMA) * 255.0f));
				outImage.push_back((unsigned char)(pow(v[2], GAMMA) * 255.0f));
				outImage.push_back(255);  // ͸��0 - 255��ֵԽ��Խ��͸��
			}

			// �ٰѲ��ںϵĽ���滻��Ŀ��ͼ��ȥ
			for (unsigned int y = my; y < my + maskImage.height; ++y) {
				for (unsigned int x = mx; x < mx + maskImage.width; ++x) {
					// �����ڦ�����
					if (isMaskPixel(x - mx, y - my)) {
						unsigned int i = varMap[maskFlatten(x - mx, y - my)];
						vec3 col = vec3((float)solutionChannels[0][i], (float)solutionChannels[1][i], (float)solutionChannels[2][i]);

						// ���ŵ�0-1��Χ��
						col[0] = clamp(col[0]);
						col[1] = clamp(col[1]);
						col[2] = clamp(col[2]);

						// outImage��RGBA��ʽ��ֻҪ�滻RGB3��ͨ�����ɣ����һ��͸���ȱ���Ĭ��255����
						outImage[4 * targetFlatten(x, y) + 0] = (unsigned char)(pow(col[0], GAMMA) * 255.0f);
						outImage[4 * targetFlatten(x, y) + 1] = (unsigned char)(pow(col[1], GAMMA) * 255.0f);
						outImage[4 * targetFlatten(x, y) + 2] = (unsigned char)(pow(col[2], GAMMA) * 255.0f);
					}
				}
			}
			lodepng::encode(outputFile.c_str(), outImage, targetImage.width, targetImage.height);
			cout << "ͼƬ�ںϳɹ�!�����ӦͼƬ�鿴���~\n";
		}
	}
	// ���������������
	void setInputParam()
	{
		cout<<"Դͼ���ļ�����������׺������";
		cin>> sourceFile;
		cout << "\nmaskͼ���ļ�����������׺������";
		cin >> maskFile;
		cout << "\nĿ��ͼ�񱣴��ļ�����������׺������";
		cin >> targetFile;
		cout << "\n���ͼ�񱣴��ļ�����������׺������";
		cin >> outputFile;

		cout << "\n��Ŀ��ͼx�����ں�λ�ã�";
		cin >> mx;
		cout << "\n��Ŀ��ͼy�����ں�λ�ã�";
		cin>>my;
		/*sourceFile = "img/bear.png";
		maskFile = "img/bear_mask.png";
		targetFile = "img/bear_bg.png";
		outputFile = "res_bear.png";*/
		/*mx = 350;
		my = 300;*/
	}

private:
	string targetFile, maskFile, sourceFile, outputFile; // �ļ���
	unsigned int mx, my;								// �ں�λ��
	ImageData maskImage;
	ImageData sourceImage;
	ImageData targetImage;


	// 2ά����ת1ά��������˳��
	int targetFlatten(unsigned int x, unsigned int y) {
		return  targetImage.width * y + x;
	}
	// 2ά����ת1ά��������˳��
	unsigned int maskFlatten(unsigned int x, unsigned int y) {
		return  maskImage.width * y + x;
	}
	// ��飨x,y�����������Ƿ�����mask�У���ֵΪ0.99����һ����margin����������ע��maskΪ��ɫ��255,0,0��
	bool isMaskPixel(unsigned int x, unsigned int y) {
		return maskImage.data[maskFlatten(x, y)][0] > 0.99;
	}

};

