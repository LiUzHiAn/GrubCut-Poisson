#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <map>
#include<string>

#include "lodepng.h"


using namespace std;


// gamma矫正参数.
constexpr float GAMMA = 2.2f;

class vec3 {
private:
	float x, y, z;
public:
	vec3(float x, float y, float z) { this->x = x; this->y = y; this->z = z; }
	vec3(float v) { this->x = v; this->y = v; this->z = v; }
	vec3() { this->x = this->y = this->z = 0; }
	vec3& operator+=(const vec3& b) { (*this) = (*this) + b; return (*this); }
	friend vec3 operator-(const vec3& a, const vec3& b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
	friend vec3 operator+(const vec3& a, const vec3& b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
	friend vec3 operator*(const float s, const vec3& a) { return vec3(s * a.x, s * a.y, s * a.z); }
	friend vec3 operator*(const vec3& a, const float s) { return s * a; }
	const float& operator[] (int index)const{return ((float*)(this))[index];}
	float& operator[] (int index){return ((float*)(this))[index];}
};

// 把输入x限定在0~1范围内
float clamp(float x) {
	if (x > 1.0f) {
		return 1.0f;
	}
	else if (x < 0.0f) {
		return 0.0f;
	}
	else {
		return x;
	}
}

struct ImageData {
	std::vector<vec3> data;  // 每个像素的RGB颜色，vector表示

	unsigned int width;
	unsigned int height;
};


// 加载图片进内存，并进行gamma 矫正
void loadImage(const char* file, ImageData& image) {
	std::vector<unsigned char> buf;

	// 将png图片从磁盘读入内存
	unsigned error = lodepng::decode(buf, image.width, image.height, file);
	if (error) {
		printf("could not open input image %s: %s\n", file, lodepng_error_text(error));
		exit(1);
	}

	for (unsigned int i = 0; i < buf.size(); i += 4) {
		vec3 v = vec3(
			pow(buf[i + 0] / 255.0f, 1.0f / GAMMA),
			pow(buf[i + 1] / 255.0f, 1.0f / GAMMA),
			pow(buf[i + 2] / 255.0f, 1.0f / GAMMA)
		); // RGB

		image.data.push_back(v);
	}
}

// 计算图像引导场（梯度场），论文中的（11）和（13）式
float vpq(float fpstar, float fqstar,float gp, float gq) {
	float fdiff = fpstar - fqstar;
	float gdiff = gp - gq;

	// equation (11) in the paper.
	 return gdiff;

	
	// we can also mix gradients using equation (13) in the paper, as shown below.
	// but I didn't find the results that compelling, so I didn't
	// implement it in the final program
	
	/*if (fabs(fdiff) > fabs(gdiff)) {
		return fdiff;
	}
	else {
		return gdiff;
	}*/
	
}
// 找用户输入的param参数值
const char* findToken(const char* param, int argc, char* argv[]) {
	const char* token = nullptr;
	for (int i = 0; i < argc; ++i) {
		if (strcmp(argv[i], param) == 0) {
			if (i + 1 < argc) {
				token = argv[i + 1];
				break;
			}
		}
	}

	if (token == nullptr) {
		printf("Could not find command-line parameter %s\n", param);
		return nullptr;
	}

	return token;
}
// 将输入param参数解析为string
const char* parseStringParam(const char* param, int argc, char* argv[]) {
	const char* token = findToken(param, argc, argv);
	return token;
}
// 将输入param参数解析为Int
bool parseIntParam(const char* param, int argc, char* argv[], unsigned int& out) {
	const char* token = findToken(param, argc, argv);
	if (token == nullptr)
		return false;

	int r = sscanf(token, "%u,", &out);   // %u 十进制无符号整数
	if (r != 1 || r == EOF) {
		return false;
	}
	else {
		return true;
	}
}

void printHelpExit() {

	printf("命令行输入参数格式不正确!\n\n");

	printf("注意: 请不要把图片融合的位置刚好指定在边界上。（如mx=0, my=0）\n\n");

	printf("可选参数: \n");

	printf("  -target\t\t目标图片\n");
	printf("  -source\t\t源图片\n");
	printf("  -output\t\t输出图片\n");

	printf("  -mask  \t\t遮罩图片\n");

	printf("  -mx       \t\t融合x轴位置\n");
	printf("  -my       \t\t融合y轴位置\n");
	exit(1);
}


