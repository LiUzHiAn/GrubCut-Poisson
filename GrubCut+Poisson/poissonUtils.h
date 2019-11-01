#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <map>
#include<string>

#include "lodepng.h"


using namespace std;


// gamma��������.
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

// ������x�޶���0~1��Χ��
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
	std::vector<vec3> data;  // ÿ�����ص�RGB��ɫ��vector��ʾ

	unsigned int width;
	unsigned int height;
};


// ����ͼƬ���ڴ棬������gamma ����
void loadImage(const char* file, ImageData& image) {
	std::vector<unsigned char> buf;

	// ��pngͼƬ�Ӵ��̶����ڴ�
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

// ����ͼ�����������ݶȳ����������еģ�11���ͣ�13��ʽ
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
// ���û������param����ֵ
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
// ������param��������Ϊstring
const char* parseStringParam(const char* param, int argc, char* argv[]) {
	const char* token = findToken(param, argc, argv);
	return token;
}
// ������param��������ΪInt
bool parseIntParam(const char* param, int argc, char* argv[], unsigned int& out) {
	const char* token = findToken(param, argc, argv);
	if (token == nullptr)
		return false;

	int r = sscanf(token, "%u,", &out);   // %u ʮ�����޷�������
	if (r != 1 || r == EOF) {
		return false;
	}
	else {
		return true;
	}
}

void printHelpExit() {

	printf("���������������ʽ����ȷ!\n\n");

	printf("ע��: �벻Ҫ��ͼƬ�ںϵ�λ�øպ�ָ���ڱ߽��ϡ�����mx=0, my=0��\n\n");

	printf("��ѡ����: \n");

	printf("  -target\t\tĿ��ͼƬ\n");
	printf("  -source\t\tԴͼƬ\n");
	printf("  -output\t\t���ͼƬ\n");

	printf("  -mask  \t\t����ͼƬ\n");

	printf("  -mx       \t\t�ں�x��λ��\n");
	printf("  -my       \t\t�ں�y��λ��\n");
	exit(1);
}


