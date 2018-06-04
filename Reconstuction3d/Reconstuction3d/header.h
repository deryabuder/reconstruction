#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\imgproc\imgproc_c.h>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <GL\freeglut.h>
#include <atlstr.h> // use STL string instead, although not as convenient...
#include <atltrace.h>
#define TRACE ATLTRACE

#include <iostream>
#include <fstream>
#include <string>
#include<time.h>
#include<vector>
using namespace std;

void match_features(cv::Mat &imgL, cv::Mat &imgR, vector<vector<cv::KeyPoint>>& key_points_for_all,
	vector<cv::Mat>& descriptor_for_all, vector<vector<cv::Vec3b>>& colors_for_all, vector<cv::DMatch>& matches);
void init_structure(
	cv::Mat &imgL,
	cv::Mat &imgR,
	cv::Mat K,
	vector<vector<cv::KeyPoint>>& key_points_for_all,
	vector<vector<cv::Vec3b>>& colors_for_all,
	vector<cv::DMatch>& matches,
	vector<cv::Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<cv::Vec3b>& colors,
	vector<cv::Mat>& rotations,
	vector<cv::Mat>& motions,
	vector<cv::Point2f>&point1,
	vector<cv::Point2f>&point2
	);
void save_structure(string file_name, vector<cv::Mat>& rotations, vector<cv::Mat>& motions, vector<cv::Point3d>& structure, vector<cv::Vec3b>& colors);

void bundle_adjustment(
	cv::Mat& intrinsic,        //内参矩阵
	vector<cv::Mat>& extrinsics, //外参矩阵集
	vector<vector<int>>& correspond_struct_idx,//correspond_struct_idx[i][j]代表第i幅图像第j个特征点所对应的空间点在点云中的索引
	vector<vector<cv::KeyPoint>>& key_points_for_all, //特征点
	vector<cv::Point3d>& structure  //三维点集
	);


void StereoTo3D(vector<cv::Point2f> ptsL, vector<cv::Point3d> structure,
	cv::Mat img, cv::Point3d &center3D, cv::Vec3d &size3D );


void TriSubDiv( vector<cv::Point2f> &pts, cv::Mat &img, vector<cv::Vec3i> &tri );


void InitGl();
GLuint Create3DTexture(cv::Mat &img, vector<cv::Vec3i> &tri, vector<cv::Point2f> pts2DTex, vector<cv::Point3d> &structure, cv::Point3d center3D, cv::Vec3d size3D);
void Show(GLuint tex, cv::Point3d center3D, cv::Vec3d size3D);

void Init_lightGl();

void displayGl();

void resizeGl(int w, int h);

void mouseGl(int button, int state, int x, int y);

void mouse_move_Gl(int x, int y);

void keyboard_control_Gl(unsigned char key, int a, int b);

void special_control_Gl(int key, int x, int y);

void Display1();