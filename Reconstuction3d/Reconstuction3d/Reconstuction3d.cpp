#include "stdlib.h"  
#include <direct.h>  
#include <string.h>  
#include "header.h"

using namespace cv;


int main(int argc, char* argv[])
{
	//fountain
	/*Mat imgL = imread("E:\\Reconstuction3d\\Reconstuction3d\\fountain1.jpg"); 
	Mat	imgR = imread("fountain2.jpg");*/
	//church
	Mat imgL = imread("E:\\Reconstuction3d\\Reconstuction3d\\church1.jpg");
	Mat	imgR = imread("church2.jpg");
	//hand
	/*Mat imgL = imread("E:\\Reconstuction3d\\Reconstuction3d\\hand1.png");
	Mat	imgR = imread("hand2.png");*/
	//medusa
	/*Mat imgL = imread("E:\\Reconstuction3d\\Reconstuction3d\\medusa1.jpg");
	Mat	imgR = imread("medusa2.jpg");*/
	
	if (!(imgL.data) || !(imgR.data))//或
	{

		cerr<<"can't load image!"<<endl;
		exit(1);
	}

	/************************************************************************/
	/*决定左图像中哪些点应该被选择，并在右图像中计算其对应点 */
	/************************************************************************/
	cout<<"calculating feature points..."<<endl;
	//fountain相机内参
	/*Mat K(Matx33d(
		645.070482,0.000000,353.968508,
        0.000000,639.683979,234.946850,
        0.000000,0.000000,1.000000));*/
	//church相机内参
	Mat K(Matx33d(
		1663.782234,0.000000,785.889057,
        0.000000,1663.367425,638.790025,
        0.000000,0.000000,1.000000));
	//hand
	/*Mat K(Matx33d(
		667.168218,0.000000,321.060607,

		0.000000,670.287385,233.834066,

		0.000000,0.000000,1.000000));*/
	//medusa
	/*Mat K(Matx33d(
		1005.279068,0.000000,357.823593,

		0.000000,1102.528021,264.015057,

		0.000000,0.000000,1.000000));*/

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<DMatch> matches;
	//提取和匹配特征点
	match_features(imgL, imgR, key_points_for_all, descriptor_for_all, colors_for_all, matches);
	
	vector<Point3d> structure;
	vector<vector<int>> correspond_struct_idx; //保存第i副图像中第j个特征点对应的structure中点的索引
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;
	vector<Point2f>point1;
	vector<Point2f>point2;

	//初始化结构（三维点云）
	init_structure(
		imgL,
		imgR,
		K,
		key_points_for_all,
		colors_for_all,
		matches,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions,point1,point2
		);



	save_structure("E:\\Reconstuction3d\\Viewer1\\structure.yml", rotations, motions, structure, colors);
	//对BA进行调用
	Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));//相机内参定义一个4×1矩阵
	vector<Mat> extrinsics;//外参矩阵集
	for (size_t i = 0; i < rotations.size(); ++i)//旋转向量的个数
	{
		Mat extrinsic(6, 1, CV_64FC1);//extrinsic设置为6个参数的矩阵
		Mat r;
		Rodrigues(rotations[i], r);//将旋转向量转换为旋转矩阵

		r.copyTo(extrinsic.rowRange(0, 3));//将r的前3行复制到extrinsic
		motions[i].copyTo(extrinsic.rowRange(3, 6));//将motions[i]复制到extrinsic的后三行

		extrinsics.push_back(extrinsic);
	}
	bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);
    for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		Mat extrinsic(6, 1, CV_64FC1);//extrinsic设置为6个参数的矩阵
		Mat r;
		Rodrigues(rotations[i], r);//将旋转向量转换为旋转矩阵
		extrinsics[i].rowRange(0,3).copyTo(r);
		extrinsics[i].rowRange(3, 6).copyTo(motions[i]);
		Rodrigues(r, rotations[i]);//将旋转矩阵转换为旋转向量
	}
	save_structure("E:\\Reconstuction3d\\Viewer2\\structure.yml", rotations, motions, structure, colors);
	Point3d center3D;
	Vec3d size3D;
	StereoTo3D(point1,structure,imgL, center3D, size3D);

	/************************************************************************/
	/* 三角剖分                                               */
	/************************************************************************/
	cout<<"doing triangulation..."<<endl;
	vector<Vec3i> tri;//三角
	TriSubDiv(point1, imgL, tri);

	/************************************************************************/
	/*纹理贴图                                          */
	/************************************************************************/
	glutInit(&argc, argv); // 调用glut函数前,要初始化glut
	InitGl(); // 调用glut函数前,要初始化glut

	cout<<"creating 3D texture..."<<endl;
	GLuint tex = Create3DTexture(imgL, tri, point1, structure, center3D, size3D);
	Show(tex,center3D, size3D);
	
	return 0;
}

