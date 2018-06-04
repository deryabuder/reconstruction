#include <ceres\ceres.h>
#include <ceres\rotation.h>
#include <opencv2\tinydir\tinydir.h>

#include <fstream>
#include "header.h"
#include "legacy.h"

#define MAXM_FILTER_TH	.8	// threshold used in GetPair
#define HOMO_FILTER_TH	60	// threshold used in GetPair
#define NEAR_FILTER_TH	40	// diff points should have distance more than NEAR_FILTER_TH

using namespace cv;

//提取与匹配特征点
void match_features(Mat &imgL, Mat &imgR, vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all, vector<vector<Vec3b>>& colors_for_all,vector<DMatch>& matches)
{   
	key_points_for_all.clear();
	descriptor_for_all.clear();
	vector<KeyPoint> keypointsL, keypointsR;
	Mat descriptorsL, descriptorsR;
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	sift->detectAndCompute(imgL, noArray(), keypointsL, descriptorsL);
	sift->detectAndCompute(imgR, noArray(), keypointsR, descriptorsR);
	key_points_for_all.push_back(keypointsL);
	key_points_for_all.push_back(keypointsR);
	descriptor_for_all.push_back(descriptorsL);
	descriptor_for_all.push_back(descriptorsR);

	vector<Vec3b> colorsL(keypointsL.size());
	vector<Vec3b> colorsR(keypointsR.size());
	for (int i = 0; i < keypointsL.size(); ++i)
	{
		Point2f& p = keypointsL[i].pt;
		colorsL[i] = imgL.at<Vec3b>(p.y, p.x);
	}
	for (int i = 0; i < keypointsR.size(); ++i)
	{
		Point2f& p = keypointsR[i].pt;
		colorsR[i] = imgR.at<Vec3b>(p.y, p.x);
	}
	colors_for_all.push_back(colorsL);
	colors_for_all.push_back(colorsR);

	vector<vector<DMatch>> knn_matches;//DMatch Class for matching keypoint descriptors
	BFMatcher matcher(NORM_L2);//BFMatcher强力descriptor匹配器，
	matcher.knnMatch(descriptor_for_all[0], descriptor_for_all[1], knn_matches, 2);//获取knn_matches[r][0]、knn_matches[r][1]
	
	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		matches.push_back(knn_matches[r][0]);
	}

	Mat OutImg1;
	drawMatches(imgL, key_points_for_all[0], imgR, key_points_for_all[1], matches,
	OutImg1, Scalar(255, 255, 255));
	/*namedWindow("KNN matching", WINDOW_NORMAL);
	imshow("KNN matching",OutImg1);*/
	char title1[100];
	sprintf_s(title1, 100, "KNN matching：%d", matches.size());
	namedWindow(title1, WINDOW_NORMAL);
	imshow(title1, OutImg1);
	
	
	//获取满足Ratio Test的所有匹配点的最小匹配的距离min_dist
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//比率测试
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//排除不满足比率测试的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
	Mat OutImg2;
	drawMatches(imgL, key_points_for_all[0], imgR, key_points_for_all[1], matches,
		OutImg2, Scalar(255, 255, 255));
	//namedWindow("Radio Test matching", WINDOW_NORMAL);
	//imshow("Radio Test matching", OutImg2);
	char title2[100];
	sprintf_s(title2, 100, "Radio Test matching：%d", matches.size());
	namedWindow(title2, WINDOW_NORMAL);
	imshow(title2, OutImg2);
}
//从匹配点钟保存特征点的坐标
void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
	)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}
//从匹配点钟保存特征点的RGB
void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
	)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}
//获取 R  T
bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	//根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	Point2f principle_point(K.at<double>(2), K.at<double>(5));

	//根据匹配点使用RANSAC求取本征矩阵，进一步排除失配点
	/*p1:矩阵1特征点
	p2：矩阵2特征点
	focal_length：焦距
	principle_point：相机主点
	RANSAC
	0.999：RANSAC的参数。 它是从点到对极的最大距离以像素为单位，超出该点被认为是失配点，不用于计算，最终基本矩阵。
	1.0：用于RANSAC或LMedS方法的参数。 它指定了一个理想的水平置信度（概率）估计矩阵是正确的
	mask：输出矩阵，每个元素的数组为失配点设置为0，为1为其他点。（包括图1和图2）
	*/

	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);//它从焦距和主点计算相机本征矩阵（p1p2两矩阵的特征点）
	if (E.empty()) return false;

	double feasible_count = countNonZero(mask);//计算mask里的非零元素数量（过滤后匹配特征点的数目）
	cout << (int)feasible_count << " -in- " << p1.size() << endl;//成功匹配的点和p1特征点
	//对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
		return false;

	//分解本征矩阵，获取相对变换
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//同时位于两个相机前方的点的数量要足够大

	if (((double)pass_count) / feasible_count < 0.7)//pass_count成功匹配的点
		return false;
	return true;
}
//获取RNASAC排除失配点后的p1特征点
void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}
//获取RNASAC排除失配点后的p1特征点的颜色
void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3d>& structure)
{
	//两个相机的投影矩阵[R T]，triangulatePoints只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;//含有焦距的转换矩阵
	K.convertTo(fK, CV_32FC1);//K转换为CV_32FC1类型的矩阵fK
	proj1 = fK*proj1;
	proj2 = fK*proj2;

	//三角重建
	/*
	函数功能是通过使用立体摄像机观察重建三维点(在齐次坐标)
	o projMatr1– 3x4 第一个相机的投影矩阵.3×4
	o projMatr2– 3x4 第二个相机的投影矩阵.3×4
	o projPoints1– 2xN 第一幅图像的特征点矩阵.
	o projPoints2– 2xN第二幅图像的特征点矩阵.
	o points4D– 4xN 在齐次坐标系之中重构的向量,引入齐次坐标的目的主要是合并矩阵运算中的乘法和加法(三维点变换齐次坐标前)
	*/

	Mat s;//structure
	triangulatePoints(proj1, proj2, p1, p2, s);

	structure.clear();
	structure.reserve(s.cols);//设置structure容器大小
	for (int i = 0; i < s.cols; ++i)//4×N，每一列是一个坐标
	{
		Mat_<float> col = s.col(i);
		col /= col(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
		structure.push_back(Point3f(col(0), col(1), col(2)));//将三维重建点放入structure中
	}
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3d>& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i];//输出三维点
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}
//首先是初始化点云，也就是通过双目重建方法对图像序列的头两幅图像进行重建，并初始化correspond_struct_idx。
void init_structure(
	Mat &imgL,
	Mat &imgR,
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<DMatch>& matches,
	vector<Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions,
	vector<Point2f>&point1,
	vector<Point2f>&point2
	)
{
	//计算头两幅图像之间的变换矩阵
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	//旋转矩阵和平移向量
	Mat mask;	//mask中大于零的点代表匹配点，等于零代表失配点
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches, p1, p2);//从匹配点钟保存特征点的坐标
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches, colors, c2);//从匹配点钟保存特征点的颜色
	find_transform(K, p1, p2, R, T, mask);//获取R T

	Mat OutImg3;
	drawMatches(imgL, key_points_for_all[0], imgR, key_points_for_all[1], matches,
		OutImg3, Scalar(255, 255, 255));
	/*namedWindow("RANSAC matching", WINDOW_NORMAL);
	imshow("RANSAC matching", OutImg3);*/
	char title[100];
	sprintf_s(title, 100, "RANSAC matching：%d", matches.size());
	namedWindow(title, WINDOW_NORMAL);
	imshow(title, OutImg3);


	//对头两幅图像进行三维重建
	maskout_points(p1, mask);//获取RNASAC排除失配点后的p1特征点
	maskout_points(p2, mask);//获取RNASAC排除失配点后的p2特征点
	maskout_colors(colors, mask);//获取RNASAC排除失配点后的p1特征点的颜色
	Mat R0 = Mat::eye(3, 3, CV_64FC1);//设置第一幅图像的R0T0
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);

	reconstruct(K, R0, T0, R, T, p1, p2, structure);//进行三维重建
	point1 = p1;
	point2 = p2;

	//保存变换矩阵
	rotations = { R0, R };
	motions = { T0, T };

	//将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());//图片数相同
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);//每幅图像的特征点设置为相同的
	}

	//填写头两幅图像的结构索引
	int idx = 0;
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
			continue;
		//idx成为两个匹配的特征点的索引
		correspond_struct_idx[0][matches[i].queryIdx] = idx;//idx成为第0幅图像第i个匹配点的索引
		correspond_struct_idx[1][matches[i].trainIdx] = idx;//idx成为第1幅图像第i个匹配点的索引
		++idx;
	}
}
//代价（coss）函数
struct ReprojectCost
{
	cv::Point2d observation;//特征点的坐标

	ReprojectCost(cv::Point2d& observation)
		: observation(observation)
	{
	}

	template <typename T>
	bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const//residuals残差是指实际观察值与估计值（拟合值）之间的差
	{
		const T* r = extrinsic;
		const T* t = &extrinsic[3];

		T pos_proj[3];
		ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);//对世界坐标系的三维点进行旋转变换

		//平移变换，转换为摄像机坐标系
		pos_proj[0] += t[0];
		pos_proj[1] += t[1];
		pos_proj[2] += t[2];

		const T x = pos_proj[0] / pos_proj[2];
		const T y = pos_proj[1] / pos_proj[2];

		const T fx = intrinsic[0];//intrinsic是 4×1的内参矩阵
		const T fy = intrinsic[1];
		const T cx = intrinsic[2];
		const T cy = intrinsic[3];


		const T u = fx * x + cx;//u,v是图片坐标系像素的行和列
		const T v = fy * y + cy;

		residuals[0] = u - T(observation.x);//x方向反向投影误差（反向投影值-实际观察值）
		residuals[1] = v - T(observation.y);//y方向反向投影误差

		return true;
	}
};
//Ceres Solver求解BA，其中使用了Ceres提供的Huber函数作为损失（loss）函数
void bundle_adjustment(
	Mat& intrinsic,        //内参矩阵
	vector<Mat>& extrinsics, //外参矩阵集
	vector<vector<int>>& correspond_struct_idx,//correspond_struct_idx[i][j]代表第i幅图像第j个特征点所对应的空间点在点云中的索引
	vector<vector<KeyPoint>>& key_points_for_all, //特征点
	vector<Point3d>& structure  //三维点集
	)
{
	ceres::Problem problem;

	// 加载外参（R T）
	for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}
	// 在优化期间保持指定的参数块（第一个相机的内参）不变
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// 加载内参
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

	// 加载特征点
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)//img_idx表示第几张图片
	{
		vector<int>& point3d_ids = correspond_struct_idx[img_idx];//第img_idx张图片的三维点索引集
		vector<KeyPoint>& key_points = key_points_for_all[img_idx];//第img_idx张图片的特征点集
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)//point_idx表示第几个3维点和特征点
		{
			int point3d_id = point3d_ids[point_idx];//指定三维点的标签
			if (point3d_id < 0)
				continue;

			Point2d observed = key_points[point_idx].pt;//特征点的坐标
			// 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

			//加载残差有关的参数
			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),            // Intrinsic
				extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
				&(structure[point3d_id].x)          // Point in 3D space
				);
		}
	}

	// Solve BA
	ceres::Solver::Options ceres_config_options;//选项结构包含控制如何的选项
	ceres_config_options.minimizer_progress_to_stdout = false;//默认情况下，Minimizer进度记录到VLOG
	ceres_config_options.logging_type = ceres::SILENT;//logging类型
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::JACOBI;//与迭代线性求解器一起使用的预处理器类型
	ceres_config_options.linear_solver_type = ceres::ITERATIVE_SCHUR;//最小二乘法选项
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;//稀疏线性代数库类型

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);//求解稀疏矩阵的线性方程

	if (!summary.IsSolutionUsable())//summary是否可用
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
	}
	else
	{
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"//图片个数
			<< " #residuals: " << summary.num_residuals << "\n"//残差的总数
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"//优化前的平均反向投影误差（Initial RMSE）
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"//优化后的该值（Final RMSE）
			<< " Time (s): " << summary.total_time_in_seconds << "\n"//当Solve被调用时，总共花费在Ceres中的所有时间
			<< std::endl;
	}
}
//使用opencv的功能进行三角剖分
bool isGoodTri( Vec3i &v, vector<Vec3i> & tri ) //Vec3i中存储的是3个表示顶点编号的整数,tri存储所有的三角形
{
	int a = v[0], b = v[1], c = v[2];
	v[0] = min(a,min(b,c));//三个顶点的最小值
	v[2] = max(a,max(b,c));//三个顶点的最大值
	v[1] = a+b+c-v[0]-v[2];
	if (v[0] == -1) return false;
	//遍历tri的所有三角形
	vector<Vec3i>::iterator iter = tri.begin();//迭代器(Iterator) 迭代器是一种设计模式,它是一个对象,它可以遍历并选择序列中的对象
	for(;iter!=tri.end();iter++)
	{
		Vec3i &check = *iter;
		if (check[0]==v[0] &&//且
			check[1]==v[1] &&
			check[2]==v[2])
		{
			break;
		}
	}
	if (iter == tri.end())
	{
		tri.push_back(v);
		return true;
	}
	return false;
}
//逐点插入方式生成的Delaunay三角网的算法主要基于Bowyer-Watson算法

void TriSubDiv( vector<Point2f> &pts, Mat &img, vector<Vec3i> &tri )//左图特征点-左图-tri 
{
	CvSubdiv2D* subdiv;//The subdivision itself // 细分 
	CvMemStorage* storage = cvCreateMemStorage(0);//用来存储三角剖分 
	Rect rc = Rect(0,0, img.cols, img.rows); //我们的外接边界盒子 

	subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv),//创建新的细分
		sizeof(CvSubdiv2DPoint),
		sizeof(CvQuadEdge2D),
		storage );//为数据申请空间  

	cvInitSubdivDelaunay2D( subdiv, rc );//初始化Delaunay三角剖分 

	//如果我们的点集不是32位的，在这里我们将其转为CvPoint2D32f，如下两种方法。
	for (size_t i = 0; i < pts.size(); i++)
	{
		CvSubdiv2DPoint *pt = cvSubdivDelaunay2DInsert( subdiv, pts[i] );//插入一个新的点到Delaunay三角剖分
		pt->id = i;
	}

	CvSeqReader reader;
	int total = subdiv->edges->total;//元素总数
	int elem_size = subdiv->edges->elem_size;//序列元素的大小（以字节计）
	//初始化序列阅读器。序列可以向前或向后读取
	cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );
	Point buf[3];//三个点的数组
	const Point *pBuf = buf;//指向三个顶点的指针
	Vec3i verticesIdx;//顶点索引
	Mat imgShow = img.clone();

	srand( (unsigned)time( NULL ) );   
	for( int i = 0; i < total; i++ )//循环所有的边
	{   
		CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);   
		//检查ptr指向的元素是否属于一个集合
		if( CV_IS_SET_ELEM( edge ))//循环一个三角形的三个顶点
		{
			CvSubdiv2DEdge t = (CvSubdiv2DEdge)edge; 
			int iPointNum = 3;
			Scalar color = CV_RGB(rand()&255,rand()&255,rand()&255);//随机取色
			int j;
			for(j = 0; j < iPointNum; j++ )//循环三个顶点
			{
				CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );
				if( !pt ) break;
				buf[j] = pt->pt;
				verticesIdx[j] = pt->id;//顶点索引
				t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );//返回左区域的下一条边
			}
			if (j != iPointNum) continue;
			if (isGoodTri(verticesIdx, tri))//是否进行三角剖分
			{   //绘制三角形
				polylines( imgShow, &pBuf, &iPointNum, 
					1, true, color,
					1, CV_AA, 0);
			}
			//添加两条边
			t = (CvSubdiv2DEdge)edge+2;

			for(j = 0; j < iPointNum; j++ )
			{
				CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );
				if( !pt ) break;
				buf[j] = pt->pt;
				verticesIdx[j] = pt->id;
				t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );
			}   
			if (j != iPointNum) continue;
			if (isGoodTri(verticesIdx, tri))
			{
				//绘制三角形
				polylines( imgShow, &pBuf, &iPointNum, 
					1, true, color,
					1, CV_AA, 0);
			}
		}

		CV_NEXT_SEQ_ELEM( elem_size, reader );

	}
    
	char title[100];
	sprintf_s(title, 100, "Delaunay: %d Triangles", tri.size());
	namedWindow(title, WINDOW_NORMAL);
	imshow(title, imgShow);
	waitKey();
}

void StereoTo3D(vector<Point2f> ptsL, vector<Point3d> structure, Mat img,//左图
				Point3d &center3D, Vec3d &size3D) //输出变量，中心坐标和大小
{
	double minX = 1e9, maxX = -1e9;//最小值不会超过1e9,最大值不会小于-1e9，即点约束在-1e9到1e9之间
	double minY = 1e9, maxY = -1e9;//10的九次方，-10的九次方
    double minZ = 1e9, maxZ = -1e9;

	for (int i = 0; i < structure.size(); ++i)
	{

		minX = min(minX, structure[i].x); maxX = max(maxX, structure[i].x);//取最大值和最小值
		minY = min(minY, structure[i].y); maxY = max(maxY, structure[i].y);
		minZ = min(minZ, structure[i].z); maxZ = max(maxZ, structure[i].z);
	}
	//绘制的中心点的坐标
	center3D.x = (minX+maxX)/2;//点云中心
	center3D.y = (minY+maxY)/2;
	center3D.z = (minZ+maxZ)/2;
	//绘制圆的大小
	size3D[0] = maxX-minX;//点云块大小
	size3D[1] = maxY-minY;
	size3D[2] = maxZ-minZ;
}
