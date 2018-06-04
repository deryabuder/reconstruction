#include "header.h"
using namespace cv;


#if !defined(GLUT_WHEEL_UP)
#  define GLUT_WHEEL_UP   3
#  define GLUT_WHEEL_DOWN 4
#endif


//绘制三角纹理
void MapTexTri(Mat & texImg, Point2f pt2D[3], Point3d pt3D[3])
{
	// 纹理过滤函数
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);//将图片的纹理像素映射到屏幕像素
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, texImg.cols, texImg.rows, 0,//生成2D纹理
		GL_RGB, GL_UNSIGNED_BYTE, texImg.data);


	glBegin(GL_TRIANGLES);// 开始绘制三角纹理

	glTexCoord2f(pt2D[0].x, pt2D[0].y); glVertex3f(pt3D[0].x, -pt3D[0].y, -pt3D[0].z);//纹理坐标
	glTexCoord2f(pt2D[1].x, pt2D[1].y); glVertex3f(pt3D[1].x, -pt3D[1].y, -pt3D[1].z);
	glTexCoord2f(pt2D[2].x, pt2D[2].y); glVertex3f(pt3D[2].x, -pt3D[2].y, -pt3D[2].z);

	glEnd();

}

// 创建三维纹理

GLuint Create3DTexture(Mat &img, vector<Vec3i> &tri,
	vector<Point2f> pts2DTex, vector<Point3d> &structure, Point3d center3D, Vec3d size3D)
{
	GLuint tex = glGenLists(1);// 显示列表初始化
	int error = glGetError();
	if (error != GL_NO_ERROR)
		cout << "An OpenGL error has occured: " << gluErrorString(error) << endl;
	if (tex == 0) return 0;

	Mat texImg;
	cvtColor(img, img, CV_BGR2RGB);  //将左图转换为RGB图像
	resize(img, texImg, Size(400, 300)); // 设置左图的大小

	glNewList(tex, GL_COMPILE);//创建一个显示列表

	vector<Vec3i>::iterator iterTri = tri.begin();// 迭代器
	Point2f pt2D[3];
	Point3d pt3D[3];

	glDisable(GL_BLEND); // 关闭混合
	glEnable(GL_TEXTURE_2D); // 启用纹理映射
	for (; iterTri != tri.end(); iterTri++)//循环tri
	{
		Vec3i &vertices = *iterTri;
		int ptIdx;
		for (int i = 0; i < 3; i++) // 循环三角形三个点
		{
			ptIdx = vertices[i];
			if (ptIdx == -1) break;
			pt2D[i].x = pts2DTex[ptIdx].x / img.cols;
			pt2D[i].y = pts2DTex[ptIdx].y / img.rows;
			pt3D[i] = (structure[ptIdx] - center3D) * (1.f / max(size3D[0], size3D[1]));//1.f 是c++里面表示浮点数的1
		}

		if (ptIdx != -1)
		{
			MapTexTri(texImg, pt2D, pt3D);
		}
	}
	glDisable(GL_TEXTURE_2D);// 关闭纹理映射

	glEndList();//替换显示列表
	return tex;

}

#define PI_180			(CV_PI/180)
#define ROTATE_STEP		5
#define TRANSLATE_STEP	.3

static float	g_rx, g_ry;
static float	g_tz;
static GLuint	g_tex;

// 对显示窗口进行初始化
void InitGl()
{
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(600, 600); // 初始化窗口大小
	glutInitWindowPosition(100, 100); // 初始化窗口位置

	glutCreateWindow("3D reconstruction");

	glClearColor(0, 0, 0, 1);
	glutDisplayFunc(displayGl); // 展示
	glutReshapeFunc(resizeGl); // 窗口设置
	glutKeyboardFunc(keyboard_control_Gl); // 重绘函数
	glutSpecialFunc(special_control_Gl); // key键控制旋转
	glutMouseFunc(mouseGl); // 重绘函数
	glutMotionFunc(mouse_move_Gl); // 鼠标滚动控制放大缩小

	wglUseFontBitmaps(wglGetCurrentDC(), 0, 256, 1000);
	glEnable(GL_DEPTH_TEST); // 开启更新深度缓冲区的功能
}

void Show(GLuint tex, Point3d center3D, Vec3d size3D)
{
	g_tz = 2; // 调整相机位置
	g_rx = 90;
	g_ry = 0;
	g_tex = tex;
	glutMainLoop(); // 进入GLUT事件处理循环
}

void displayGl()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);// 设置当前矩阵为模型矩阵
	glPushMatrix(); // 将当前矩阵保存入堆栈顶
	// 相机在世界坐标系中的坐标
	//glRotatef(180.0, 0.0, 1.0, 0.0);
	float eyey = g_tz*sin(g_ry*PI_180),
		eyex = g_tz*cos(g_ry*PI_180)*cos(g_rx*PI_180),
		eyez = g_tz*cos(g_ry*PI_180)*sin(g_rx*PI_180);
	gluLookAt(eyex, eyey, eyez, 0, 0, 0, 0, 1, 0);// 设置视景体在世界坐标的位置和方向来观察物体（相机在世界坐标的位置-相机镜头对准的物体在世界坐标的位置-相机向上的方向在世界坐标中的方向）
	TRACE("%.1f,%.1f,%.1f,%.1f,%.1f\n", g_rx, g_ry, eyex, eyey, eyez);

	//glColor3f(1, 1, 1);
	glCallList(g_tex);// 执行显示列表

	glPopMatrix();// 当前矩阵出栈
	glPushMatrix();
	//glColor3f(0, 1, 0);
	//glTranslatef(-0.08, 0.08, -0.2);//平移函数
	
	glListBase(1000);// 使得OpengL可以找到绘制对应字符的显示列表的位置
	glRasterPos3f(0, 0, 0);//图像平移
	string help = "use arrow keys to rotate, mouse wheel to zoom";
	glCallLists(help.size(), GL_UNSIGNED_BYTE, help.c_str());

	glPopMatrix(); // 当前矩阵出栈
	glFlush(); // 强制刷新缓冲，保证绘图命令将被执行,而不是存储在缓冲区中等待其他的OpenGL命令
	glutSwapBuffers(); // 实行双缓冲技术来进行绘图
}

void resizeGl(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h); // 图形绘制窗口区域的位置和大小
	glMatrixMode(GL_PROJECTION); // 设置当前矩阵为投影矩阵
	glLoadIdentity(); // 设置当前矩阵为单位矩阵
	gluPerspective(45, GLdouble(w) / GLdouble(h), 0.01, 10000.0); // 指定观察的视景体在世界坐标中的大小（眼睛睁开的角度-实际窗口横纵比-近处的裁面-远处的裁面）
	glMatrixMode(GL_MODELVIEW); //  设置当前矩阵为模型矩阵
	glLoadIdentity(); // 设置当前矩阵为单位矩阵
}

void mouseGl(int button, int state, int x, int y)
{
	switch (button)
	{
	case GLUT_WHEEL_UP: // 放大
		g_tz -= TRANSLATE_STEP;
		break;

	case  GLUT_WHEEL_DOWN: // 缩小
		g_tz += TRANSLATE_STEP;
		break;

	default:
		break;
	}
	if (g_tz < 0) g_tz = 0;
	glutPostRedisplay();
}

void mouse_move_Gl(int x, int y)
{

	glutPostRedisplay();
}

void keyboard_control_Gl(unsigned char key, int a, int b)
{
	if (key == 0x1B)
		exit(1);
	glutPostRedisplay();//表示当前窗口重新绘制
}

void special_control_Gl(int key, int x, int y)
{
	if (key == GLUT_KEY_LEFT)
	{
		g_rx -= ROTATE_STEP;
		if (g_rx<1) g_rx = 1;
	}
	else if (key == GLUT_KEY_RIGHT)
	{
		g_rx += ROTATE_STEP;
		if (g_rx >= 179) g_rx = 179;
	}
	else if (key == GLUT_KEY_UP)
	{
		g_ry -= ROTATE_STEP;
		if (g_ry<-89) g_ry = -89;
	}
	else if (key == GLUT_KEY_DOWN)
	{
		g_ry += ROTATE_STEP;
		if (g_ry >= 89) g_ry = 89;
	}
	glutPostRedisplay();
}
