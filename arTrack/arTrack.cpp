#include <stdio.h>
#include <sys/time.h>

//GL
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

//ximea camera
#include "xiApi.h"
#include "xiExt.h"

//opencv
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//network
#include <arpa/inet.h>
#include <errno.h>
#include <unistd.h>

//network
#define NETWORK_DESTINATION "152.2.133.182" 
#define NETWORK_PORT 60000
int networkSocket;
struct sockaddr_in networkSocketAddress;

//ximea camera settings
int exposure_micro = 5000;
float gain_db = 22;
int downsample = 4;
int pixel_skip = 1; //skip pixels, rather than bin
int pixel_format = XI_RAW8;// XI_MONO8,XI_RAW8

//ximea calibration
float imgPlaneWidth = 2592*0.0022;	//sensor size
float imgPlaneHeight = 1944*0.0022;
float focalLength = 0.875958387*4;
int imageWidth = 648, imageHeight = 486;
float imageCenter[2] = {319.3654785,imageHeight-224.315933}; //calibrated camera center, adjusted so (0,0) is bottom left corner
int scaleFactor = 2;


//ximea state variables
XI_IMG image;
HANDLE xiH = NULL;
XI_RETURN stat = XI_OK;

//misc
long currentFrame = 0;
struct timeval lasttime;
int screenWidth, screenHeight;
float shiftY=0, shiftZ = 0;

//gl
GLuint tex;
float near = 1;				
float far = 1000;

//opencv
IplImage *imgOrig,*imgResize, *img,*imgGray;

//checkerboard
float square_size = 9.0;
int corner_rows = 5, corner_cols = 6;

//circles grid
bool useCircleDetector = false;
float circle_diameter = 5.2;
float circle_spacing = 6.4; //spacing between centers of columns/rows 
int circle_rows = 4;	    //note pair of staggered rows counts as 1 row
int circle_cols = 11;

CvMat* cam_mat;
CvMat* dist_coeff;

using namespace cv;

float lastRot[3] = {0};
float lastTrans[3] = {0};


#define HandleResult(res,place) if (res!=XI_OK) {printf("Error after %s (%d)\n",place,res); exit(1); }


void initNetwork() {

	networkSocket = socket(AF_INET, SOCK_DGRAM, 0);
   	bzero(&networkSocketAddress, sizeof(networkSocketAddress));
   	networkSocketAddress.sin_family = AF_INET;
   	networkSocketAddress.sin_addr.s_addr = inet_addr(NETWORK_DESTINATION);
   	networkSocketAddress.sin_port = htons(NETWORK_PORT);
}

void closeNetwork() {
	close(networkSocket);
}

void sendMatrix(float *mat) {


	int idx = 0;
	int bufSize = 1024;
	char msg[bufSize];
	for(int i = 0; i < 16; i++) {
		idx += snprintf(msg+idx, bufSize-idx, "%.4f ", mat[i]); 
		
	}
	//printf("message: '%s'\n", msg);	

	//send matrix as text
	int bytes = sendto(networkSocket, msg, strlen(msg), 0, (struct sockaddr *)&networkSocketAddress, sizeof(networkSocketAddress));
	if(bytes == -1) {
		printf("packet send failed, errno = %d\n", errno);
	}
}


void glutDisplay() {

	glClear(/*GL_COLOR_BUFFER_BIT|*/GL_DEPTH_BUFFER_BIT);


	//compute FPS
	int printNthFrame = 60;
	if(currentFrame%printNthFrame==0&&currentFrame!=0) {
		struct timeval now;
		gettimeofday(&now, NULL);

		double elapsed_sec = (now.tv_sec-lasttime.tv_sec)+(now.tv_usec-lasttime.tv_usec)/1000000.0;
		printf("avg fps: %f\n", printNthFrame/elapsed_sec); 

		lasttime.tv_sec = now.tv_sec;
		lasttime.tv_usec = now.tv_usec;
	}
	currentFrame++; 


	//get image
	stat = xiGetImage(xiH, 5000, &image);
	HandleResult(stat,"xiGetImage");

	//copy pointer to cv image
	imgOrig->imageData = (char*)image.bp;

	//downsample
	cvResize(imgOrig, imgResize, CV_INTER_LINEAR);

	//undistort
	cvUndistort2(imgResize, img, cam_mat, dist_coeff);

	//upload to texture
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screenWidth/scaleFactor, screenHeight/scaleFactor, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, img->imageData);

	//draw video frame in background
	glDepthMask(GL_FALSE);
	glEnable(GL_TEXTURE_2D);
	glColor3f(1,1,1);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0,1,0,1,-1,1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glBegin(GL_QUADS);
		glTexCoord2f(1,0); glVertex2f(0,0); //account for camera being mounted upside down
		glTexCoord2f(0,0); glVertex2f(1,0);
		glTexCoord2f(0,1); glVertex2f(1,1);
		glTexCoord2f(1,1); glVertex2f(0,1);
	glEnd();
	glDepthMask(GL_TRUE);

	//detect circles/checkerboard
	bool success = false;
	CvMat* image_points = cvCreateMat(useCircleDetector?(circle_rows*circle_cols):(corner_rows*corner_cols), 2, CV_32FC1 );
	CvMat* object_points = cvCreateMat(useCircleDetector?(circle_rows*circle_cols):(corner_rows*corner_cols), 3, CV_32FC1 );
	if(useCircleDetector) {	//circles
		std::vector<Point2f> centers; 
		Mat imgMat(img);
		float boardWidth = circle_spacing*(circle_cols-1);
	  	float boardHeight = circle_spacing*(2*circle_rows-1);


		SimpleBlobDetector::Params params;
		params.minArea = 10;
		params.minDistBetweenBlobs = 5;
		Ptr<FeatureDetector> blobDetector = new SimpleBlobDetector(params);
		success = findCirclesGrid(imgMat, Size(circle_rows, circle_cols), centers, CALIB_CB_ASYMMETRIC_GRID,blobDetector);
		if(success) {

			for(int j = 0; j < circle_rows*circle_cols; j++) {
				int x = j/circle_rows;
				int y = (j%circle_rows)*2+(x%2==1?1:0);

				float coordX = x*circle_spacing-(boardWidth/2.0);
				float coordY = y*circle_spacing-(boardHeight/2.0);

				CV_MAT_ELEM(*image_points, float, j, 0 ) = centers[j].x*scaleFactor;
				CV_MAT_ELEM(*image_points, float, j, 1 ) = centers[j].y*scaleFactor;
			 	CV_MAT_ELEM(*object_points, float, j, 0 ) = coordX;
			 	CV_MAT_ELEM(*object_points, float, j, 1 ) = coordY;
			 	CV_MAT_ELEM(*object_points, float, j, 2 ) = 0.0f;

				//drawChessboardCorners(imgMat, Size(circle_cols, circle_rows), Mat(centers), true);
		       	}     
		}
	} else { //checkerboard

		int corner_count = 0;
		CvPoint2D32f* corners = (CvPoint2D32f*)malloc(sizeof(CvPoint2D32f)*corner_rows*corner_cols);
		success = cvFindChessboardCorners(img, cvSize(corner_cols, corner_rows), corners, &corner_count,CV_CALIB_CB_ADAPTIVE_THRESH);

		if(success) {
			//cvCvtColor(img,imgGray, CV_RGB2GRAY);

			cvFindCornerSubPix(img, corners, corner_count, cvSize(4,4), cvSize(-1, -1),
                        	cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.01));

			for(int j = 0; j < corner_rows*corner_cols; j++) {
				float coordY = square_size*(j%corner_cols)-square_size*(corner_cols/2.0-0.5);
				float coordX = square_size*(j/corner_cols)-square_size*(corner_rows/2.0-0.5);
				CV_MAT_ELEM(*image_points, float, j, 0 ) = corners[j].x*scaleFactor;
				CV_MAT_ELEM(*image_points, float, j, 1 ) = corners[j].y*scaleFactor;
			 	CV_MAT_ELEM(*object_points, float, j, 0 ) = coordX;
			 	CV_MAT_ELEM(*object_points, float, j, 1 ) = coordY;
			 	CV_MAT_ELEM(*object_points, float, j, 2 ) = 0.0f;
			}
					
     			//cvDrawChessboardCorners(img, cvSize(corner_cols, corner_rows), corners, corner_rows*corner_cols, 1);
		}
	}

	if(success) {

		CvMat* rRot = cvCreateMat( 3, 1, CV_32FC1 );
		CvMat* rot = cvCreateMat( 3, 3, CV_32FC1 );
		CvMat* trans = cvCreateMat( 3, 1, CV_32FC1 );
		cvFindExtrinsicCameraParams2(object_points, image_points, cam_mat, dist_coeff, rRot, trans, 0);

		float t[3],r[3];
		for(int i = 0; i < 3; i++) {
			t[i] = CV_MAT_ELEM(*trans, float, i, 0);
			r[i] = CV_MAT_ELEM(*rRot, float, i, 0);
			CV_MAT_ELEM(*trans, float, i, 0) =(t[i]*0.5+lastTrans[i]*0.5); //smooth by weighting with last frame
			CV_MAT_ELEM(*rRot, float, i, 0) = (r[i]*0.5+lastRot[i]*0.5);
		}

		cvRodrigues2(rRot, rot, 0);


		CvMat* xform = cvCreateMat( 4, 4, CV_32FC1 );
		cvZero(xform);
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				CV_MAT_ELEM(*xform, float, i, j) = CV_MAT_ELEM(*rot, float, i, j);
			}
		}
		CV_MAT_ELEM(*xform, float, 0, 3) = CV_MAT_ELEM(*trans, float, 0, 0);
		CV_MAT_ELEM(*xform, float, 1, 3) = CV_MAT_ELEM(*trans, float, 1, 0);
		CV_MAT_ELEM(*xform, float, 2, 3) = CV_MAT_ELEM(*trans, float, 2, 0);
		CV_MAT_ELEM(*xform, float, 3, 3) = 1;


		for(int i = 0; i < 3; i++) {
			lastTrans[i] = t[i];
			lastRot[i] = r[i];
		}


		float mat[16];
		for(int i = 0; i < 16; i++) {
			mat[i] = CV_MAT_ELEM(*xform, float, i%4, i/4);
			//printf("%f, ", mat[i]);
		}
		//printf("\n");

		sendMatrix(mat);

		glEnable(GL_DEPTH_TEST);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		
		//float aspect = (float)screenWidth/screenHeight;
		//gluPerspective(fov/aspect, aspect, 1, 1000);
		float w = imgPlaneWidth/focalLength*near, h = imgPlaneHeight/focalLength*near;
		//glFrustum(-imageCenter[0]/imageWidth*w, (imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[1]/imageHeight*h,(imageHeight-imageCenter[1])/imageHeight*h, near, far); 
		glFrustum((imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[0]/imageWidth*w,  (imageHeight-imageCenter[1])/imageHeight*h, -imageCenter[1]/imageHeight*h, near, far); //flip l,r and b,t to account for camera being upside down

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glScalef(1,-1,-1);
		glMultMatrixf(mat);

		//glRotatef(90,1,0,0);
		//glRotatef(-90,0,1,0);

		//glTranslatef(31.5,shiftY,shiftZ);
		//glRotatef(90,0,0,1);

		//glutSolidTeapot(12.0);
		glDisable(GL_TEXTURE_2D);
		glColor3f(1,0,1);
		float sx = useCircleDetector?(circle_spacing*(circle_cols-1)+circle_diameter):(square_size*(corner_rows+1));
		float sy = useCircleDetector?(circle_spacing*(2*circle_rows-1)+circle_diameter):(square_size*(corner_cols+1));
		glBegin(GL_QUADS);
			glVertex2f(-sx/2,-sy/2);
			glVertex2f(sx/2,-sy/2);
			glVertex2f(sx/2,sy/2);
			glVertex2f(-sx/2,sy/2);
		glEnd();
		glPointSize(15);
		glBegin(GL_POINTS);
			glColor3f(1,0,0); glVertex2f(-sx/2,-sy/2);
			glColor3f(0,1,0); glVertex2f( sx/2,-sy/2);
			glColor3f(0,0,1); glVertex2f(-sx/2, sy/2);
			glColor3f(1,1,0); glVertex3f(0,0,50);
		glEnd();
		glBegin(GL_LINES);
			glColor3f(1,1,0); glVertex3f(0,0,0); glVertex3f(0,0,50);
		glEnd();

		cvReleaseMat(&image_points);
		cvReleaseMat(&object_points);
		cvReleaseMat(&rRot);
		cvReleaseMat(&rot);
		cvReleaseMat(&trans);
		cvReleaseMat(&xform);
	} else {
		printf("pattern not found!\n");
	}

	glutSwapBuffers();
}

void glutIdle() {
 
	glutPostRedisplay();

}

void glutKeyboard(unsigned char key, int x, int y) {
 
	switch(key) {
		case 27: closeNetwork(); if (xiH) xiCloseDevice(xiH); exit(0); break;
		case 'q': shiftY-=1.0; printf("shift y %f z %f\n",shiftY,shiftZ); break;
		case 'w': shiftY+=1.0; printf("shift y %f z %f\n",shiftY,shiftZ); break;
		case 'a': shiftZ-=1.0; printf("shift y %f z %f\n",shiftY,shiftZ); break;
		case 's': shiftZ+=1.0; printf("shift y %f z %f\n",shiftY,shiftZ); break;
		case 'z': focalLength-=0.1; printf("focal length %f\n", focalLength); break;	
		case 'x': focalLength+=0.1; printf("focal length %f\n", focalLength); break;
	}
}


void initXimea() {

	//setimage buffer
	image.size = sizeof(XI_IMG);
	image.bp = NULL;
	image.bp_size = 0;

	//check number of devices
	DWORD dwNumberOfDevices = 0;
	stat = xiGetNumberDevices(&dwNumberOfDevices);
	HandleResult(stat,"xiGetNumberDevices (no camera found)");
	if (!dwNumberOfDevices) {
		printf("No camera found\n");
		exit(1);
	}
	
	//open camera device
	stat = xiOpenDevice(0, &xiH);
	HandleResult(stat,"xiOpenDevice");
	
	//set exposure
	stat = xiSetParamInt(xiH, XI_PRM_EXPOSURE, exposure_micro);
	HandleResult(stat,"xiSetParam (exposure set)");

	//set gain
	stat = xiSetParamInt(xiH, XI_PRM_GAIN, gain_db);
	HandleResult(stat,"xiSetParam (gain set)");
	
	//set downsampling
	stat = xiSetParamInt(xiH, XI_PRM_DOWNSAMPLING, downsample);
	HandleResult(stat,"xiSetParam (downsample set)");

	//set skipping/binning
	stat = xiSetParamInt(xiH, XI_PRM_DOWNSAMPLING_TYPE, pixel_skip);
	HandleResult(stat,"xiSetParam (downsample type set)");

	//set pixel format
	stat = xiSetParamInt(xiH, XI_PRM_IMAGE_DATA_FORMAT, pixel_format);
	HandleResult(stat,"xiSetParam (pixel format set)");

	//set buffers queue 
	xiSetParamInt(xiH, XI_PRM_BUFFERS_QUEUE_SIZE, 1);
	HandleResult(stat,"xiSetParam (buffers queue size)");

	//set to retreive most recent frame
	xiSetParamInt(xiH, XI_PRM_RECENT_FRAME, 1);
	HandleResult(stat,"xiSetParam (recent frame)");

	stat = xiStartAcquisition(xiH);
	HandleResult(stat,"xiStartAcquisition");
}

int main(int argc, char** argv)
{

	//network init
	initNetwork();

	//camera init
	initXimea();

	//get image from camera to get the size
	stat = xiGetImage(xiH, 5000, &image);
	HandleResult(stat,"xiGetImage");
	
	screenWidth = (int)image.width; 
	screenHeight = (int)image.height;


	//OpenGL init
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	glutInitWindowSize(screenWidth,screenHeight);
	glutCreateWindow ("Camera Capture");
	glutDisplayFunc(glutDisplay); 
	glutKeyboardFunc(glutKeyboard);
	glutIdleFunc(glutIdle);

	//create texture
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//create cv images
	imgOrig = cvCreateImage(cvSize(screenWidth,screenHeight),IPL_DEPTH_8U,1);
	imgResize = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,1);
	img = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,1);
	imgGray = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,1);

	cam_mat = (CvMat*)cvLoad("cam_rgb_0.xml", NULL, NULL, NULL);
  	dist_coeff = (CvMat*)cvLoad("distort_rgb_0.xml",  NULL, NULL, NULL);
	if(cam_mat == NULL || dist_coeff == NULL) {
		printf("can't load camera calibration, exiting...\n");
		exit(1);
	}


	glutMainLoop();



	return 0;
}

