#include <stdio.h>
#include <sys/time.h>

//GL
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

//PointGrey
#include "stdafx.h"
#include "FlyCapture2.h"

//opencv
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//cvblob
#include <cvblob.h>

//ximea calibration
float imgPlaneWidth = 1024*0.0046875;	//sensor size
float imgPlaneHeight = 768*0.0046875;
float focalLength = 2.846840287;
int imageWidth = 1024, imageHeight = 768;
float imageCenter[2] = {547.927368,imageHeight-377.493683}; //calibrated camera center, adjusted so (0,0) is bottom left corner
int scaleFactor = 1;

//misc
long currentFrame = 0;
struct timeval lasttime;
int screenWidth, screenHeight;

//gl
GLuint tex;
float near = 1;				
float far = 1000;

// PointGrey
FlyCapture2::Camera cam;

//opencv
IplImage *imgOrig,*imgResize, *img,*imgGray;

//checkerboard
float square_size = 2.4;
int corner_rows = 9, corner_cols = 6;

//circles grid
bool useCircleDetector = true;
float circle_diameter = 5.2;
float circle_spacing = 6.4; //spacing between centers of columns/rows 
int circle_rows = 4;	    //note pair of staggered rows counts as 1 row
int circle_cols = 11;

CvMat* cam_mat;
CvMat* dist_coeff;

//Modes
std::string modeDetail[] = {
  "Calibration pattern"
  , "Detect cue ball"
};

int mode = 0;
int numModes = sizeof(modeDetail)/sizeof(modeDetail[0]);
FILE* pFile;

// Cueball colors

// Saving images
bool tSaveImage = false, tSaveHSVImage = false;

// misc
bool first = true;

using namespace cv;

float lastRot[3] = {0};
float lastTrans[3] = {0};

void PrintError( FlyCapture2::Error error ) {
  error.PrintErrorTrace();
}

// void printMat2(cvMat curr) {
//   int rows = curr.rows;
//   int cols = curr.cols;
  
//   for(int i = 0; i < rows; i++) {
//     for(int j = 0; j < cols; j++) {
//       printf("%f, ", CV_MAT_ELEM(curr, float, i, j));
//     }
//     printf("\n");
//   }
// }

void saveImage() {
  cvSaveImage("./outputs/imgOrig.jpg", imgOrig);
  cvSaveImage("./outputs/img.jpg", img);
  printf("Done saving images...\n");
}

void saveHSVImage() {
  IplImage *hsvImg;
  hsvImg = cvCreateImage(cvSize(screenWidth,screenHeight),IPL_DEPTH_8U,3);

  cvCvtColor(imgOrig, hsvImg, CV_BGR2HSV);
  cvSaveImage("./outputs/hsv_imgOrig.jpg", hsvImg);
  
  cvCvtColor(img, hsvImg, CV_BGR2HSV);
  cvSaveImage("./outputs/hsv_img.jpg", hsvImg);
  printf("Done saving images...\n");
  
}

void detectCalibrationPattern() {

  //undistort
  cvUndistort2(imgResize, img, cam_mat, dist_coeff);

  //upload to texture
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screenWidth/scaleFactor, screenHeight/scaleFactor, 0, GL_BGR, GL_UNSIGNED_BYTE, img->imageData);

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
  glTexCoord2f(0,1); glVertex2f(0,0); //account for camera being mounted upside down
  glTexCoord2f(1,1); glVertex2f(1,0);
  glTexCoord2f(1,0); glVertex2f(1,1);
  glTexCoord2f(0,0); glVertex2f(0,1);
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
      cvCvtColor(imgResize,imgGray, CV_RGB2GRAY);

      cvFindCornerSubPix(imgGray, corners, corner_count, cvSize(7, 7), cvSize(-1, -1),
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
    cvFindExtrinsicCameraParams2(object_points, image_points, cam_mat, NULL, rRot, trans, 0);
    if(first) {
      std::cout << "cam_mat = " << std::endl << " " << format(cam_mat, "c") << std::endl << std::endl;
      std::cout << "object_points = " << std::endl << " " << format(object_points, "c") << std::endl << std::endl;
      std::cout << "image_points = " << std::endl << " " << format(image_points, "c") << std::endl << std::endl;
      std::cout << "rRot = " << std::endl << " " << format(rRot, "c") << std::endl << std::endl;
      std::cout << "trans = " << std::endl << " " << format(trans, "c") << std::endl << std::endl;
    }
    // printMat2(cam_mat);
    // Computes rotation and translation required to convert model coordinate system to camera coordinate system

    cvRodrigues2(rRot, rot, 0);

    if(first) {
      std::cout << "rot = " << std::endl << " " << format(rot, "c") << std::endl << std::endl;
    }


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

    if(first) {
      std::cout << "xform = " << std::endl << " " << format(xform, "c") << std::endl << std::endl;
    }

    float mat[16];
    for(int i = 0; i < 16; i++) {
      mat[i] = CV_MAT_ELEM(*xform, float, i%4, i/4);
      //printf("%f, ", mat[i]);
    }
    //printf("\n");

    // if(first) {
    //   std::cout << "mat = " << std::endl << " " << format(mat, "c") << std::endl << std::endl;
    // }

    // sendMatrix(mat);

    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
		
    //float aspect = (float)screenWidth/screenHeight;
    //gluPerspective(fov/aspect, aspect, 1, 1000);
    float w = imgPlaneWidth/focalLength*near, h = imgPlaneHeight/focalLength*near;
    glFrustum(-imageCenter[0]/imageWidth*w, (imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[1]/imageHeight*h,(imageHeight-imageCenter[1])/imageHeight*h, near, far); 
    //glFrustum((imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[0]/imageWidth*w,  (imageHeight-imageCenter[1])/imageHeight*h, -imageCenter[1]/imageHeight*h, near, far); //flip l,r and b,t to account for camera being upside down

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
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(1,0,1, 0.5);
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

  first = false;

}

// void detectCueball2() {
//   cv::Mat matOrig(imgOrig), resultImg;
//   cv::SimpleBlobDetector::Params params;
//   params.minDistBetweenBlobs = 1;


//   // By area
//   params.filterByArea = true;
//   // The values below are obtained from the color picker tool in gimp
//   params.minArea = 20*20;
//   params.maxArea = 34*34;

//   // By circularity
//   params.filterByCircularity = true;
//   params.minCircularity = 0.8f;
//   params.maxCircularity = 1.0f;
 
//   // By color
//   params.filterByColor = true;
//   params.blobColor = cvScalar(150, 200, 45);

//   SimpleBlobDetector blobDetector(params);
//   blobDetector.create("SimpleBlob");
 
//   vector<KeyPoint> keypoints;
  
//   blobDetector.detect(matOrig, keypoints);
//   drawKeypoints(matOrig, blob, resultImg, DrawMatchesFlags::DEFAULT);
 
//   imshow("Blobs", resultImg);
// }


void detectCueball() {

  //upload to texture
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screenWidth/scaleFactor, screenHeight/scaleFactor, 0, GL_BGR, GL_UNSIGNED_BYTE, imgOrig->imageData);

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
  glTexCoord2f(0,1); glVertex2f(0,0); //account for camera being mounted upside down
  glTexCoord2f(1,1); glVertex2f(1,0);
  glTexCoord2f(1,0); glVertex2f(1,1);
  glTexCoord2f(0,0); glVertex2f(0,1);
  glEnd();
  glDepthMask(GL_TRUE);

  //Start with imgOrig
  IplImage *hsvImg, *inRangesImg, *labelImg;
  cvb::CvBlobs blobs;
  hsvImg = cvCreateImage(cvSize(screenWidth,screenHeight),IPL_DEPTH_8U,3);
  inRangesImg = cvCreateImage(cvSize(screenWidth,screenHeight),IPL_DEPTH_8U,1);
  labelImg = cvCreateImage(cvSize(screenWidth,screenHeight),IPL_DEPTH_LABEL,1);

  cvCvtColor(imgOrig, hsvImg, CV_BGR2HSV);
  cvInRangeS(hsvImg, cvScalar(45, 175, 20), cvScalar(270, 230, 60), inRangesImg);

  cvShowImage("image", inRangesImg);
  cvWaitKey(1000);

  cvSmooth(inRangesImg, inRangesImg, CV_MEDIAN, 3, 3);
  unsigned int result = cvb::cvLabel(inRangesImg, labelImg, blobs);
 
}

void detectCircles() {

  Mat matOrig(imgOrig);
  Mat matGray;
  cvtColor( matOrig, matGray, CV_BGR2GRAY );
  GaussianBlur( matGray, matGray, Size(7,7), 2, 2);
  
  vector<Vec3f> circles;
  
  HoughCircles( matGray, circles, CV_HOUGH_GRADIENT, 1, 1, 20, 20, 10, 20);

  for(size_t i = 0; i < circles.size(); i++) {
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    // circle center
    circle( matOrig, center, 3, Scalar(0,255,0), -1, 8, 0 );
    // circle outline
    circle( matOrig, center, radius, Scalar(0,0,255), 3, 8, 0 );
  }

  namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
  imshow( "Hough Circle Transform Demo", matOrig );
  waitKey(1000);
}

void glutDisplay() {

  FlyCapture2::Image rgbImage;
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

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

  FlyCapture2::Image rawImage; 
  FlyCapture2::Error flyCapError;

  rawImage.SetDefaultColorProcessing(FlyCapture2::HQ_LINEAR);
  flyCapError = cam.RetrieveBuffer( &rawImage );
  if (flyCapError != FlyCapture2::PGRERROR_OK) {
    PrintError( flyCapError );
  }

  // Convert image to opencv format and observe the image
  rawImage.Convert( FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage );
  imgOrig->imageData = (char*)rgbImage.GetData();

  // cvShowImage("image", imgOrig);
  // cvWaitKey(1000);

  if(tSaveImage) {
    saveImage();
    tSaveImage = false;
  }

  if(tSaveHSVImage) {
    saveHSVImage();
    tSaveHSVImage = false;
  }

  //downsample
  cvResize(imgOrig, imgResize, CV_INTER_LINEAR);

  if(mode == 0) 
    detectCalibrationPattern();

  if(mode == 1)
    detectCircles();

    // detectCueball();

  glutSwapBuffers();
}

void glutIdle() {
 
  glutPostRedisplay();

}

int disconnectCamera() {
  FlyCapture2::Error flyCapError;

  // Stop capturing images
  flyCapError = cam.StopCapture();
  if (flyCapError != FlyCapture2::PGRERROR_OK)
    {
      PrintError( flyCapError );
      return -1;
    }       
 
  // Disconnect the camera
  flyCapError = cam.Disconnect();
  if (flyCapError != FlyCapture2::PGRERROR_OK)
    {
      PrintError( flyCapError );
      return -1;
    }
}

void keyPressed(unsigned char key, int x, int y) {
 
  switch(key) {
  case 27: disconnectCamera(); exit(0); break;
  case 'z': focalLength-=0.1; printf("focal length %f\n", focalLength); break;	
  case 'x': focalLength+=0.1; printf("focal length %f\n", focalLength); break;
  case 'm': mode=++mode%numModes; printf("current mode: %s \n", modeDetail[mode].c_str()); break;
  case 'p': tSaveImage = true;    break;
  case 'P': tSaveHSVImage = true; break;
  }
}

void PrintCameraInfo( FlyCapture2::CameraInfo* pCamInfo )
{
  printf(
	 "\n*** CAMERA INFORMATION ***\n"
	 "Serial number - %u\n"
	 "Camera model - %s\n"
	 "Camera vendor - %s\n"
	 "Sensor - %s\n"
	 "Resolution - %s\n"
	 "Firmware version - %s\n"
	 "Firmware build time - %s\n\n",
	 pCamInfo->serialNumber,
	 pCamInfo->modelName,
	 pCamInfo->vendorName,
	 pCamInfo->sensorInfo,
	 pCamInfo->sensorResolution,
	 pCamInfo->firmwareVersion,
	 pCamInfo->firmwareBuildTime );
}

int initPointGreyCamera() {

  FlyCapture2::Error flyCapError;
  FlyCapture2::BusManager busMgr;
  FlyCapture2::Image rawImage; 
  FlyCapture2::PGRGuid guid;

  unsigned int numCameras;

  flyCapError = busMgr.GetNumOfCameras(&numCameras);
  if (flyCapError != FlyCapture2::PGRERROR_OK) {
    PrintError( flyCapError );
    return -1;
  }

  if ( numCameras < 1 ) {
    printf( "No camera detected.\n" );
    return -1;
  }
  else {
    printf( "Number of cameras detected: %u\n", numCameras );
  }

  flyCapError = busMgr.GetCameraFromIndex(0, &guid);
  if (flyCapError != FlyCapture2::PGRERROR_OK) {
    PrintError( flyCapError );
    return -1;
  }
  // Connect to a camera
  flyCapError = cam.Connect(&guid);
  if (flyCapError != FlyCapture2::PGRERROR_OK)
    {
      PrintError( flyCapError );
      return -1;
    }

  // Get the camera information
  FlyCapture2::CameraInfo camInfo;
  flyCapError = cam.GetCameraInfo(&camInfo);
  if (flyCapError != FlyCapture2::PGRERROR_OK)
    {
      PrintError( flyCapError );
      return -1;
    }

  PrintCameraInfo(&camInfo);  

  // Start capturing images
  printf( "Start capturing... \n" );
  flyCapError = cam.StartCapture();
  if (flyCapError != FlyCapture2::PGRERROR_OK)
    {
      PrintError( flyCapError );
      return -1;
    }

  flyCapError = cam.RetrieveBuffer( &rawImage );
  if (flyCapError != FlyCapture2::PGRERROR_OK) {
    PrintError( flyCapError );
  }

  screenWidth = rawImage.GetCols();
  screenHeight = rawImage.GetRows();


}

int main(int argc, char** argv) {

  // PointGrey init
  initPointGreyCamera();

  //OpenGL init
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
  glutInitWindowSize(screenWidth,screenHeight);
  glutCreateWindow ("Camera Capture");
  glutDisplayFunc(glutDisplay); 
  glutKeyboardFunc(keyPressed);
  glutIdleFunc(glutIdle);

  //create texture
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  //create cv images
  imgOrig   = cvCreateImage(cvSize(screenWidth,screenHeight),IPL_DEPTH_8U,3);
  imgResize = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,3);
  img       = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,3);
  imgGray   = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,1);

  cam_mat = (CvMat*)cvLoad("cam_rgb_0.xml", NULL, NULL, NULL);
  dist_coeff = (CvMat*)cvLoad("distort_rgb_0.xml",  NULL, NULL, NULL);
  (CvMat*)cvLoad("distort_rgb_0.xml",  NULL, NULL, NULL);
  if(cam_mat == NULL || dist_coeff == NULL) {
    printf("can't load camera calibration, exiting...\n");
    exit(1);
  }

  glutMainLoop();

}
