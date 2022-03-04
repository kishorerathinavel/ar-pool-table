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

CvMat *cam_mat, *extrinsics;
CvMat *invExt, *invInt;
CvMat *dist_coeff;

//Modes
std::string modeDetail[] = {
  "Calibration pattern"
  , "Detect cue ball using blob detection"
  , "Calibrating for the pocket positions"
  , "Displaying last found shot"
};

int mode = 1;
int numModes = sizeof(modeDetail)/sizeof(modeDetail[0]);
FILE* pFile;

// Saving images
bool tSaveImage = false, tSaveHSVImage = false;

bool first = true, tSim = false;

using namespace cv;
using namespace std;

RNG rng(12345);

// For 2d to 3d conversion.
Mat wcWOrigin = (Mat_<float>(4,1) << 0.0, 0.0, 0.0, 1.0 );
Mat wcWXpoint = (Mat_<float>(4,1) << 1.0, 0.0, 0.0, 1.0 );
Mat wcWYpoint = (Mat_<float>(4,1) << 0.0, 1.0, 0.0, 1.0 );
Mat wcWZpoint = (Mat_<float>(4,1) << 0.0, 0.0, 1.0, 1.0 );
Mat ccCOrigin = (Mat_<float>(4,1) << 0.0, 0.0, 0.0, 1.0 );
Mat ccWOrigin, ccWXpoint, ccWYpoint, ccWZpoint, ccWXaxis, ccWYaxis, ccWZaxis, mDistance;
Mat mInt, mExt, mInvInt, mInvExt;

// For drawing the shot
Mat whiteCue, targetCue, prevWhiteCue, prevTargetCue, finalWhiteCuePos;
int targetPocket = -1, targetedCue = 0;
bool moveWhite;

// For calculating the shot
Mat pocketPositions;
int maxPocketPos = 6, countKnownPocketPos = 0, simCount = 0, maxCount = 10;

float lastRot[3] = {0};
float lastTrans[3] = {0};
float matExt[16];

void printMat(Mat curr) {
  printf("rows: %d cols: %d \n", curr.rows, curr.cols);
  for(int i = 0; i < curr.rows; i++) {
    std::cout << format(curr.row(i), "c") << std::endl; 
  }
  std::cout << std::endl;
}

void PrintError( FlyCapture2::Error error ) {
  error.PrintErrorTrace();
}

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

    cvSave("extrinsics.xml", xform);

    if(first) {
      std::cout << "xform = " << std::endl << " " << format(xform, "c") << std::endl << std::endl;
    }

    for(int i = 0; i < 16; i++) {
      matExt[i] = CV_MAT_ELEM(*xform, float, i%4, i/4);
      //printf("%f, ", matExt[i]);
    }
    //printf("\n");

    // if(first) {
    //   std::cout << "matExt = " << std::endl << " " << format(matExt, "c") << std::endl << std::endl;
    // }

    // sendMatrix(matExt);

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
    glMultMatrixf(matExt);

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

float median1ch (cv::Mat image) {
  double m=(image.rows*image.cols)/2;
  int bin = 0;
  float med;
  med =-1;
  int histSize = 256;
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist;
  cv::calcHist( &image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  for (int i=0; i<256 && ( med <0 );i++) {
    bin = bin + cvRound(hist.at<float>(i));
    if (bin>m && med<0)
      med=i;
  }

  return med;
}

cv::Scalar median3ch (cv::Mat image) {
  double m=(image.rows*image.cols)/2;
  int bin0=0, bin1=0, bin2=0;
  cv::Scalar med;
  med.val[0]=-1;
  med.val[1]=-1;
  med.val[2]=-1;
  int histSize = 256;
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  bool uniform = true;
  bool accumulate = false;
  cv::Mat hist0, hist1, hist2;
  std::vector<cv::Mat> channels;
  cv::split( image, channels );
  cv::calcHist( &channels[0], 1, 0, cv::Mat(), hist0, 1, &histSize, &histRange, uniform, accumulate );
  cv::calcHist( &channels[1], 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange, uniform, accumulate );
  cv::calcHist( &channels[2], 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange, uniform, accumulate );

  for (int i=0; i<256 && ( med.val[0]<0 || med.val[1]<0 || med.val[2]<0);i++)
    {
      bin0=bin0+cvRound(hist0.at<float>(i));
      bin1=bin1+cvRound(hist1.at<float>(i));
      bin2=bin2+cvRound(hist2.at<float>(i));
      if (bin0>m && med.val[0]<0)
	med.val[0]=i;
      if (bin1>m && med.val[1]<0)
	med.val[1]=i;
      if (bin2>m && med.val[2]<0)
	med.val[2]=i;
    }

  return med;
}

void init2Dto3Dconversion() {

  ccWOrigin = mExt * wcWOrigin;
  ccWXpoint = mExt * wcWXpoint;
  ccWYpoint = mExt * wcWYpoint;
  ccWZpoint = mExt * wcWZpoint;

  ccWXaxis = ccWXpoint - ccWOrigin;
  ccWYaxis = ccWYpoint - ccWOrigin;
  ccWZaxis = ccWZpoint - ccWOrigin;

  // Normal to the plane
  normalize(ccWZaxis, ccWZaxis);

  // Distance from camera origin to plane along plane's normal
  mDistance = ccWOrigin.t() * ccWZaxis;

  for(int i = 0; i < 16; i++) {
    matExt[i] = CV_MAT_ELEM(*extrinsics, float, i%4, i/4);
  }
}

Mat convert2Dto3D (Mat pt2d) {

  // Mat pt2d = (Mat_<float>(3,1) << sortedKeypoints[i].pt.x, sortedKeypoints[i].pt.y, 1.0 );
  Mat ccPt3d = invInt * pt2d; 

 
  // Determining the direction of ray traveling between the center of projection to the detected 2D keypoint
  Mat ccRay;
  normalize(ccPt3d, ccRay);
  Mat lastElement = (Mat_<float>(1,1) << 0.0 );
  ccRay.push_back(lastElement);
  
  // Ray plane intersection: 
  // Distance to travel along ray to hit plane
  Mat t = ccRay.t()*ccWZaxis;
  t = mDistance / t;

  // Point on plane where the ray intersects
  float scalart = t.at<float>(0,0);
  Mat ccIntPt = ccCOrigin + scalart*ccRay;
  // Converting intersection point from camera coordinates to world coordinates

  Mat wcIntPt = invExt * ccIntPt;
  return wcIntPt;
}

void calculateShot() {

  targetPocket = -1;
  IplImage *hsvImg;
  IplImage *H, *S, *V;
  hsvImg = cvCreateImage(cvSize(screenWidth,screenHeight),IPL_DEPTH_8U,3);

  cvCvtColor(img, hsvImg, CV_BGR2HSV);

  H = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,1);
  S = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,1);
  V = cvCreateImage(cvSize(screenWidth/scaleFactor,screenHeight/scaleFactor),IPL_DEPTH_8U,1);
  
  cvSplit(hsvImg, H, S, V, NULL);

  if(first) {
    cvSaveImage("./outputs/imgOrig.jpg", imgOrig);
    cvSaveImage("./outputs/img.jpg", img);
    cvSaveImage("./outputs/H.jpg", H);
    cvSaveImage("./outputs/S.jpg", S);
    cvSaveImage("./outputs/V.jpg", V);
  }
  Mat oH1(H), oS1(S), oV1(V);
  Mat mH1, mH2, mH3, mH4, mH5;
  Mat resultImg;


  inRange(oH1, Scalar(42), Scalar(55), mH1);
  // imshow("Stage 2", mH1);
  morphologyEx(mH1, mH1, MORPH_OPEN, Mat(), Point(-1, -1), 2);
  // imshow("Stage 3", mH1);
  morphologyEx(mH1, mH1, MORPH_CLOSE, Mat(), Point(-1, -1), 25);
  bitwise_and(oH1, mH1, mH2);
  // imshow("Stage 5", mH2);
  // if(first) 
  //   imwrite("./outputs/roi.jpg", mH2);
  inRange(mH2, Scalar(0), Scalar(40), mH3);
  bitwise_and(mH1, mH3, mH3);
  // imshow("Stage 6", mH3);
  inRange(mH2, Scalar(54), Scalar(255), mH4);
  bitwise_or(mH3, mH4, mH5);
  morphologyEx(mH5, mH5, MORPH_CLOSE, Mat(), Point(-1, -1), 4);
  medianBlur(mH5, mH5, 3);
  // imshow("Stage 7", mH5);

  mH1 = mH5;
  // imshow("mH1", mH1);
  // waitKey(1);
  // medianBlur(mH1, mH1, 9);
  // waitKey(1);
  // morphologyEx(mH1, mH1, MORPH_CLOSE, Mat(), Point(-1, -1), 5);
  // medianBlur(mH1, mH1, 9);
  // imshow("Stage 3", mH1);
  // waitKey(1);
  // bitwise_not(mH1, mH1);
  // imshow("Stage 4", mH1);
  // waitKey(1);

  // vector<vector<Point> > contours;
  // vector<Vec4i> hierarchy;
  // findContours(mH1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  // /// Draw contours
  // Mat drawing = Mat::zeros( oH1.size(), CV_8UC3 );
  // for( int i = 0; i< contours.size(); i++ ) {
  //   Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
  //   drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
  // }

  // /// Get the moments
  // vector<Moments> mu(contours.size() );
  // for( int i = 0; i < contours.size(); i++ )
  //   { mu[i] = moments( contours[i], false ); }

  // ///  Get the mass centers:
  // vector<Point2f> mc( contours.size() );
  // for( int i = 0; i < contours.size(); i++ )
  //   { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }   /// Show in a window

  // namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  // imshow( "Contours", drawing );

  // if(true) {
  //   waitKey(1);
  //   first = false;
  //   cvReleaseImage(&H);
  //   cvReleaseImage(&S);
  //   cvReleaseImage(&V);
  //   cvReleaseImage(&hsvImg);
  //   return;
  // }



  SimpleBlobDetector::Params params;
  params.minDistBetweenBlobs = 10;

  params.filterByInertia = false;
  params.filterByConvexity = false;
  params.filterByColor = false;
  params.filterByCircularity = true;
  params.filterByArea = true;

  params.minArea = 100.0f;
  params.minCircularity = 0.6f;

  Ptr<FeatureDetector> blobDetector = new SimpleBlobDetector(params);
  blobDetector->create("SimpleBlob");
 
  vector<KeyPoint> keypoints;
  blobDetector->detect(mH1, keypoints);

  if(keypoints.size() == 0)  {
    first = false;
    cvReleaseImage(&H);
    cvReleaseImage(&S);
    cvReleaseImage(&V);
    cvReleaseImage(&hsvImg);
    return;
  }

  // std::cout << keypoints.size() << std::endl;
  // drawKeypoints(oH1, keypoints, resultImg, Scalar::all(255), DrawMatchesFlags::DEFAULT);
  // imshow("resultImg", resultImg);
  // waitKey(1);

  // if(true) {
  //   waitKey(1);
  //   first = false;
  //   cvReleaseImage(&H);
  //   cvReleaseImage(&S);
  //   cvReleaseImage(&V);
  //   cvReleaseImage(&hsvImg);
  //   return;
  // }

   // Converting the keypoint to 3D coordinates
 
  // if(first) {
  //   std::cout << "cam_mat = " << std::endl << " " << format(cam_mat, "c") << std::endl << std::endl;
  //   std::cout << "invInt = " << std::endl << " " << format(invInt, "c") << std::endl << std::endl; 
  //   std::cout << "extrinsics = " << std::endl << " " << format(extrinsics, "c") << std::endl << std::endl; 
  //   std::cout << "invExt = " << std::endl << " " << format(invExt, "c") << std::endl << std::endl; 
  // }

  // Prefixes: 
  // wc - world coordinates
  // cc - camera coordinates

  // Finding out the white cue ball
  Mat medH;
  Scalar medh;
  int pixelRadius = 10;
  for(int i = 0; i < keypoints.size(); i++) {
    Rect roi( floor(keypoints[i].pt.x) - pixelRadius, floor(keypoints[i].pt.y) - pixelRadius, 2*pixelRadius, 2*pixelRadius );
    if(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= oV1.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= oV1.rows) {
      Mat roiImg = oV1( roi );
      medh = mean(roiImg);
      medH.push_back(medh[0]);
    }
    else{
      printf("%d %d %d %d \n", roi.x, roi.y, roi.width, roi.height);
      continue;
    }
  }
  medH.convertTo(medH, CV_32FC1);

  // Mat sortOrder;
  // sortIdx(medH, sortOrder, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
  // sort(medH, medH, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

  int whiteIdx = 0;
  float min = 100000;
  float max = -1;
  for(int i = 0; i < medH.rows; i++) {
    // Using the highest mean from Value channel
    if(medH.at<float>(i,0) > max){
      max = medH.at<float>(i,0);
      whiteIdx = i;
    }

    // // Using the difference between the mean and the hue value of 19 (determined from gimp)
    // float diff = medH.at<float>(i,0) - 19;
    // // if(first)
    // //   printf("diff %f \n", diff);
    // if(diff*diff < min) {
    //   min = diff*diff;
    //   whiteIdx = i;
    // }
  }
  // printf("whiteIdx = %d \n", whiteIdx);
  // vector<KeyPoint> sortedKeypoints;
  // for(int i = 0; i < keypoints.size(); i++) 
  //   sortedKeypoints.push_back(keypoints[sortOrder.at<int>(i,0)]);
  
  Mat keypoints3D;
  Mat cuePos;
  for(int i = 0; i < keypoints.size(); i++) {
    Mat pt2d = (Mat_<float>(3,1) << keypoints[i].pt.x, keypoints[i].pt.y, 1.0 );
    Mat wcIntPt = convert2Dto3D(pt2d);
    wcIntPt = wcIntPt.t();
    
    if(i == whiteIdx)
      cuePos = wcIntPt;
    else
      keypoints3D.push_back(wcIntPt);
  }

  if(keypoints3D.rows == 0) {
    first = false;
    cvReleaseImage(&H);
    cvReleaseImage(&S);
    cvReleaseImage(&V);
    cvReleaseImage(&hsvImg);
    return;
  }

  // Drawing the keypoints and lines between them

  glEnable(GL_DEPTH_TEST);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  float w = imgPlaneWidth/focalLength*near, h = imgPlaneHeight/focalLength*near;

  glFrustum(-imageCenter[0]/imageWidth*w, (imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[1]/imageHeight*h,(imageHeight-imageCenter[1])/imageHeight*h, near, far); 

  //flip l,r and b,t to account for camera being upside down
  //glFrustum((imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[0]/imageWidth*w,  (imageHeight-imageCenter[1])/imageHeight*h, -imageCenter[1]/imageHeight*h, near, far); 

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glScalef(1,-1,-1);
  glMultMatrixf(matExt);

  glDisable(GL_TEXTURE_2D);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glColor4f(1,0,1, 0.5);

  // Drawing the white sphere over the white cue ball
  glPushMatrix();
  glTranslatef(cuePos.at<float>(0,0),cuePos.at<float>(0,1), 0.0f);
  glColor3f(1.0, 1.0, 1.0);
  glutSolidSphere(2.858, 100, 100);
  glPopMatrix();

  for(int i = 0; i < keypoints3D.rows; i++) {
    glPushMatrix();
    glTranslatef(keypoints3D.at<float>(i,0),keypoints3D.at<float>(i,1), 0.0f);
    glColor3f(0.7, 0.7, 0.7);
    glutSolidSphere(2.858, 100, 100);
    glPopMatrix();

    // glBegin(GL_LINES);
    // glColor3f(1,1,0); 
    // glVertex3f(keypoints3D.at<float>(i,0),keypoints3D.at<float>(i,1), 0.0f); 
    // glVertex3f(cuePos.at<float>(0,0), cuePos.at<float>(0,1), 0.0f);
    // glEnd();
  }

  // Getting rid of the 3rd and 4th column which contributes nothing further. 
  cuePos = cuePos.colRange(0, 2);
  pocketPositions = pocketPositions.colRange(0,2);
  keypoints3D = keypoints3D.colRange(0,2);

  // if(first) {
  //   printMat(cuePos);
  //   printMat(pocketPositions);
  //   printMat(keypoints3D);
  // }

  for(int i = 0; i < maxPocketPos; i++) {
    // Drawing Pocket position
    glBegin(GL_POINTS);
    glColor3f(1,1,1); glVertex2f(pocketPositions.at<float>(i,0), pocketPositions.at<float>(i,1));
    glEnd();
  }

  glPointSize(15);
  float maxDotProduct = -100;
  int nearestKeypointId = 0;
  int bestPocketPos = -1;
  Mat sC, sF, sD;
  Mat allDotProducts;
  targetedCue = targetedCue%keypoints3D.rows;

    
  for(int i = 0; i < maxPocketPos; i++) {
    // Determining the line from cue to pocket:
    Mat finalDirection = pocketPositions.row(i) - keypoints3D.row(targetedCue);
    normalize(finalDirection, finalDirection);

    // Determining the line from cue to keypoint
    Mat coarseHitDirection = keypoints3D.row(targetedCue) - cuePos;
    normalize(coarseHitDirection, coarseHitDirection);
      
    // Determining the keypoint which lies closest to the ideal direction
    Mat dotProduct = coarseHitDirection*finalDirection.t();
    allDotProducts.push_back(dotProduct);
    if(dotProduct.at<float>(0,0) > maxDotProduct) {
      sF = finalDirection;
      sC = coarseHitDirection;
      sD = dotProduct;
      maxDotProduct = dotProduct.at<float>(0,0);
      nearestKeypointId = targetedCue;
      bestPocketPos = i;
    }
  } 

  // Mat sortOrder;
  // sortIdx(allDotProducts, sortOrder, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
  // // sort(allDotProducts, allDotProducts, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

  // if(first) {
  //   printMat(allDotProducts);
  //   printMat(sortOrder);
  // }

  // printf("maxDotProduct = %f \n", maxDotProduct);
  // printMat(sF);
  // printMat(sC);
  // printMat(sD);
  // Determining fine hit direction:
  Mat hitCuePos = keypoints3D.row(nearestKeypointId) - sF*2*2.858;
  glPushMatrix();
  glTranslatef(hitCuePos.at<float>(0,0), hitCuePos.at<float>(0,1), 0.0f);
  glColor4f(0.1, 0.1, 0.1, 0.5);
  glutSolidSphere(2.858, 100, 100);
  glPopMatrix();


  glLineWidth(6.0f);
  glBegin(GL_LINES);
  glColor3f(1,1,0); 
  glVertex3f(cuePos.at<float>(0,0), cuePos.at<float>(0,1), 0.0f);
  glVertex3f(hitCuePos.at<float>(0,0),hitCuePos.at<float>(0,1), 0.0f); 
  glVertex3f(keypoints3D.at<float>(nearestKeypointId,0),keypoints3D.at<float>(nearestKeypointId,1), 0.0f); 
  glVertex3f(pocketPositions.at<float>(bestPocketPos, 0), pocketPositions.at<float>(bestPocketPos, 1), 0.0f);
  glEnd();

  mode = 4;
  moveWhite = true;
  whiteCue = cuePos;
  prevWhiteCue = cuePos;
  finalWhiteCuePos = hitCuePos;
  targetCue = keypoints3D.row(nearestKeypointId);
  prevTargetCue = keypoints3D.row(nearestKeypointId);
  targetPocket = bestPocketPos;
 
  first = false;
  cvReleaseImage(&H);
  cvReleaseImage(&S);
  cvReleaseImage(&V);
  cvReleaseImage(&hsvImg);
  return;

}

void getPocketPositions(int x, int y) {

  Mat pt2d = (Mat_<float>(3,1) << x, y, 1.0 );
  Mat pt3d = convert2Dto3D(pt2d);
  pt3d = pt3d.t();
  pocketPositions.push_back(pt3d);
  countKnownPocketPos++;
  if(countKnownPocketPos == maxPocketPos) {
    FileStorage file("pocketPositions.xml", FileStorage::WRITE);
    file << "pocketPositions" << pocketPositions;
    file.release();
    mode = 1;
  }

}

void simShot() {
 
  if(targetPocket == -1) {
    mode = 1;
    return;
  }
  glEnable(GL_DEPTH_TEST);
  glPointSize(15);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  float w = imgPlaneWidth/focalLength*near, h = imgPlaneHeight/focalLength*near;

  glFrustum(-imageCenter[0]/imageWidth*w, (imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[1]/imageHeight*h,(imageHeight-imageCenter[1])/imageHeight*h, near, far); 

  //flip l,r and b,t to account for camera being upside down
  //glFrustum((imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[0]/imageWidth*w,  (imageHeight-imageCenter[1])/imageHeight*h, -imageCenter[1]/imageHeight*h, near, far); 

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glScalef(1,-1,-1);
  glMultMatrixf(matExt);

  glDisable(GL_TEXTURE_2D);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glColor4f(1,0,1, 0.5);

  for(int i = 0; i < maxPocketPos; i++) {
    // Drawing Pocket position
    glBegin(GL_POINTS);
    glColor3f(1,1,1); glVertex2f(pocketPositions.at<float>(i,0), pocketPositions.at<float>(i,1));
    glEnd();
  }

  // Need to set the speed based on how far away the target cue is from the target pocket
  Mat diff = pocketPositions.row(targetPocket) - targetCue;
  Mat sqDiff = diff*diff.t();
  maxCount = sqrt(sqDiff.at<int>(0,0))/3000;

  Mat currWhiteCue, currTargetCue;
  float ratio = (float)simCount/maxCount;

  if(moveWhite) {
    currWhiteCue = whiteCue*(1 - ratio) + targetCue*ratio;
    currTargetCue = targetCue;
  }
  else {
    currWhiteCue = prevWhiteCue;
    currTargetCue = targetCue*(1 - ratio) + pocketPositions.row(targetPocket)*ratio;
  } 
  
  for(int i = 0; i < maxPocketPos; i++) {
    // Drawing Pocket position
    glBegin(GL_POINTS);
    glColor3f(1,1,1); glVertex2f(pocketPositions.at<float>(i,0), pocketPositions.at<float>(i,1));
    glEnd();
  }

  // Drawing the white cue
  glPushMatrix();
  glTranslatef(currWhiteCue.at<float>(0,0),currWhiteCue.at<float>(0,1), 0.0f);
  glColor3f(1.0, 1.0, 1.0);
  glutSolidSphere(2.858, 100, 100);
  glPopMatrix();

  // Drawing the target cue
  glPushMatrix();
  glTranslatef(currTargetCue.at<float>(0,0),currTargetCue.at<float>(0,1), 0.0f);
  glColor3f(0.7, 0.7, 0.7);
  glutSolidSphere(2.858, 100, 100);
  glPopMatrix();

  simCount = ++simCount%maxCount;
  prevTargetCue = currTargetCue;
  prevWhiteCue = currWhiteCue;
  if(simCount == 0) 
    moveWhite = !moveWhite;

}


void drawShot() {

  if(targetPocket == -1) {
    mode = 1;
    return;
  }
  glEnable(GL_DEPTH_TEST);
  glPointSize(15);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  float w = imgPlaneWidth/focalLength*near, h = imgPlaneHeight/focalLength*near;

  glFrustum(-imageCenter[0]/imageWidth*w, (imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[1]/imageHeight*h,(imageHeight-imageCenter[1])/imageHeight*h, near, far); 

  //flip l,r and b,t to account for camera being upside down
  //glFrustum((imageWidth-imageCenter[0])/imageWidth*w, -imageCenter[0]/imageWidth*w,  (imageHeight-imageCenter[1])/imageHeight*h, -imageCenter[1]/imageHeight*h, near, far); 

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glScalef(1,-1,-1);
  glMultMatrixf(matExt);

  glDisable(GL_TEXTURE_2D);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glColor4f(1,0,1, 0.5);

  for(int i = 0; i < maxPocketPos; i++) {
    // Drawing Pocket position
    glBegin(GL_POINTS);
    glColor3f(1,1,1); glVertex2f(pocketPositions.at<float>(i,0), pocketPositions.at<float>(i,1));
    glEnd();
  }

  // Drawing the white cue
  glPushMatrix();
  glTranslatef(whiteCue.at<float>(0,0),whiteCue.at<float>(0,1), 0.0f);
  glColor3f(1.0, 1.0, 1.0);
  glutSolidSphere(2.858, 100, 100);
  glPopMatrix();

  // Drawing the final white cue position
  glPushMatrix();
  glTranslatef(finalWhiteCuePos.at<float>(0,0),finalWhiteCuePos.at<float>(0,1), 0.0f);
  glColor4f(0.1, 0.1, 0.1, 0.5);
  glutSolidSphere(2.858, 100, 100);
  glPopMatrix();

  // Drawing the target cue
  glPushMatrix();
  glTranslatef(targetCue.at<float>(0,0),targetCue.at<float>(0,1), 0.0f);
  glColor3f(0.7, 0.7, 0.7);
  glutSolidSphere(2.858, 100, 100);
  glPopMatrix();

  // Drawing lines
  glLineWidth(6.0f);
  glBegin(GL_LINES);
  glColor3f(1,1,0); 
  glVertex3f(whiteCue.at<float>(0,0), whiteCue.at<float>(0,1), 0.0f);
  glVertex3f(finalWhiteCuePos.at<float>(0,0),finalWhiteCuePos.at<float>(0,1), 0.0f); 
  glVertex3f(targetCue.at<float>(0,0),targetCue.at<float>(0,1), 0.0f); 
  glVertex3f(pocketPositions.at<float>(targetPocket, 0), pocketPositions.at<float>(targetPocket, 1), 0.0f);
  glEnd();

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

  if(mode == 0) 
    detectCalibrationPattern();

  if(mode == 1) 
    calculateShot();

  if(mode == 4 && tSim == true)
    simShot();

  if(mode == 4 && tSim == false)
    drawShot();

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
  case 'R': mode = 1; break;
  case 't': tSim = !tSim; break;
  case 'c': targetedCue++; mode = 1; break;
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


void mouseButton(int button, int state, int x, int y) {
  if( mode == 2) {
    switch(button) {
    case GLUT_LEFT_BUTTON:
      if(state == GLUT_UP) {
	getPocketPositions(x, y); break;
      }
    }
  }
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
  glutMouseFunc(mouseButton);
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
  invInt = cvCreateMat(3, 3, CV_32FC1);
  cvInvert(cam_mat, invInt);


  extrinsics = (CvMat*)cvLoad("extrinsics.xml", NULL, NULL, NULL);
  if(extrinsics == NULL) {
    printf("Extrinsics file not found. Place the circle grid pattern on the table \n");
    mode = 0;
  }
  else {
    invExt = cvCreateMat(4, 4, CV_32FC1);
    cvInvert(extrinsics, invExt);

    //    Check for pocket position matrix
    FileStorage file("pocketPositions.xml", FileStorage::READ);
    file["pocketPositions"] >> pocketPositions;
    file.release();
    // pocketPositions = (CvMat*)cvLoad("pocketPositions.xml", NULL, NULL, NULL);
    if(pocketPositions.rows == 0) {
      printf("Pocket positions matrix not found. Click on the 6 pocket positions in the image. \n");
      mode = 2;
    }
    else
      countKnownPocketPos = pocketPositions.rows;

    mInt = cam_mat;
    mExt = extrinsics;
    mInvInt = invInt;
    mInvExt = invExt;
    init2Dto3Dconversion();
  }

  glutMainLoop();

}
