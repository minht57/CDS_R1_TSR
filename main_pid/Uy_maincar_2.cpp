#include "api_kinect_cv.h"
// api_kinect_cv.h: manipulate openNI2, kinect, depthMap and object detection
#include "api_lane_detection.h"
// api_lane_detection.h: manipulate line detection, finding lane center and vanishing point
#include "api_i2c_pwm.h"
#include "multilane.h"
#include <iostream>
#include "Hal.h"
#include "LCDI2C.h"
#include <sys/stat.h>
#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/opencv.h>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cv.h>
#include <time.h>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include "openni2.h"
#include "../openni2/Singleton.h"
#include <unistd.h>
#include "../sign_detection/SignDetection.h"
#include <chrono>
#include "extractInfo.h"
#include "../ObjectRecognition/SignRecognition.h"
#include <thread>

using namespace std;
using namespace dlib;
using namespace framework;
using namespace signDetection;
using namespace SignRecognition;
using namespace openni;
using namespace EmbeddedFramework;
using namespace cv;
signDetection::SignDetection detection;

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
#define VIDEO_FRAME_WIDTH 640	
#define VIDEO_FRAME_HEIGHT 480
#define SW1_PIN	160
#define SW2_PIN	161
#define SW3_PIN	163
#define SW4_PIN	164
#define SENSOR	166

struct TrafficSign {
	string name;
	string svm_path;
	rgb_pixel color;
	TrafficSign(string name, string svm_path, rgb_pixel color) :
	name(name), svm_path(svm_path), color(color) {};
};

enum StateMachine{
	SM_START = 0,
	SM_STOP,
	SM_RUN,
	SM_PAUSE, 
	SM_NONE
};

enum TSR{
	TSR_NONE = 0,
	TSR_TURN_LEFT,
	TSR_TURN_RIGHT,
	
};

typedef struct {
	cv::Point x[6];
	float MinRatioRange;
	float MaxRatioRange;
	float AlphaLPF;
}CenterPoint_t;

typedef	struct
{
	cv::Rect roi;
	int car_offset;
	int yCalc;
	int point_offset;
	float MinRatioRange;
	float MaxRatioRange;
	float AlphaLPF;
	double angRatioL1;
	double angRatioL2;
	double angRatioR1;
	double angRatioR2;
	double throttleRatio;
	int mindl;
	int mindr;
	int TSR_OffsetLeft;
	int TSR_OffsetRight;
	int S_val;
	int S_max;
	int V_val;
	int V_max;
	bool is_show_cam;
	bool is_save_file;
}Parameters;

//============================================================GLOBAL VARIABLES=========================================================================//
int preSTT_Machine = SM_NONE, STT_Machine = SM_NONE;
bool isCalibDone = false;
double st = 0, et = 0, fps = 0, freq;
int frame_id = 0;
VideoWriter depth_videoWriter;  
VideoWriter color_videoWriter;
VideoWriter gray_videoWriter;
//string gray_filename = "gray.avi";
string color_filename = "color.avi";
string depth_filename = "depth.avi";

int dir = 0, throttle_val = 0;
int set_throttle_val = 0;
double theta = 0;

PCA9685 *pca9685 = new PCA9685();
LCDI2C *lcd = new LCDI2C();
FILE *thetaLogFile; // File creates log of signal send to pwm control

ushort l_th = 600, h_th = 2000;//previous: 600-2000
cv::Mat depthImg, colorImg, grayImage, disparity;
typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
std::vector<object_detector<image_scanner_type> > detectors;
std::vector<TrafficSign> signs;
std::vector<Mat> tmp_img;
std::vector<Rect> boxes;
std::vector<int> labels;
int TSR = TSR_NONE;
cv::Rect tsr_rect_res = cv::Rect(0,0,0,0); 

RNG rng(12345);
cv::Point preMidLane(0,0);
cv::Point pointMidLane;
cv::Point FilteredMidlanePoint;
CenterPoint_t CenterPoint; 
std::vector<cv::Point> Lp4_, Rp4_;
cv::Mat lane;
int x = 1;
int y = 68;
//=============================================================================================================================================//

//=====================================================SUBROUTINES PROTOTYPES==================================================================//
bool exists_files (const std::string& name);
double getTheta(Point car, Point dst);
void lane_detection(Parameters par,cv::Mat &colorImg, cv::Mat &depthImg, cv::Point &FilteredMidlanePoint);
void object_detection(cv::Mat &colorImg, cv::Mat &depthImg);
void tsr_depth_process(cv::Mat &colorImg, cv::Mat &depthImg, int &tsr__);
void tsr_process(cv::Mat &colorImg, cv::Mat &depthImg, int &tsr__, int S_val, int S_max, int V_val, int V_max);
void getImg(cv::Mat &color, cv::Mat &depth);
void SM_Start(void);
bool SM_Run(Parameters par);
void SM_Pause(void);
void SM_Stop(void);
//=============================================================================================================================================//

//===========================================================SUBROUTINES=======================================================================//
bool exists_files (const std::string& name) 
{
	struct stat buffer;   
	return (stat (name.c_str(), &buffer) == 0); 
}

double getTheta(cv::Point car, cv::Point dst) 
{
	if (dst.x == car.x) return 0;
	if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
	double pi = acos(-1.0);
	double dx = dst.x - car.x;
	double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
	if (dx < 0) return -atan(-dx / dy) * 180 / pi;
	return atan(dx / dy) * 180 / pi;
}

void SM_Start(void)
{

}

void SM_Stop(void)
{
	if (preSTT_Machine != STT_Machine)
	{
		//Status
		preSTT_Machine = STT_Machine;
		//Exec
		theta = 0;
		throttle_val = 0;
		api_set_FORWARD_control( pca9685,throttle_val);
		api_set_STEERING_control(pca9685,theta);
	}
}

void SM_Pause(void) 
{
	if (preSTT_Machine != STT_Machine)
	{
		//Status
		preSTT_Machine = STT_Machine;
		//Exec
		theta = 0;
		throttle_val = 0;
		api_set_FORWARD_control( pca9685,throttle_val);
		api_set_STEERING_control(pca9685,theta);
	}
}

bool SM_Run(Parameters par)
{
	cv::Point carPosition = cv::Point(par.roi.width/2 + par.car_offset, par.roi.height-1);

	//Check PCA9685 driver
	if (pca9685->error < 0)
	{
		cout<< endl<< "Error: PWM driver"<< endl<< flush;
		return false;
	}

	std::thread thread_0;
	std::thread thread_1;
	std::thread thread_2;
	std::thread thread_3;

	//=========================================================MULTITHREADS===========================================================//
	thread_0 = std::thread(getImg,std::ref(colorImg),std::ref(depthImg));
    thread_1 = std::thread(lane_detection,par,std::ref(colorImg),std::ref(depthImg),std::ref(FilteredMidlanePoint));
    thread_2 = std::thread(object_detection,std::ref(colorImg),std::ref(depthImg));
    thread_3 = std::thread(tsr_depth_process,std::ref(colorImg),std::ref(depthImg),std::ref(TSR));
	//thread_3 = std::thread(tsr_process,std::ref(colorImg),std::ref(depthImg),std::ref(TSR),par.S_val,par.S_max,par.V_val,par.V_max);
	thread_0.join();
    thread_1.join();
    thread_2.join();
    thread_3.join();
	//================================================================================================================================//
	//========================================================CONTROL PHASE===========================================================// 
	double angDiff = getTheta(carPosition, FilteredMidlanePoint);
	
	if (angDiff < -40)
		theta = (angDiff*par.angRatioL1);
	else if (angDiff < 0)
		theta = (angDiff*par.angRatioL2);
	else if (angDiff > 40)
	    theta = angDiff*par.angRatioR1;
	else theta = angDiff*par.angRatioR2;

	if (frame_id == 1)
		throttle_val = set_throttle_val*0.6;
	else if (frame_id == 2)
		throttle_val = set_throttle_val*0.8;
	else throttle_val = set_throttle_val;
	
	//theta = (-90.00);
	api_set_STEERING_control(pca9685,theta);
	
	int pwm2 =  api_set_FORWARD_control( pca9685,throttle_val);
	//================================================================================================================================//


	if(par.is_save_file)
	{
		Mat dst_s;
		if (!lane.empty())
		{

		    cvtColor(lane, dst_s, CV_GRAY2BGR); 
		    dst_s.copyTo(colorImg(Rect(0, 0, dst_s.cols, dst_s.rows)));
		}
		circle(colorImg, pointMidLane, 2, Scalar(255,0,0), 5);
		circle(colorImg, FilteredMidlanePoint, 2, Scalar(0,255,0), 5);
		circle(colorImg, carPosition, 2, Scalar(0,0,255), 5);
		if (!colorImg.empty())
			color_videoWriter.write(colorImg);
	}
	//========================================================CALCULATE FPS===========================================================//
	et = getTickCount();
	fps = 1.0 / ((et-st)/freq);
	cout << "frame_id " << frame_id << "FPS:     " << fps << endl;
	//================================================================================================================================//
	return true;
}

cv::Point FilterMidlanePoint(cv::Point pointMidLane, CenterPoint_t *CenterPoint)
{	
	cv::Point OffsetDistance;
	cv::Point PredictedPoint;
	cv::Point TempPoint;
	cv::Point FilteredPoint(0,0);
	static cv::Point prePointMidLane;

	FilteredPoint = CenterPoint->x[5];

	OffsetDistance = (CenterPoint->x[5] - CenterPoint->x[0])/5;
	PredictedPoint = FilteredPoint + OffsetDistance;

	//if ((pointMidLane.x > CenterPoint->MinRatioRange*PredictedPoint.x) && (pointMidLane.x < CenterPoint->MaxRatioRange*PredictedPoint.x))
	if ((pointMidLane.x > -(int)(CenterPoint->MinRatioRange) + PredictedPoint.x) && (pointMidLane.x < (int)(CenterPoint->MaxRatioRange) + PredictedPoint.x) && (pointMidLane.y != 0))
	{
		TempPoint = pointMidLane;
	}
	else TempPoint = PredictedPoint;

	FilteredPoint = FilteredPoint + CenterPoint->AlphaLPF*(TempPoint - FilteredPoint);

	for(int i = 0; i < 5; i++)
	{
		CenterPoint->x[i] = CenterPoint->x[i+1];
	}
	CenterPoint->x[5] = FilteredPoint;

	prePointMidLane = pointMidLane;
	return FilteredPoint;
}

void lane_detection(Parameters par,cv::Mat &colorImg, cv::Mat &depthImg, cv::Point &FilteredMidlanePoint)
{	
	double st1 = 0, et1 = 0, fps1 = 0;
	st1 = getTickCount();
	cv::Mat colorImg_clone = colorImg.clone();
	cv::Mat depthImg_clone = depthImg.clone();	
	if((!colorImg_clone.empty()) && (!depthImg_clone.empty()))
    {
		cv::Mat im_raw = colorImg(par.roi).clone();
		cv::Mat grayImage;
		cvtColor(im_raw, grayImage, CV_BGR2GRAY);

		if((!im_raw.empty()) && (!grayImage.empty()))
		{	
			//cout << "lane detection" << endl;
			cv::Mat dst = keepLanes_new(grayImage);
			imshow("dst", dst);
			pointMidLane = twoLaneMostLanes_th(dst.size(), dst, lane, par.yCalc, par.point_offset, par.mindl, par.mindr, Lp4_, Rp4_);
			imshow("Lane", lane);
			FilteredMidlanePoint = FilterMidlanePoint(pointMidLane, &CenterPoint);
			//cout << "Filtered Midlane Point: " << FilteredMidlanePoint.x << " - " << FilteredMidlanePoint.y << endl;
		}
	}
	else
		cout << "nothing 1" << endl;
	et1 = getTickCount();
	fps1 = 1.0 / ((et1-st1)/freq);
	//cout << "FPS lane:     " << fps1 << endl;
}

void object_detection(cv::Mat &colorImg, cv::Mat &depthImg)
{
	double st2 = 0, et2 = 0, fps2 = 0;
	st2 = getTickCount();

	cv::Mat colorImg_clone = colorImg.clone();
	cv::Mat depthImg_clone = depthImg.clone();
	if((!colorImg_clone.empty()) && (!depthImg_clone.empty()));
    	//cout << "object detection" << endl;
	else
		cout << "nothing 2" << endl;
	et2 = getTickCount();
	fps2 = 1.0 / ((et2-st2)/freq);
	//cout << "FPS object:     " << fps2 << endl;	
}

void tsr_depth_process(cv::Mat &colorImg, cv::Mat &depthImg, int &tsr__)
{
	double st3 = 0, et3 = 0, fps3 = 0;
	st3 = getTickCount();

	cv::Mat colorImg_clone = colorImg.clone();
	cv::Mat depthImg_clone = depthImg.clone();	
	if((!colorImg_clone.empty()) && (!depthImg_clone.empty()))
    {
		cv::Mat TSRDepthImg = depthImg_clone(cv::Rect(0,200,640,100)).clone();
		//cv::imshow("depth",TSRDepthImg);
		static int TSR_DetectedFrameCnt = 0;
		static int TSR_NumFrameCnt = 0;
	
		std::vector<Rect> boxes;
    	std::vector<int> labels;
		cv::Mat DetectionResult;
		detection.objectLabeling(boxes, labels, TSRDepthImg, colorImg, DetectionResult, l_th, h_th, 1000, 4000, 36, 200, 1.2);//1000-4000-36-200-1.2
	
		if(DetectionResult.empty())
			cout<<"No traffic sign"<<endl;	
		else
		{  
			//cout<<"Traffic sign detected"<<endl;
			//cv::imshow("Detection Result",DetectionResult);
			Mat src;
			src = DetectionResult.clone();
			resize(src,src,Size(80,80),INTER_LANCZOS4);// vi dieu
			//cv::imshow("src",src);				
			cv_image<bgr_pixel> images_HOG(src);
			std::vector<rect_detection> rects;

			evaluate_detectors(detectors, images_HOG, rects, 0.9);
			cout << "Rects size: " << rects.size() << endl;
			if(rects.size() > 0)
			{
				if 	(1)
				{
					if (rects[0].weight_index == 6) //&& (TSR_NumFrameCnt <= 5))
					{		                                           		                                            
						//TSR_DetectedFrameCnt++;
						//if (TSR_DetectedFrameCnt >= 3)
						//{
							cout << "                                   TURN LEFT: " << rects[0].detection_confidence << endl;
							tsr__ = TSR_TURN_LEFT;
						//	TSR_NumFrameCnt = 0;
						//	TSR_DetectedFrameCnt = 0;
						//}
					}
					else if (rects[0].weight_index == 7) //&& (TSR_NumFrameCnt <= 5))
					{
						//TSR_DetectedFrameCnt++;
						//if (TSR_DetectedFrameCnt >= 3)
						//{
							cout << "                                   TURN RIGHT: " << rects[0].detection_confidence << endl;
							tsr__ = TSR_TURN_RIGHT;
						//	TSR_NumFrameCnt = 0;
						//	TSR_DetectedFrameCnt = 0;
						//}
					}
					else 
					{
						tsr__ = TSR_NONE;
					}
				}
			}
    	}
	}
	else
		cout << "nothing 3" << endl;
	et3 = getTickCount();
	fps3 = 1.0 / ((et3-st3)/freq);
	cout << "FPS TSR:     " << fps3 << endl;	
}

void tsr_process(cv::Mat &colorImg, cv::Mat &depthImg, int &tsr__, int S_val, int S_max, int V_val, int V_max)
{
	double st4 = 0, et4 = 0, fps4 = 0;
	st4 = getTickCount();
	static int TSR_DetectedFrameCnt = 0;
	static int TSR_NumFrameCnt = 0;
	cv::Mat colorImg_clone = colorImg.clone();
	cv::Mat depthImg_clone = depthImg.clone();	
	if((!colorImg_clone.empty()) && (!depthImg_clone.empty()))
    {
	cv::Mat TSRColorImg = colorImg(cv::Rect(0,180,640,100)).clone();
    Mat img_raw = TSRColorImg;
	Mat img_HSV, maskr, maskr1, maskr2, maskb, maskk, edged;
    std::vector<char> save_template;
    //imshow("img_raw", colorImg);
    cvtColor(TSRColorImg, img_HSV, COLOR_BGR2HSV);
    inRange(img_HSV, Scalar(100, S_val, V_val), Scalar(150, S_max, V_max), maskb);
    edged = maskb;
	//imshow("edged", edged);
    Mat kernel = (Mat_<float>(3,3) <<
							1,  1, 1,
							1, -8, 1,
            				1,  1, 1); // an approximation of second derivative, a quite strong kernel 

    Mat imgLaplacian;

    Mat sharp = edged; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    edged.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<Vec4i> hierarchy;
    findContours(imgLaplacian, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    std::vector<Rect> boundRect( contours.size() );
    Mat drawing = Mat::zeros(imgLaplacian.size(), CV_8UC1);
    for(int idx = 0; idx < contours.size(); idx++)
    {
        boundRect[idx] = boundingRect( Mat(contours[idx]) );
        if ((boundRect[idx].width*boundRect[idx].height > 2000) && (boundRect[idx].width*boundRect[idx].height < 50000))
        {
	        if((boundRect[idx].tl().x != 0)&&(boundRect[idx].tl().y != 0) && (boundRect[idx].br().x != img_raw.cols) && (boundRect[idx].br().y != img_raw.rows))
	        {
	            if ( ( (float)boundRect[idx].width/boundRect[idx].height > 0.5) && ( (float)boundRect[idx].width/boundRect[idx].height < 1.3 ) )
	            {
	                drawContours( imgLaplacian, contours, idx, 255, CV_FILLED, 8, hierarchy );
	                Mat crp_drw = imgLaplacian(boundRect[idx]).clone();
	                Mat src = img_raw(boundRect[idx]).clone();
	                int linewidth = 2;
	                Scalar value;
	                value = Scalar(0, 0, 0);
	                Mat roi_d;
	                copyMakeBorder( crp_drw, roi_d, linewidth, linewidth, linewidth, linewidth, BORDER_CONSTANT, value );
	                Mat roi_s;
	                copyMakeBorder( src, roi_s, linewidth, linewidth, linewidth, linewidth, BORDER_CONSTANT, value );

	                Mat dist;
	                distanceTransform(roi_d, dist, CV_DIST_L2, 3);
	                normalize(dist, dist, 0, 1., NORM_MINMAX);

	                Mat dist_bw;
	                threshold(dist, dist_bw, .4, 1., CV_THRESH_BINARY);

	                Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
	                dilate(dist_bw, dist_bw, kernel1);

	                Mat dist_8u;
	                dist_bw.convertTo(dist_8u, CV_8U);

	                std::vector<std::vector<cv::Point> > contoursm;
	                std::vector<Vec4i> hierarchym;

	                findContours(dist_8u, contoursm, hierarchym, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	                Mat markers = Mat::zeros(dist.size(), CV_32SC1);

	                for(int isx = 0; isx < contoursm.size(); isx++) {
	                    drawContours(markers, contoursm, static_cast<int>(isx), Scalar::all(static_cast<int>(isx)+1), -1);	
	                }

	                circle(markers, Point(1,1), 1, CV_RGB(255,255,255), -1);

	                watershed(roi_s, markers);


	                Mat dst = Mat::zeros(markers.size(), CV_8UC1);

	                for (int i = 0; i < markers.rows; i++)
	                {
	                    for (int j = 0; j < markers.cols; j++)
	                    {
	                        int index = markers.at<int>(i,j);
	                        if(index==1)
	                            dst.at<uchar>(i,j) = 255;
	                        else dst.at<uchar>(i,j) = 0;
	                    }
	                }

	                dst = dst & roi_d;

	                std::vector<std::vector<cv::Point> > contoursw;
	                std::vector<Vec4i> hierarchyw;

	                findContours(dst, contoursw, hierarchyw, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	                std::vector<Rect> boundRectw( contoursw.size() );

	                if(contoursw.size() > 0)
	                {
	                    for(int iw = 0; iw < contoursw.size(); iw++) 
	                    {
	                        if (hierarchyw[iw][2]< 0)
	                        {        						
	                            boundRectw[iw] = boundingRect( Mat(contoursw[iw]) );
	                            if ( ( (boundRectw[iw].width*boundRectw[iw].height) > 2000) && ( (boundRectw[iw].width*boundRectw[iw].height)  < 50000))
	                            {        							
	                                if ( ( (float)boundRectw[iw].width/boundRectw[iw].height < 1.3) && ( (float)boundRectw[iw].width/boundRectw[iw].height > 0.5) )
	                                {
	        							Rect dstbound (boundRectw[iw].tl().x + boundRect[idx].tl().x - 2, boundRectw[iw].tl().y + boundRect[idx].tl().y - 2,boundRectw[iw].width, boundRectw[iw].height);
	        							Mat image_roi = TSRColorImg(dstbound);
	        							resize(image_roi,image_roi,Size(80,80),INTER_LANCZOS4);
	        							//imshow("result",image_roi);
	        							cv_image<bgr_pixel> images_HOG(image_roi);
	        							std::vector<rect_detection> rects;
	        							evaluate_detectors(detectors, images_HOG, rects, 0.9);
										cout << rects.size() << endl;
       									if(rects.size() > 0)
	        							{
	        								cout << rects[0].weight_index << endl;
	        								if 	(1)
	        								{
	                                            if ((rects[0].weight_index == 6) && (TSR_NumFrameCnt <= 5))
		                                        {
		                                           	cout << "area: " << boundRectw[iw].width <<" " << boundRectw[iw].height << endl;                              
			                                        TSR_DetectedFrameCnt++;
			                                        if (TSR_DetectedFrameCnt >= 1)
			                                        {
			                                            cout << "TURN LEFT:         " << rects[0].detection_confidence << endl;
				                                        tsr__ = TSR_TURN_LEFT;
				                                        TSR_NumFrameCnt = 0;
				                                        TSR_DetectedFrameCnt = 0;
				                                        tsr_rect_res = dstbound;
			                                        }
		                                        }
		                                        else if ((rects[0].weight_index == 7) && (TSR_NumFrameCnt <= 5))
		                                        {
		                                            
			                                        TSR_DetectedFrameCnt++;
			                                        if (TSR_DetectedFrameCnt >= 1)
			                                        {
			                                            cout << "TURN RIGHT:        " << rects[0].detection_confidence << endl;
				                                        tsr__ = TSR_TURN_RIGHT;
				                                        TSR_NumFrameCnt = 0;
				                                        TSR_DetectedFrameCnt = 0;
				                                        tsr_rect_res = dstbound;
			                                        }
		                                        }
		                            	        else 
		                                	    {
			                            	        tsr__ = TSR_NONE;
		                                        }
	        								}
	        							}
	        						}
	        					}
	        				}
	        			}
	        		}
	        	}
	        }
       	}
	}
	}
	else
		cout << "nothing 4" << endl;
	et4 = getTickCount();
	fps4 = 1.0 / ((et4-st4)/freq);
	cout << "FPS TSR:     " << fps4 << endl;
}

void getImg(cv::Mat &color, cv::Mat &depth)
{
	OpenNI2::Instance()->getData_new(color, depth);
}
//=============================================================================================================================================//

//============================================================MAIN PROGRAM=====================================================================//
int main( int argc, char* argv[] ) 
{
	//=============================================LOAD PARAMETERS=================================================//
	Parameters par;
	string line;
	std::vector<float> value;
	ifstream myfile ("parameters.txt");
	if (myfile.is_open())
	{
		while ( getline (myfile,line) )
		{
			value.push_back(std::stof(line));
		}
		myfile.close();
	}
	else 
	{
		cout << "Unable to open file" << endl;
		return 0;
	}

	if(value.size() == 25)
	{
		par.roi = cv::Rect(value[0], value[1], value[2], value[3]);
		par.car_offset = value[4];
		par.yCalc = value[5];
		par.point_offset = value[6];
		par.MinRatioRange = value[7];
		par.MaxRatioRange = value[8];
		par.AlphaLPF = value[9];
		par.angRatioL1 = value[10];
		par.angRatioL2 = value[11];
		par.angRatioR1 = value[12];
		par.angRatioR2 = value[13];
		par.throttleRatio = value[14];
		par.mindl = value[15];
		par.mindr = value[16];
		par.TSR_OffsetLeft = value[17];
		par.TSR_OffsetRight = value[18];
		par.S_val = value[19];
		par.S_max = value[20];
		par.V_val = value[21];
		par.V_max = value[22];
		par.is_show_cam = value[23];
		par.is_save_file = value[24];
		
		cout << par.roi << endl;
		cout << par.car_offset << endl;
		cout << par.yCalc << endl;
		cout << par.point_offset << endl;
		cout << par.MinRatioRange << endl;
		cout << par.MaxRatioRange << endl;
		cout << par.AlphaLPF << endl;
		cout << par.angRatioL1 << endl;
		cout << par.angRatioL2 << endl;
		cout << par.angRatioR1 << endl;
		cout << par.angRatioR2 << endl;
		cout << par.throttleRatio << endl;
		cout << par.mindl << endl;
		cout << par.mindr << endl;
		cout << par.TSR_OffsetLeft << endl;
		cout << par.S_val << endl;
		cout << par.S_max << endl;
		cout << par.V_val << endl;
		cout << par.V_max << endl;
		cout << par.is_show_cam << endl;
		cout << par.is_save_file << endl;
		cout << endl << "Loaded all parameters" << endl << endl;
	}
	else 
	{
		cout << "Missing Parameters" << endl;
		return 0;
	}
	//================================================================================================================//

	//=============================================LOAD SVM DETECTORS=================================================//	
	cout << "Loading SVM detectors..." << endl;

	signs.push_back(TrafficSign("maximum-speed", "svm_detectors/maximum-speed.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("minimum-speed", "svm_detectors/minimum-speed.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("minimum-speed-end", "svm_detectors/minimum-speed-end.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("national-speed-limit", "svm_detectors/national-speed-limit.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("no-entry", "svm_detectors/no-entry.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("no-left-turn", "svm_detectors/no-left-turn.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("no-right-turn", "svm_detectors/no-right-turn.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("stop-give-way", "svm_detectors/stop-give-way.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("turn-left-ahead", "svm_detectors/turn-left-ahead.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("turn-right-ahead", "svm_detectors/turn-right-ahead.svm", rgb_pixel(0,0,255)));

	for (int i = 0; i < signs.size(); i++) {
		object_detector<image_scanner_type> detector;
		deserialize(signs[i].svm_path) >> detector;
		detectors.push_back(detector);
	}
	//==============================================================================================================//

	//=============================================PERIPHERAL INIT==================================================//
	GPIO *gpio = new GPIO();
	I2C *i2c_device = new I2C();
	
	int sw1_stat = 1;
	int sw2_stat = 1;
	int sw3_stat = 1;
	int sw4_stat = 1;
	int sensor = 0;
	
	//Setup input
	gpio->gpioExport(SW1_PIN);
	gpio->gpioExport(SW2_PIN);
	gpio->gpioExport(SW3_PIN);
	gpio->gpioExport(SW4_PIN);
	gpio->gpioExport(SENSOR);
	gpio->gpioSetDirection(SW1_PIN, INPUT);
	gpio->gpioSetDirection(SW2_PIN, INPUT);
	gpio->gpioSetDirection(SW3_PIN, INPUT);
	gpio->gpioSetDirection(SW4_PIN, INPUT);
	gpio->gpioSetDirection(SENSOR, INPUT);
	usleep(10000);

	//Init LCD
	i2c_device->m_i2c_bus = 2;
	
	if (!i2c_device->HALOpen()) {
		printf("Cannot open I2C peripheral\n");
		exit(-1);
	} else printf("I2C peripheral is opened\n");
	
	unsigned char data;
	if (!i2c_device->HALRead(0x38, 0xFF, 0, &data, "")) {
		printf("LCD is not found!\n");
		exit(-1);
	} else printf ("LCD is connected\n");
	usleep(10000);
	lcd->LCDInit(i2c_device, 0x38, 20, 4);
	lcd->LCDBacklightOn();
	lcd->LCDCursorOff();
	lcd->LCDSetCursor(3,0);
	lcd->LCDPrintStr("DRIVERLESS CAR");
	lcd->LCDSetCursor(5,1);
	lcd->LCDPrintStr("2017-2018");

	//Init OpenNI2
	OpenNI2::Instance() -> init();

	//Init log files
	int i_name = 0;
	while(exists_files(color_filename))
	{
		color_filename = "color" + to_string(i_name++) + ".avi";
	}
	int i_name_depth = 0;
	while(exists_files(depth_filename))
	{
		depth_filename = "depth" + to_string(i_name_depth++) + ".avi";
	}
	int codec = CV_FOURCC('M','J','P', 'G');
	int video_frame_width = VIDEO_FRAME_WIDTH;
	int video_frame_height = VIDEO_FRAME_HEIGHT;
	Size output_size(video_frame_width, video_frame_height);
	if(par.is_save_file) {
		// gray_videoWriter.open(gray_filename, codec, 8, output_size, false);
		color_videoWriter.open(color_filename, codec, 8, output_size, true);
		//depth_videoWriter.open(depth_filename, CV_FOURCC('M','J','P', 'G'), 8, output_size, true);
		thetaLogFile = fopen("thetaLog.txt", "w");
	}

	//Init PCA9685 driver
	api_pwm_pca9685_init( pca9685 );
	if (pca9685->error >= 0)
	//api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
	api_set_FORWARD_control( pca9685,throttle_val);

	//Init MSAC vanishing point library
	MSAC msac;
	cv::Rect roi1 = cv::Rect(0, VIDEO_FRAME_HEIGHT*3/4,VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT/4);
	api_vanishing_point_init( msac );

	//Init direction and ESC speed 
	throttle_val = 0;
	theta = 0;
	if(argc == 2 ) set_throttle_val = atoi(argv[1]);
	fprintf(stderr, "Initial throttle: %d\n", set_throttle_val);
	int frame_width = VIDEO_FRAME_WIDTH;
	int frame_height = VIDEO_FRAME_HEIGHT;
	lcd->LCDSetCursor(4,3);
	lcd->LCDPrintStr("WAITING ....");

	char key = 0;
	freq = getTickFrequency();
	CenterPoint.x[0] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[1] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[2] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[3] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[4] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc); 
	CenterPoint.x[5] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.MinRatioRange = par.MinRatioRange; //150;
	CenterPoint.MaxRatioRange = par.MaxRatioRange; //150;
	CenterPoint.AlphaLPF = par.AlphaLPF; //0.9;
	//==============================================================================================================//

	while (true)
	{
		Point center_point(0,0);
	 	st = getTickCount();
	 	key = getkey();
		unsigned int bt_status = 0;

		//Buttons Order - X(BT4) X(BT3) D(BT2) D(BT1)
		//STOP
		gpio->gpioGetValue(SW1_PIN, &bt_status);
		if (!bt_status) 
		{
			if (bt_status != sw1_stat) 
			{
				//Status
				sw1_stat = bt_status;
				preSTT_Machine = STT_Machine;
				STT_Machine = SM_STOP;
				fprintf(stderr, "End process.\n");
				//Exec
				lcd->LCDClear();
				lcd->LCDSetCursor(7,0);
				lcd->LCDPrintStr("STOP");
				break;
			}
		} else sw1_stat = bt_status;
		//PAUSE
		gpio->gpioGetValue(SW2_PIN, &bt_status);
		if (!bt_status) 
		{
			if (bt_status != sw2_stat) 
			{
				//Status
				sw2_stat = bt_status;
				preSTT_Machine = STT_Machine;
				if(STT_Machine == SM_START)
				{
					if(preSTT_Machine == SM_PAUSE)
					{
						SM_Start();
					}
					else
						SM_Start();
				}
				else if(STT_Machine == SM_RUN)
				{
					isCalibDone = false;
					STT_Machine = SM_PAUSE;
					fprintf(stderr, "Pause.\n");
					//Exec
					lcd->LCDClear();
					lcd->LCDSetCursor(7,0);
					lcd->LCDPrintStr("PAUSE");
				}
			}
		} 
		else sw2_stat = bt_status;
		//RUN
		gpio->gpioGetValue(SW3_PIN, &bt_status);
		if (!bt_status) 
		{
			if (bt_status != sw3_stat) 
			{
				//Status
				sw3_stat = bt_status;
				preSTT_Machine = STT_Machine;
				if(STT_Machine == SM_START)
				{
					if(preSTT_Machine == SM_PAUSE)
					{
						SM_Start();
					}
					else
						SM_Start();
				}
				else if(STT_Machine == SM_PAUSE)
				{
					SM_Start();
					STT_Machine = SM_START;
					lcd->LCDClear();
					lcd->LCDSetCursor(6,0);
					lcd->LCDPrintStr("RESTART");
				}
			}
		} 
		else sw3_stat = bt_status;

		static int flag_wait = 0;
		gpio->gpioGetValue(SW4_PIN, &bt_status);
		if (!bt_status) 
		{
			if (bt_status != sw4_stat) 
			{
				//Status
				sw4_stat = bt_status;
				preSTT_Machine = STT_Machine;

				if(STT_Machine == SM_START)
				{
					flag_wait = 1;
					// STT_Machine = SM_RUN;
					fprintf(stderr, "Wait Run.\n");
					//Exec
					lcd->LCDClear();
					lcd->LCDSetCursor(5,0);
					lcd->LCDPrintStr("WAIT RUN");
				}
				else
				{
					STT_Machine = SM_START;
					fprintf(stderr, "Start.\n");
					//Exec
					throttle_val = set_throttle_val;
					lcd->LCDClear();
					lcd->LCDSetCursor(7,0);
					lcd->LCDPrintStr("START");
				}
			}
		} 
		else sw4_stat = bt_status;
	
		if(flag_wait)
		{
			gpio->gpioGetValue(SENSOR, &bt_status);
			if (bt_status) 
			{
				    //Status
				// sensor = bt_status;
				preSTT_Machine = STT_Machine;

				if(STT_Machine == SM_START)
				{
					STT_Machine = SM_RUN;
					fprintf(stderr, "Run.\n");
					    //Exec
					lcd->LCDClear();
					lcd->LCDSetCursor(7,0);
					lcd->LCDPrintStr("RUNNING");
					flag_wait = 0;
				}
			}
		}

		switch(key) 
		{
			case 's':
				//Status
			preSTT_Machine = STT_Machine;
			STT_Machine = SM_START;
			fprintf(stderr, "Start.\n");
				//Exec
			throttle_val = set_throttle_val;
			lcd->LCDClear();
			lcd->LCDSetCursor(7,0);
			lcd->LCDPrintStr("START");
			break;
			case 'f':
				//Status
			preSTT_Machine = STT_Machine;
			STT_Machine = SM_STOP;
				//Exec
			lcd->LCDClear();
			lcd->LCDSetCursor(7,0);
			lcd->LCDPrintStr("STOP");
			fprintf(stderr, "End process.\n");
			theta = 0;
			throttle_val = 0;
			api_set_FORWARD_control(pca9685,throttle_val);
			break;
			case 'r':
				//Status
			preSTT_Machine = STT_Machine;
			STT_Machine = SM_RUN;
			fprintf(stderr, "Run.\n");
				//Exec
			lcd->LCDClear();
			lcd->LCDSetCursor(7,0);
			lcd->LCDPrintStr("RUNNING");
			break;
			case 'p':
				//Status
			preSTT_Machine = STT_Machine;
			STT_Machine = SM_PAUSE;
			fprintf(stderr, "Pause.\n");
				//Exec
			lcd->LCDClear();
			lcd->LCDSetCursor(7,0);
			lcd->LCDPrintStr("PAUSE");
			theta = 0;
			throttle_val = 0;
			api_set_FORWARD_control(pca9685,throttle_val);
			break;
		}

		//================================================STATE MACHINE=========================================================//
		switch(STT_Machine)
		{
			case SM_START:
			if(preSTT_Machine == SM_PAUSE)
			{
				SM_Start();
			}
			else
				SM_Start();
			break;
			case SM_STOP:
			SM_Stop();
			if(par.is_save_file)
			{
				//gray_videoWriter.release();
				color_videoWriter.release();
				//depth_videoWriter.release();
				fclose(thetaLogFile);
			}
			return 0;
			break;
			case SM_RUN:
			frame_id++;
			SM_Run(par);
			break;
			case SM_PAUSE:
			SM_Pause();
			CenterPoint.x[0] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[1] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[2] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[3] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[4] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc); 
			CenterPoint.x[5] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			frame_id = 0;
			break;
			default:
			break;
		}
		//==============================================================================================================//
		if(par.is_show_cam)
			waitKey(1);
	}

	//Video Release
	if(par.is_save_file)
	{
		//gray_videoWriter.release();
		color_videoWriter.release();
		//depth_videoWriter.release();
		//fclose(thetaLogFile);
	}
	return 0;
}
