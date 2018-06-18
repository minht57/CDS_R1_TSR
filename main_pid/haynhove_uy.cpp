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
#include <iomanip>
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

#define FILTER_COLOR 0

struct TrafficSign 
{
	string name;
	string svm_path;
	rgb_pixel color;
	TrafficSign(string name, string svm_path, rgb_pixel color) :
	name(name), svm_path(svm_path), color(color) {};
};

enum StateMachine
{
	SM_START = 0,
	SM_STOP,
	SM_RUN,
	SM_PAUSE, 
	SM_NONE
};

enum TSR
{
	TSR_NONE = 0,
	TSR_TURN_LEFT,
	TSR_TURN_RIGHT,
	TSR_STOP,
};

typedef struct 
{
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
	double test_delta_x;
	double Kp;
	double Ki;
	double Kd;
}Parameters;

//============================================================GLOBAL VARIABLES=============================//
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
double delta_x = 0;
double pre_delta_x = 0; 
double u_control = 0;


PCA9685 *pca9685 = new PCA9685();
LCDI2C *lcd = new LCDI2C();
FILE *thetaLogFile; // File creates log of signal send to pwm control

ushort l_th_test = 2000, h_th_test = 3000;
ushort l_th = 1000, h_th = 5000;//previous: 600-2000
ushort l_th_obstacle = 1900, h_th_obstacle = 2200;
cv::Mat depthImg, colorImg, grayImage, disparity;
typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
std::vector<object_detector<image_scanner_type> > detectors;
std::vector<TrafficSign> signs;
std::vector<Mat> tmp_img;
std::vector<Rect> boxes;
std::vector<int> labels;
int tsr_stt = TSR_NONE;
int tsr_1_stt = TSR_NONE;
int tsr_2_stt = TSR_NONE;
bool stp = false;
cv::Rect tsr_rect_res = cv::Rect(0,0,0,0); 
int turn_count = 0;
RNG rng(12345);
int lane_pos = 0;
int angle_lane = 0;
cv::Point preMidLane(0,0);
cv::Point pointMidLane;
cv::Point FilteredMidlanePoint;
cv::Point carPosition = cv::Point(320, 120);
CenterPoint_t CenterPoint; 
std::vector<cv::Point> Lp4_, Rp4_;
cv::Mat lane;
cv::Mat bn_dh;
int x = 1;
int y = 68;
int img_cnt = 0;
//=========================================================================================================//

//=====================================================SUBROUTINES PROTOTYPES==============================//
bool exists_files (const std::string& name);
double getTheta(Point car, Point dst);
void lane_detection(Parameters par,cv::Mat &colorImg, cv::Point &pointMidLane, cv::Point &FilteredMidlanePoint);
void object_detection(cv::Mat &color, cv::Mat &depth);
void tsr_depth_process(cv::Mat &colorImg, cv::Mat &depthImg, int &tsr__);
void tsr_process(cv::Mat &colorImg, int &tsr__, int S_val, int S_max, int V_val, int V_max);
void new_depth_process(cv::Mat &color, cv::Mat &depth, int &tsr_rs);
void getImg(cv::Mat &color, cv::Mat &depth);
void SM_Start(void);
bool SM_Run(Parameters par);
void SM_Pause(void);
void SM_Stop(void);
void setThrottle(int throttle);
void setSteering(int steering);
//=========================================================================================================//

//===========================================================SUBROUTINES===================================//
// CHECK EXIST FILE =======================================================================================//
bool exists_files (const std::string& name) 
{
	struct stat buffer;   
	return (stat (name.c_str(), &buffer) == 0); 
}
// GET THETA ==============================================================================================//
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
// START STATE ============================================================================================//
void SM_Start(void)
{
	// static double tmp = -90;
	// api_set_STEERING_control(pca9685, tmp);
	// usleep(1000000);
	// tmp += 10;
	// if(tmp > 90)
	// 	tmp = -90;
	// cout << tmp << endl << endl;

	// Mat tmp;
	// OpenNI2::Instance()->getDepthMap(tmp);
	// double min, max;
	// minMaxLoc(tmp, &min, &max);
	// float scaleFactor = 256.0/max;
	// cout << "min\t" << scaleFactor << "\tmax\t" << max << endl; 
    // tmp.convertTo( tmp, CV_8UC1, scaleFactor );
	// imshow("x", tmp);

	// Mat tmp2;
	// OpenNI2::Instance()->getDisparityMap(tmp2);
	// imshow("y", tmp2);
}
// STOP STATE =============================================================================================//
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
		OpenNI2::Instance()->release();
	}
}
// PAUSE STATE ============================================================================================//
void SM_Pause(void) 
{
	if (preSTT_Machine != STT_Machine)
	{
		//Status
		preSTT_Machine = STT_Machine;
		//Exec
		lcd->LCDClear();
		if (tsr_stt == TSR_STOP)
		{
			lcd->LCDSetCursor(4,0);
			lcd->LCDPrintStr("PAUSE by TSR");
		}
		else
		{
			lcd->LCDSetCursor(7,0);
			lcd->LCDPrintStr("PAUSE");
		}
		tsr_stt = TSR_NONE;

		theta = 0;
		throttle_val = 0;
		api_set_FORWARD_control( pca9685,throttle_val);
		api_set_STEERING_control(pca9685,theta);
	}
}
// RUN STATE ==============================================================================================//
bool SM_Run(Parameters par)
{
// CHECK PCA9685 DRIVER ===================================================================================//
	//Check PCA9685 driver
	if (pca9685->error < 0)
	{
		cout<< endl<< "Error: PWM driver"<< endl<< flush;
		return false;
	}
// GET IMAGE ==============================================================================================//
	OpenNI2::Instance()->getData_new(colorImg, depthImg);
	et = getTickCount();
	fps = 1.0 / ((et-st)/freq);
	cout << "\tImage\tTime\t" <<  setprecision(3) << (et-st)/freq << "\tFPS\t" << fps << endl;
// MULTITHREADS ===========================================================================================//
	// std::thread thread_0;
	std::thread thread_1;
	// std::thread thread_2;
	std::thread thread_3;
	// thread_0 = std::thread(getImg,std::ref(colorImg),std::ref(depthImg));
    thread_1 = std::thread(lane_detection,par,std::ref(colorImg),std::ref(pointMidLane),std::ref(FilteredMidlanePoint));
    // thread_2 = std::thread(object_detection,std::ref(colorImg),std::ref(depthImg));

    // thread_3 = std::thread(tsr_depth_process,std::ref(colorImg),std::ref(depthImg),std::ref(tsr_stt));
	// thread_3 = std::thread(tsr_process,std::ref(colorImg),std::ref(tsr_stt),par.S_val,par.S_max,par.V_val,par.V_max);
	thread_3 = std::thread(new_depth_process,std::ref(colorImg),std::ref(depthImg),std::ref(tsr_stt));
	// thread_0.join();
    thread_1.join();
    // thread_2.join();
    thread_3.join();
	// ================================================================================================================================//
// CONTROL PHASE ==========================================================================================// 
	//double angDiff = getTheta(carPosition, pointMidLane);
	
	// if (angDiff < -/40)
	// 	theta = (angDiff*par.angRatioL1);
	// else if (angDiff < 0)
	// 	theta = (angDiff*par.angRatioL2);
	// else if (angDiff > 40)
	//     theta = angDiff*par.angRatioR1;
	// else theta = angDiff*par.angRatioR2;
	// cout << par.angRatioL1 << endl << par.angRatioL2 << endl <<par.angRatioR1 << endl << par.angRatioR2 << endl;
	// if (angDiff < -40)
	// 	theta = (angDiff*1.35);
	// else if (angDiff < 0)
	// 	theta = (angDiff*1.10);
	// else if (angDiff > 40)
	//     theta = angDiff*1.15;
	// else theta = angDiff*1.05;
	// cout << "			THETA:	" << theta << endl;
// CAL STEERING ===========================================================================================//
	delta_x = (FilteredMidlanePoint.x - carPosition.x)/3; // 320/90 = 3.5
	//u_control = par.Kp*delta_x + par.Ki*(delta_x + pre_delta_x);  // PI Controller
	u_control = 0.8*delta_x + 0.2*(delta_x + pre_delta_x);// + 0.1*(delta_x - pre_delta_x);  // PI Controller
	pre_delta_x = delta_x;

// SET STEERING & THROTTLE ================================================================================//
	//theta = (-90.00);
	// api_set_STEERING_control(pca9685,theta);

	if((u_control <= 10) && (u_control >= -10))
	{
		u_control *= 2;
	}
	else if(u_control > 15)
	{
		u_control = 20 + (u_control - 10)*0.9; //(90 - 40)/(90 - 20) = 0.75
	}
	else if(u_control < -15)
	{
		u_control = -20 + (u_control + 10)*0.9; //(90 - 40)/(90 - 20) = 0.75
		// if (u_control < -75)
		// 	u_control = -90;
	}

// CAL THROTTLE ===========================================================================================//
	if(set_throttle_val >= 50)
	{
		if((u_control > 75) || (u_control < -75))
		{
			throttle_val = set_throttle_val*0.8;
		}
		else
		{
			throttle_val = set_throttle_val;
		}
	}
	else if(set_throttle_val >= 60)
	{
		if((u_control > 80) || (u_control < -80))
		{
			throttle_val = set_throttle_val*0.8;
		}
		else if((u_control > 70) || (u_control < -70))
		{
			throttle_val = set_throttle_val*0.9;
		}
		else
		{
			throttle_val = set_throttle_val;
		}
	}
	else if(set_throttle_val >= 90)
	{
		if((u_control > 80) || (u_control < -80))
		{
			throttle_val = set_throttle_val*0.75;
		}
		else if((u_control > 70) || (u_control < -70))
		{
			throttle_val = set_throttle_val*0.8;
		}
		else if((u_control > 60) || (u_control < -60))
		{
			throttle_val = set_throttle_val*0.9;
		}
		else
		{
			throttle_val = set_throttle_val;
		}
	}
	else
	{
		if((u_control > 80) || (u_control < -80))
		{
			throttle_val = set_throttle_val*0.9;
		}
		else
		{
			throttle_val = set_throttle_val;
		}
	}

	if(set_throttle_val >= 60)
	{
		if(frame_id < 4)
		{
			throttle_val = set_throttle_val*0.6;
		}
		else if(frame_id < 10)
		{
			throttle_val = set_throttle_val*0.8;
		}
	}
	else if(set_throttle_val >= 90)
	{
		if(frame_id < 4)
		{
			throttle_val = set_throttle_val*0.5;
		}
		else if(frame_id < 10)
		{
			throttle_val = set_throttle_val*0.75;
		}
	}

	// if((angle_lane < -150) && (u_control <-20))
	// {
	// 	u_control = -90;
	// }
	// else if((angle_lane > 150) && (u_control > 20))
	// {
	// 	u_control = 90;
	// }

	// u_control = 0.0;
	api_set_STEERING_control(pca9685,u_control);
	int pwm2 =  api_set_FORWARD_control( pca9685,throttle_val);
	// setSteering(u_control);
	// setThrottle(throttle_val);

// IS SHOW FILE ===========================================================================================//
	Mat dst_s;
	if (!lane.empty())
	{
	    cvtColor(lane, dst_s, CV_GRAY2BGR); 
	    dst_s.copyTo(colorImg(Rect(0, 0, dst_s.cols, dst_s.rows)));
	}
	circle(colorImg, cv::Point(pointMidLane.x, 60 + par.roi.y), 2, Scalar(255,0,0), 5);
	circle(colorImg, cv::Point(FilteredMidlanePoint.x, 80+ par.roi.y + 5 ), 2, Scalar(0,255,0), 5);
	circle(colorImg, cv::Point(carPosition.x, 100 + par.roi.y), 2, Scalar(0,0,255), 5);
	circle(colorImg, cv::Point(pointMidLane.x, par.yCalc), 2, Scalar(255,0,0), 5);
	char buf[100];
	sprintf(buf, "ID: %d - delta_x %.1f u_control %.1f throttle_val %d angleLane %d",frame_id, delta_x, u_control, throttle_val, angle_lane);
	putText(colorImg, buf, cvPoint(10, 30), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255), 1, CV_AA);
	sprintf(buf, "pMidLane: %d - Filter %d",pointMidLane.x, FilteredMidlanePoint.x);
	putText(colorImg, buf, cvPoint(10, 60), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255), 1, CV_AA);
	sprintf(buf, "TSR stt: %d lanepos %d ",tsr_stt, lane_pos);
	putText(colorImg, buf, cvPoint(10, 90), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255), 1, CV_AA);


	// cv::imshow("depth",tmp_depth);

	// Mat dep;
	if (!bn_dh.empty())
	{
	    // cvtColor(bn_dh, dep, CV_GRAY2BGR); 
	    bn_dh.copyTo(colorImg(Rect(0, dst_s.rows, bn_dh.cols, bn_dh.rows)));
	}
	
	imshow("result", colorImg);
	if(par.is_save_file)
	{
		if (!colorImg.empty())
			color_videoWriter.write(colorImg);
	}

// CALCULATE FPS ==========================================================================================//
	et = getTickCount();
	fps = 1.0 / ((et-st)/freq);
	cout << "\tCar\t" << carPosition.x << endl;
	cout << "\tMid\t" << pointMidLane.x << "\tFilter\t" << FilteredMidlanePoint.x << "\tDelta\t" << delta_x << endl;
	cout << frame_id << "\tMain\tTime\t" <<  setprecision(3) << (et-st)/freq << "\tFPS\t" << fps << "\n" << endl;
// END RUN=================================================================================================//
	return true;
}
// FILTER POINTMIDLANE ====================================================================================//
cv::Point FilterMidlanePoint(cv::Point pointMidLane, CenterPoint_t *CenterPoint)
{	
	cv::Point OffsetDistance;
	cv::Point PredictedPoint;
	cv::Point TempPoint;
	cv::Point FilteredPoint(0,0);
	static cv::Point prePointMidLane;
	static int cnt_noPointMidLane = 0;

	FilteredPoint = CenterPoint->x[5];

	OffsetDistance = (CenterPoint->x[5] - CenterPoint->x[0])/5;
	PredictedPoint = FilteredPoint + OffsetDistance;

	//if ((pointMidLane.x > CenterPoint->MinRatioRange*PredictedPoint.x) && (pointMidLane.x < CenterPoint->MaxRatioRange*PredictedPoint.x))
	if ((pointMidLane.x > -(int)(CenterPoint->MinRatioRange) + PredictedPoint.x) && 
		(pointMidLane.x < (int)(CenterPoint->MaxRatioRange) + PredictedPoint.x) && 
		(pointMidLane.y != -1))
	{
		cnt_noPointMidLane = 0;
		TempPoint = pointMidLane;
	}
	else 
	{
		cnt_noPointMidLane++;
		TempPoint = PredictedPoint;
	}

	cv::Point rs_FilteredPoint = FilteredPoint + CenterPoint->AlphaLPF*(TempPoint - FilteredPoint);
	if((rs_FilteredPoint.x < 750) && (rs_FilteredPoint.x > -100))
	{
		FilteredPoint = rs_FilteredPoint;
	}

	for(int i = 0; i < 5; i++)
	{
		CenterPoint->x[i] = CenterPoint->x[i+1];
	}
	CenterPoint->x[5] = FilteredPoint;

	prePointMidLane = pointMidLane;
	return FilteredPoint;
}
// LANE DETECT ============================================================================================//
void lane_detection(Parameters par,cv::Mat &colorImg, cv::Point &pointMidLane, cv::Point &FilteredMidlanePoint)
{	
	double st1 = 0, et1 = 0, fps1 = 0;
	st1 = getTickCount();
	cv::Mat colorImg_clone = colorImg.clone();	
	if(!colorImg_clone.empty())
    {
		cv::Mat im_raw = colorImg(par.roi).clone();
		//imshow("Color Image", im_raw);
		// cv::Mat grayImage;
		// cvtColor(im_raw, grayImage, CV_BGR2GRAY);
		// cout <<"			SIGN: " << tsr_stt << endl;
		if((!im_raw.empty()) )//&& (!grayImage.empty()))
		{	
			cout << "\ttsr_stt: " << tsr_stt << endl;

			// if(tsr_stt == TSR_TURN_RIGHT)
			// {
			// 	//priority follow lane right
            // 	pointMidLane = new_lane_process(im_raw.size(), im_raw, lane, FilteredMidlanePoint.x, par.yCalc, par.point_offset, tsr_stt, lane_pos, angle_lane, 1);
			// }
			// else
			// {
				//priority follow lane left
				if (lane_pos <2)
					pointMidLane = new_lane_process_yu(im_raw.size(), im_raw, lane, FilteredMidlanePoint.x + 70, par.yCalc, par.point_offset, tsr_stt, lane_pos, angle_lane);
				else pointMidLane = new_lane_process_yu(im_raw.size(), im_raw, lane, FilteredMidlanePoint.x - 70, par.yCalc, par.point_offset, tsr_stt, lane_pos, angle_lane);
			// }
            FilteredMidlanePoint = FilterMidlanePoint(pointMidLane, &CenterPoint);
			// std::cout << "Point Mid: " << pointMidLane << std::endl;
		}
	}
	et1 = getTickCount();
	fps1 = 1.0 / ((et1-st1)/freq);
	// cout << "FPS lane:     " << fps1 << endl;
	cout << "\tLane\tTime\t" <<  setprecision(3) << (et1-st1)/freq << "\tFPS\t" << fps1 << endl;
}
// OBSTACLE ===============================================================================================//
void object_detection(cv::Mat &color, cv::Mat &depth)
{
	double st2 = 0, et2 = 0, fps2 = 0;
	st2 = getTickCount();

	// ushort depth_val;
	// cv::Mat depthImg_clone = depth.clone();	
	// if((!depthImg_clone.empty()))
	// {
	// 	cv::Mat depth_obstacle = depth(cv::Rect(0,300,640,85)).clone();
	// 	cv::Mat depth_bin_obstacle = cv::Mat::zeros(depth_obstacle.size(), CV_8UC1);
				
	// 	for (int y = 1; y < depth_obstacle.rows; y++)
	// 	{
	// 		for (int x = 1; x < depth_obstacle.cols; x++)
	// 		{
	// 			depth_val = depth_obstacle.at<ushort>(y,x);
	// 			if ((depth_val < l_th_obstacle) || (depth_val > h_th_obstacle)) 
	// 				depth_bin_obstacle.at<uchar>(y,x) = 0;
	// 			else depth_bin_obstacle.at<uchar>(y,x) = 255;
	// 		}
	// 	}
	// 	if(!depth_bin_obstacle.empty())
	// 	{
	// 		imshow("depth_bin_obstacle", depth_bin_obstacle);
	// 	}
	// }

// TEST FLOODFILL
	cv::Mat colorImg_clone = color(cv::Rect(320 - 100,240,200,80)).clone();
	cv::Mat depthImg_clone = depth(cv::Rect(320 - 100,240,200,80)).clone();	
	cv::Mat DetectionResult;
	Rect boxes;

	if((!colorImg_clone.empty()) && (!depthImg_clone.empty()))
	{
		detection.objectLabeling(boxes, depthImg_clone, l_th_test, h_th_test, 250, 800, 36, 300, 1.4);
	}

	DetectionResult = colorImg_clone(boxes).clone();

	if(!DetectionResult.empty())
	{ 
		
		resize(DetectionResult,DetectionResult,Size(80,80),INTER_LANCZOS4);
		cv::imshow("Detection Result",DetectionResult);	
							// cv::imwrite("_1roi" + to_string(img_cnt++) + "-" + to_string(boundRect[idx].tl().x) + ".png",image_roi);
		cv_image<bgr_pixel> images_HOG(DetectionResult);
		std::vector<rect_detection> rects;
		evaluate_detectors(detectors, images_HOG, rects, 0.5);
		cout << "rect size: " << rects.size() << endl;
		if((rects.size() > 0))// && (rects.size() < 3))
		{
#if 1
			if ((rects[0].weight_index == 0)  && (rects[0].detection_confidence > 0.9))
			{
				cout << "FTURN LEFT: " << rects[0].detection_confidence << endl;
			}
			else if ((rects[0].weight_index == 1)  && (rects[0].detection_confidence > 0.95) &&
					 (rects[1].weight_index == 2)  && (rects[1].detection_confidence < 0.925))
			{
				cout << "FTURN RIGHT: " << rects[0].detection_confidence << endl;
			}
			else if (((rects[0].weight_index == 1)  && (rects[0].detection_confidence > 0.95) &&
					 (rects[1].weight_index == 2)  && (rects[1].detection_confidence > 0.925)) ||
					 ((rects[0].weight_index == 2)  && (rects[0].detection_confidence > 0.95)))
			{
				cout << "FSTOP: " << rects[0].detection_confidence << endl;
			}

#else
			if (  (rects[0].weight_index == 0))// 8 ))//  && (rects[0].detection_confidence > 0.91) )
			{
//[Tri] Test filter
				Mat color_hsv, mask;
				cvtColor(DetectionResult, color_hsv, COLOR_BGR2HSV);
				inRange(color_hsv, Scalar(100,20,0), Scalar(135, 255, 130), mask);
				int count_ = 0;
				for (int y = 0; y < mask.rows; y++)
				{
					for (int x = 0; x < mask.cols; x++)
					{
					if (mask.at<char>(y,x) == 255)
					count_++;
					}
				}
											// imshow("bin", mask);
				if (count_ < 2500)
				{
												// cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tfail" << endl; 
					return;;
				}
											// cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tOK" << endl;
	//[Tri] End test                                         		                                           
				cout << "FTURN LEFT: " << rects[0].detection_confidence << endl;
											// tsr_rs = TSR_TURN_LEFT;
											// tsr_rs = TSR_STOP;
											// cv::rectangle( colorImg, Point(boxes.tl().x + 220, boxes.tl().y+240),  Point(boxes.br().x + 220, boxes.br().y+240), Scalar(0,255,0), 2, 8, 0 );
											// cv::imwrite("_1left" + to_string(img_cnt++) + "_" + to_string(rects[0].detection_confidence) + "-" + to_string(count_bp) + ".png",image_roi);
				return;
			}
			else if (  (rects[0].weight_index == 1))// && (rects[0].detection_confidence > 0.91) )
			{
	//[Tri] Test filter
				Mat color_hsv, mask;
				cvtColor(DetectionResult, color_hsv, COLOR_BGR2HSV);
				inRange(color_hsv, Scalar(100,20,0), Scalar(135, 255, 130), mask);
				int count_ = 0;
				for (int y = 0; y < mask.rows; y++)
				{
										for (int x = 0; x < mask.cols; x++)
										{
											if (mask.at<char>(y,x) == 255)
												count_++;
										}
									}
									// imshow("bin", mask);
									if (count_ < 2999 )
									{
										// cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tfail" << endl; 
										return;
									}
									// cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tOK" << endl;
//[Tri] End test
									cout << "FTURN RIGHT: " << rects[0].detection_confidence << endl;
									// tsr_rs = TSR_TURN_RIGHT;
									// tsr_rs = TSR_STOP;
									// cv::rectangle( colorImg, Point(boxes.tl().x + 220, boxes.tl().y+240),  Point(boxes.br().x + 220, boxes.br().y+240), Scalar(255,255,0), 2, 8, 0 );
									// cv::rectangle( colorImg, Point(boundRect[idx].tl().x, boundRect[idx].tl().y+240),  Point(boundRect[idx].br().x, boundRect[idx].br().y+240), Scalar(255,255,0), 2, 8, 0 );
									// cv::imwrite("_1right" + to_string(img_cnt++) + "_" + to_string(rects[0].detection_confidence) + "-" + to_string(count_bp) + ".png",image_roi);
									return;
								}
								else if ( (rects[0].weight_index == 2) )//&& (rects[0].detection_confidence > 0.91) )
								{
//[Tri] Test filter
									Mat color_hsv, mask;
									cvtColor(DetectionResult, color_hsv, COLOR_BGR2HSV);
									inRange(color_hsv, Scalar(165,105,55), Scalar(255, 255, 170), mask);
									int count_ = 0;
									for (int y = 0; y < mask.rows; y++)
									{
										for (int x = 0; x < mask.cols; x++)
										{
											if (mask.at<ushort>(y,x) == 255)
												count_++;
										}
									}
									imshow("bin", mask);
									if (count_ < 2500 )
									{
										// cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_ << endl;
										return;
									}
//[Tri] End test
									cout << "FSTOP: "<< rects[0].detection_confidence << endl;
									// tsr_rs = TSR_SsTOP;
									// cv::rectangle( colorImg, Point(boxes.tl().x + 220, boxes.tl().y+240),  Point(boxes.br().x + 220, boxes.br().y+240), Scalar(0,0,255), 2, 8, 0 );
									// cv::imwrite("_1stop" + to_string(img_cnt++) + "_" + to_string(rects[0].detection_confidence) + "-" + to_string(count_bp) + ".png",image_roi);
									// cv::rectangle( colorImg, Point(boundRect[idx].tl().x, boundRect[idx].tl().y+240),  Point(boundRect[idx].br().x, boundRect[idx].br().y+240), Scalar(0,0,255), 2, 8, 0 );
									return;
								}
								else 
								{
									cout << "FNONE: " << endl;
									turn_count = 20;
								}
#endif
		}
		for(int x = 0; x < rects.size(); x++)
		{
			cout << x << " " << rects[x].weight_index << " " << rects[x].detection_confidence << endl;
		}
		cout << "-----------------------" << endl;
	}

	et2 = getTickCount();
	fps2 = 1.0 / ((et2-st2)/freq);
	//cout << "FPS object:     " << fps2 << endl;	
}
// TSR DEPTH ==============================================================================================//
void tsr_depth_process(cv::Mat &colorImg, cv::Mat &depthImg, int &tsr__)
{
	double st3 = 0, et3 = 0, fps3 = 0;
	st3 = getTickCount();
	cout << "\tturn count " << turn_count << endl;
	if ( ( (tsr__ == TSR_TURN_LEFT) || (tsr__ == TSR_TURN_RIGHT) ) && (turn_count < 20) )
	{
		turn_count++;
		// if (turn_count == 1)
		// {
			// tsr__ = TSR_STOP;
		// }
	}
	else 
	{
		turn_count = 0;
		cv::Mat colorImg_clone = colorImg.clone();
		cv::Mat depthImg_clone = depthImg.clone();
			
		if((!colorImg_clone.empty()) && (!depthImg_clone.empty()))
		{
			cv::Mat TSRDepthImg = depthImg_clone(cv::Rect(0,180,640,150)).clone();
			// cv::Mat TSRDepthImg = depthImg_clone.clone();
			cv::Mat tmp_depth;
			double min, max;
			minMaxLoc(depthImg_clone, &min, &max);
			float scaleFactor = 256.0/max;
            TSRDepthImg.convertTo( tmp_depth, CV_8UC1, scaleFactor );
			cv::imshow("depth",tmp_depth);
			//cv::imshow("color",colorImg_clone);

			cv::Mat DetectionResult;

			// std::vector<Rect> boxes;
			// std::vector<int> labels;
			// detection.objectLabeling(boxes, labels, TSRDepthImg, colorImg, DetectionResult, l_th, h_th, 1000, 4000, 36, 200, 1.2);//1000-4000-36-200-1.2
			Rect boxes;
			detection.objectLabeling(boxes, TSRDepthImg, l_th, h_th, 4000, 6000, 36, 300, 1.4);

			if(boxes.width != 0)
			{
				DetectionResult = colorImg(boxes);
				cout << "\tTSRFrame\t" << frame_id << endl;
			}

			if(DetectionResult.empty())
			{
				cout<<"----No traffic sign"<<endl;
				// break;
			}	
			else
			{  
				cv::imwrite("test" + to_string(img_cnt++) + ".png",DetectionResult);
				cout<<"++++Traffic sign detected"<<endl;
				//cv::imshow("Detection Result",DetectionResult);
				Mat src;
				src = DetectionResult.clone();
				resize(src,src,Size(80,80),INTER_LANCZOS4);// vi dieu
				//cv::imshow("src",src);				
				cv_image<bgr_pixel> images_HOG(src);
				std::vector<rect_detection> rects;

				evaluate_detectors(detectors, images_HOG, rects, 0.9);
				// cout << "Rects size: " << rects.size() << endl;
				if(rects.size() > 0)
				{
					//cout << "		index: " <<rects[0].weight_index<< endl;
					if (rects[0].weight_index == 6) 
					{		                                           		                                            
						cout << "TURN LEFT: " << rects[0].detection_confidence << endl;
						// cout << "TSR frame id" << frame_id << endl;
						tsr__ = TSR_TURN_LEFT;
					}
					else if (rects[0].weight_index == 7) 
					{
						cout << "TURN RIGHT: " << rects[0].detection_confidence << endl;
						// cout << "TSR frame id" << frame_id << endl;
						tsr__ = TSR_TURN_RIGHT;
					}
					else if (rects[0].weight_index == 5)
					{
						cout << "STOP: "<< rects[0].detection_confidence << endl;
						// cout << "TSR frame id" << frame_id << endl;
						tsr__ = TSR_STOP;
					}
					else 
					{
						cout << "NONE: " << endl;
						turn_count = 20;
					}	
				}
			}
		}
		// else
		// 	cout << "nothing 3" << endl;
	}	
	et3 = getTickCount();
	fps3 = 1.0 / ((et3-st3)/freq);
	// cout << "FPS TSR:     " << fps3 << endl;
	cout << "\tDTSR\tTime\t" <<  setprecision(3) << (et3-st3)/freq << "\tFPS\t" << fps3 << endl;	
}
// NEW DEPTH PROCESS
void new_depth_process(cv::Mat &color, cv::Mat &depth, int &tsr_rs)
{
#if 0 // Floodfill
// TEST FLOODFILL
	cv::Mat colorImg_clone = color(cv::Rect(320 - 100,240,200,80)).clone();
	cv::Mat depthImg_clone = depth(cv::Rect(320 - 100,240,200,80)).clone();	
	cv::Mat DetectionResult;
	Rect boxes;

	if((!colorImg_clone.empty()) && (!depthImg_clone.empty()))
	{
		detection.objectLabeling(boxes, depthImg_clone, l_th_test, h_th_test, 250, 800, 36, 300, 1.4);
	}

	DetectionResult = colorImg_clone(boxes).clone();

	if(!DetectionResult.empty())
	{ 
		
		resize(DetectionResult,DetectionResult,Size(80,80),INTER_LANCZOS4);
		cv::imshow("Detection Result",DetectionResult);	
		cv::imwrite("_1roi" + to_string(img_cnt++) + ".png",DetectionResult);
		cv_image<bgr_pixel> images_HOG(DetectionResult);
		std::vector<rect_detection> rects;
		evaluate_detectors(detectors, images_HOG, rects, 0.5);
		cout << img_cnt << " rect size: " << rects.size() << endl;
		if((rects.size() > 0))// && (rects.size() < 3))
		{
			if ((rects[0].weight_index == 0)  && (rects[0].detection_confidence > 0.9))
			{
				cout << "FTURN LEFT: " << rects[0].detection_confidence << endl;
				// tsr_rs = TSR_TURN_LEFT;
			}
			else if ((rects[0].weight_index == 1)  && (rects[0].detection_confidence > 0.95) &&
					 (rects[1].weight_index == 2)  && (rects[1].detection_confidence < 0.925))
			{
				cout << "FTURN RIGHT: " << rects[0].detection_confidence << endl;
				// tsr_rs = TSR_TURN_RIGHT;
			}
			else if (((rects[0].weight_index == 1)  && (rects[0].detection_confidence > 0.95) &&
					 (rects[1].weight_index == 2)  && (rects[1].detection_confidence > 0.925)) ||
					 ((rects[0].weight_index == 2)  && (rects[0].detection_confidence > 0.95)))
			{
				cout << "FSTOP: " << rects[0].detection_confidence << endl;
				// tsr_rs = TSR_STOP;
			}
			else
			{
				cout << "FNONE" << endl;
			}
		}
		for(int x = 0; x < rects.size(); x++)
		{
			cout << x << " " << rects[x].weight_index << " " << rects[x].detection_confidence << endl;
		}
		cout << "-----------------------" << endl;
	}

#else
	cout << "\tturn count " << turn_count << endl;
	if ( ( (tsr_rs == TSR_TURN_LEFT) || (tsr_rs == TSR_TURN_RIGHT) ) && (turn_count < 0) )
	{
		turn_count++;
		// tsr_rs = TSR_STOP;
	}
	else 
	{
		turn_count = 0;
		ushort depth_val;
		cv::Mat colorImg_clone = color.clone();
		cv::Mat depthImg_clone = depth.clone();	
		if((!colorImg_clone.empty()) && (!depthImg_clone.empty()))
		{
			cv::Mat depth_tsr = depth(cv::Rect(0,200,640,80)).clone();
			cv::Mat color_tsr = color(cv::Rect(0,200,640,80)).clone();
			cv::Mat depth_bin = cv::Mat::zeros(depth_tsr.size(), CV_8UC1);
			// imshow("color", color_tsr);
                   
			for (int y = 0; y < depth_tsr.rows; y++)
			{
				for (int x = 0; x < depth_tsr.cols; x++)
				{
					depth_val = depth_tsr.at<ushort>(y,x);
					if ((depth_val < l_th) || (depth_val > h_th)) depth_bin.at<uchar>(y,x) = 0;
					else depth_bin.at<uchar>(y,x) = 255;
				}
			}
			if(!depth_bin.empty())
				cvtColor(depth_bin, bn_dh, CV_GRAY2BGR);
			// imshow("depth_bin", depth_bin);
			std::vector<std::vector<cv::Point> > contours_tsr;
			std::vector<Vec4i> hierarchy_tsr;
			findContours(depth_bin, contours_tsr, hierarchy_tsr, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			std::vector<Rect> boundRect( contours_tsr.size() );
			for(int idx = 0; idx < contours_tsr.size(); idx++)
			{
				boundRect[idx] = boundingRect( Mat(contours_tsr[idx]) );

//[Tri] Filter depth img
				int count = 0;
				Mat depthImage_roi = depth_bin(boundRect[idx]).clone();
				for (int y = 0; y < depthImage_roi.rows; y++)
				{
					for (int x = 0; x < depthImage_roi.cols; x++)
					{
						if (depthImage_roi.at<char>(y,x) == 255)
							count++;
					}
				}
				if(count < (0.5 * depthImage_roi.rows * depthImage_roi.cols))
				{
					continue;
				}
				// if((boundRect[idx].tl().x - FilteredMidlanePoint.x > 150) ||
				//    (FilteredMidlanePoint.x - boundRect[idx].tl().x > 200))
				//    {
				// 		continue;
				//    }

				// cout << depthImage_roi << endl;
				// imshow("depthImage_roi", depthImage_roi);
				// cout << "x " << boundRect[idx].tl().x << " " << count << "\tcols: " << depthImage_roi.cols << "\t row: " << depthImage_roi.rows << endl;
//[Tri] End test

				if ((boundRect[idx].width*boundRect[idx].height > 400) && (boundRect[idx].width*boundRect[idx].height < 6000))
				{
					// if((boundRect[idx].tl().x != 0)&&(boundRect[idx].tl().y != 0) && (boundRect[idx].br().x != depth_bin.cols) && (boundRect[idx].br().y != depth_bin.rows))
					// {
						if ( ( (float)boundRect[idx].width/boundRect[idx].height > 0.5) && ( (float)boundRect[idx].width/boundRect[idx].height < 1.3 ) )
						{
							Mat image_roi = color_tsr(boundRect[idx]).clone();
							// Mat color_hsv, mask_b, mask_s;
							// cvtColor(image_roi, color_hsv, COLOR_BGR2HSV);
							// inRange(color_hsv, Scalar(95,135,70), Scalar(120, 210, 110), mask_b);
							// inRange(color_hsv,Scalar(0, 30, 140), Scalar(20, 255, 220), mask_s);
							// Mat masksum = mask_b;// + mask_s;
							// // imshow("SUM", masksum);
							int count_bp = 0;
							// for (int y = 0; y < masksum.rows; y++)
							// {
							// 	for (int x = 0; x < masksum.cols; x++)
							// 	{
							// 		if (masksum.at<ushort>(y,x) == 255)
							// 			count_bp++;
							// 	}
							// }
							// // imshow("bin", mask_b);
							// if (count_bp < 80 )
							// {
							// 	cout << "\t\t\t id: " << img_cnt << "\t bp: " << count_bp << "  " << (mask_b.rows * mask_b.cols) << endl; 
							// 	continue;
							// }
							// int tich = mask_b.rows * mask_b.cols;
							cv::rectangle( colorImg, Point(boundRect[idx].tl().x, boundRect[idx].tl().y+200),  Point(boundRect[idx].br().x, boundRect[idx].br().y+200), Scalar(255,0,0), 2, 8, 0 );
							// std::vector<std::vector<cv::Point> > contours_color;
							// std::vector<Vec4i> hierarchy_color;
							// findContours(mask_b, contours_color, hierarchy_color, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

							
							resize(image_roi,image_roi,Size(80,80),INTER_LANCZOS4);
							// cv::imwrite("_1roi" + to_string(img_cnt++) + "-" + to_string(boundRect[idx].tl().x) + ".png",image_roi);
							cv_image<bgr_pixel> images_HOG(image_roi);
							std::vector<rect_detection> rects;
							evaluate_detectors(detectors, images_HOG, rects, 0.85);
							// cout << "rect size: " << rects.size() << endl;
							cout << boundRect[idx] << endl;

									Mat calc_dis_img = depth_tsr(boundRect[idx]).clone();
									int calc_count = 0;
									int calc_sum = 0;
									for (int y = 0; y < depthImage_roi.rows; y++)
									{
										for (int x = 0; x < depthImage_roi.cols; x++)
										{
											if (depthImage_roi.at<char>(y,x) == 255)
											{
												calc_count++;
												calc_sum += calc_dis_img.at<ushort>(y,x);
											}
										}
									}
									double dist = calc_sum/calc_count;
								cout << "dis " << dist << endl;
								for(int x = 0; x < rects.size(); x++)
								{
									cout << x << " " << rects[x].weight_index << " " << rects[x].detection_confidence << endl;
								}
								cout << "-----------------------" << endl;

							if((rects.size() > 0) && (rects.size() < 3))
							{
								if (  (rects[0].weight_index == 0))// 8 ))//  && (rects[0].detection_confidence > 0.91) )
								{		
//[Tri] Test filter left
#if 1
									Mat color_hsv, mask;
									cvtColor(image_roi, color_hsv, COLOR_BGR2HSV);
									inRange(color_hsv, Scalar(70,115,145), Scalar(125, 255, 255), mask);
									int count_ = 0;
									for (int y = 0; y < mask.rows; y++)
									{
										for (int x = 0; x < mask.cols; x++)
										{
											if (mask.at<char>(y,x) == 255)
												count_++;
										}
									}
									// imshow("bin", mask);
									if (count_ < 1000)
									{
										cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tfail" << endl; 
										break;
									}
									cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tOK" << endl;
#endif
//[Tri] End test left                              		                                            
									cout << "TURN LEFT: " << rects[0].detection_confidence << endl;
									tsr_rs = TSR_TURN_LEFT;
									// tsr_rs = TSR_STOP;
									cv::rectangle( colorImg, Point(boundRect[idx].tl().x, boundRect[idx].tl().y+200),  Point(boundRect[idx].br().x, boundRect[idx].br().y+200), Scalar(0,255,0), 2, 8, 0 );
									cv::imwrite("_1left" + to_string(img_cnt++) + "_" + to_string(rects[0].detection_confidence) + "-" + to_string(count_bp) + ".png",image_roi);
									break;
								}
								else if (  (rects[0].weight_index == 1))// && (rects[0].detection_confidence > 0.91) )
								{
//[Tri] Test filter right
#if 1
									Mat color_hsv, mask;
									cvtColor(image_roi, color_hsv, COLOR_BGR2HSV);
									// inRange(color_hsv, Scalar(100,20,0), Scalar(135, 255, 130), mask);
									inRange(color_hsv, Scalar(70,115,145), Scalar(125, 255, 255), mask);
									int count_ = 0;
									for (int y = 0; y < mask.rows; y++)
									{
										for (int x = 0; x < mask.cols; x++)
										{
											if (mask.at<char>(y,x) == 255)
												count_++;
										}
									}
									// imshow("bin", mask);
									if (count_ < 1000 )
									{
										cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tfail" << endl; 
										break;
									}
									cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tOK" << endl;
#endif
//[Tri] End test right
									cout << "TURN RIGHT: " << rects[0].detection_confidence << endl;
									tsr_rs = TSR_TURN_RIGHT;
									// tsr_rs = TSR_STOP;
									cv::rectangle( colorImg, Point(boundRect[idx].tl().x, boundRect[idx].tl().y+200),  Point(boundRect[idx].br().x, boundRect[idx].br().y+200), Scalar(255,255,0), 2, 8, 0 );
									cv::imwrite("_1right" + to_string(img_cnt++) + "_" + to_string(rects[0].detection_confidence) + "-" + to_string(count_bp) + ".png",image_roi);
									break;
								}
								else if ( (rects[0].weight_index == 2) )//&& (rects[0].detection_confidence > 0.91) )
								{
//[Tri] Test filter stop
#if 0
									Mat color_hsv, mask;
									cvtColor(image_roi, color_hsv, COLOR_BGR2HSV);
									inRange(color_hsv, Scalar(165,105,55), Scalar(255, 255, 170), mask);
									int count_ = 0;
									for (int y = 0; y < mask.rows; y++)
									{
										for (int x = 0; x < mask.cols; x++)
										{
											if (mask.at<ushort>(y,x) == 255)
												count_++;
										}
									}
									imshow("bin", mask);
									if (count_ < 2000 )
									{
										cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_ << endl;
										break;
									}
									cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_  << "\tOK" << endl;
#endif
//[Tri] End test stop
									cout << "STOP: "<< rects[0].detection_confidence << endl;
									tsr_rs = TSR_STOP;
									cv::imwrite("_1stop" + to_string(img_cnt++) + "_" + to_string(rects[0].detection_confidence) + "-" + to_string(count_bp) + ".png",image_roi);
									cv::rectangle( colorImg, Point(boundRect[idx].tl().x, boundRect[idx].tl().y+200),  Point(boundRect[idx].br().x, boundRect[idx].br().y+200), Scalar(0,0,255), 2, 8, 0 );
									break;
								}
								else 
								{
									// Mat color_hsv, mask;
									// cvtColor(image_roi, color_hsv, COLOR_BGR2HSV);
									// inRange(color_hsv, Scalar(165,105,55), Scalar(255, 255, 170), mask);
									// int count_ = 0;
									// for (int y = 0; y < mask.rows; y++)
									// {
									// 	for (int x = 0; x < mask.cols; x++)
									// 	{
									// 		if (mask.at<ushort>(y,x) == 255)
									// 			count_++;
									// 	}
									// }
									// imshow("bin", mask);
									// if (count_ < 2500 )
									// {
									// 	cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_ << endl;
									// 	// break;
									// }
									// cout << "STOP: "<< rects[0].detection_confidence << endl;
									// tsr_rs = TSR_STOP;
									// cv::imwrite("_1stop" + to_string(img_cnt++) + "_" + to_string(rects[0].detection_confidence) + "-" + to_string(count_bp) + ".png",image_roi);
									// cv::rectangle( colorImg, Point(boundRect[idx].tl().x, boundRect[idx].tl().y+240),  Point(boundRect[idx].br().x, boundRect[idx].br().y+240), Scalar(0,0,255), 2, 8, 0 );

									cout << "NONE: " << endl;
									turn_count = 20;
								}
							}


// 							else
// 							{
// //[Tri] Test filter
// 								Mat color_hsv, mask;
// 								cvtColor(image_roi, color_hsv, COLOR_BGR2HSV);
// 								inRange(color_hsv, Scalar(165,105,55), Scalar(255, 255, 170), mask);
// 								int count_ = 0;
// 								for (int y = 0; y < mask.rows; y++)
// 								{
// 									for (int x = 0; x < mask.cols; x++)
// 									{
// 										if (mask.at<ushort>(y,x) == 255)
// 											count_++;
// 									}
// 								}
// 								imshow("bin", mask);
// 								if (count_ < 2500 )
// 								{
// 									cout << "x: " << boundRect[idx].tl().x << "\t bp: " << count_ << endl;
// 									// break;
// 								}
// 								cout << "STOP: "<< rects[0].detection_confidence << endl;
// 								tsr_rs = TSR_STOP;
// 								cv::imwrite("_1stop" + to_string(img_cnt++) + "_" + to_string(rects[0].detection_confidence) + "-" + to_string(count_bp) + ".png",image_roi);
// 								cv::rectangle( colorImg, Point(boundRect[idx].tl().x, boundRect[idx].tl().y+240),  Point(boundRect[idx].br().x, boundRect[idx].br().y+240), Scalar(0,0,255), 2, 8, 0 );
// //[Tri] End test
// 							}
						// }
					}
				}
			}
		}
	}
#endif
}

// TSR COLOR ==============================================================================================//
void tsr_process(cv::Mat &colorImg, int &tsr__, int S_val, int S_max, int V_val, int V_max)
{
	double st4 = 0, et4 = 0, fps4 = 0;
	st4 = getTickCount();
	// cout << "turn count " << turn_count << endl;
	if ( ( (tsr__ == TSR_TURN_LEFT) || (tsr__ == TSR_TURN_RIGHT) ) && (turn_count < 20) )
	{
		turn_count++;
		// if (turn_count == 19)
		// {
			// tsr__ = TSR_STOP;
		// }
	}
	else 
	{
		turn_count = 0;
		// tsr__ = TSR_NONE;
		static int TSR_DetectedFrameCnt = 0;
		static int TSR_NumFrameCnt = 0;
		cv::Mat colorImg_clone = colorImg.clone();

		if(!colorImg_clone.empty())
		{
			cv::Mat TSRColorImg = colorImg(cv::Rect(0,180,640,150)).clone();
			Mat img_raw = TSRColorImg;
			Mat img_HSV, maskr, maskr1, maskr2, maskb1, maskb2, maskk, edged;
			std::vector<char> save_template;

			// imshow("img_raw", TSRColorImg);
			cvtColor(TSRColorImg, img_HSV, COLOR_BGR2HSV);
			inRange(img_HSV, Scalar(85, 70, 130), Scalar(150, 230, 255), maskb1);
			inRange(img_HSV, Scalar(109, 52, 72), Scalar(146, 210, 160), maskb2);

			inRange(img_HSV, Scalar(150, 50, 50), Scalar(155, 255, 255), maskr1);
			inRange(img_HSV, Scalar(0, 90, 50), Scalar(5, 255, 255), maskr2);

			edged = maskb1 + maskb2 + maskr1 + maskr2;
			// imshow("edged", edged);
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
										if ( ( (boundRectw[iw].width*boundRectw[iw].height) > 900) && ( (boundRectw[iw].width*boundRectw[iw].height)  < 50000))
										{        							
											if ( ( (float)boundRectw[iw].width/boundRectw[iw].height < 1.3) && ( (float)boundRectw[iw].width/boundRectw[iw].height > 0.5) )
											{
												Rect dstbound (boundRectw[iw].tl().x + boundRect[idx].tl().x - 2, boundRectw[iw].tl().y + boundRect[idx].tl().y - 2,boundRectw[iw].width, boundRectw[iw].height);
												Mat image_roi = TSRColorImg(dstbound);
												resize(image_roi,image_roi,Size(80,80),INTER_LANCZOS4);
												imshow("result",image_roi);
												cv_image<bgr_pixel> images_HOG(image_roi);
												std::vector<rect_detection> rects;
												evaluate_detectors(detectors, images_HOG, rects, 0.9);
												//cout << rects.size() << endl;
												if(rects.size() > 0)
												{
													//cout << "		index: " <<rects[0].weight_index<< endl;
													if (rects[0].weight_index == 6) 
													{		                                           		                                            
														cout << "TURN LEFT: " << rects[0].detection_confidence << endl;
														// cout << "TSR frame id" << frame_id << endl;
														tsr__ = TSR_TURN_LEFT;
														cv::imwrite("left" + to_string(img_cnt++) + ".png",image_roi);
														break;
													}
													else if (rects[0].weight_index == 7) 
													{
														cout << "TURN RIGHT: " << rects[0].detection_confidence << endl;
														// cout << "TSR frame id" << frame_id << endl;
														tsr__ = TSR_TURN_RIGHT;
														cv::imwrite("right" + to_string(img_cnt++) + ".png",image_roi);
														break;
													}
													else if (rects[0].weight_index == 5)
													{
														cout << "STOP: "<< rects[0].detection_confidence << endl;
														// cout << "TSR frame id" << frame_id << endl;
														tsr__ = TSR_STOP;
														cv::imwrite("stop" + to_string(img_cnt++) + ".png",image_roi);
														break;
													}
													else 
													{
														cout << "NONE: " << endl;
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
	et4 = getTickCount();
	fps4 = 1.0 / ((et4-st4)/freq);
	cout << "\tCTSR\tTime\t" <<  setprecision(3) << (et4-st4)/freq << "\tFPS\t" << fps4 << endl;	
}
// GET IMAGE  =============================================================================================//
void getImg(cv::Mat &color, cv::Mat &depth)
{
	double st9 = 0, et9 = 0, fps9 = 0;
	st9 = getTickCount();
	OpenNI2::Instance()->getData_new(color, depth);
	//cv::Mat color_clone = color(Rect(0,360,640,120)).clone();
	//imshow("color", color_clone);
	et9 = getTickCount();
	fps9 = 1.0 / ((et9-st9)/freq);
	cout << "FPS img:     " << fps9 << endl;
}
//=========================================================================================================//
void setThrottle(int throttle)
{
	static int throttleSet = 0;
	throttleSet += 5;

	if(frame_id == 1)
	{  
		throttleSet = 0;
	}

	if(throttleSet > throttle + 5)
	{
		throttleSet -= 5;
	}
	else if (throttleSet < throttle - 5)
	{
		throttleSet += 5;
	}
	else
	{
		throttleSet = throttle;
	}
	api_set_FORWARD_control( pca9685, throttleSet);
}

void setSteering(int steering)
{
	static double steeringSet = 0;

	if(frame_id == 1)
	{
		steeringSet = 0;
	}

	if(steeringSet > steering + 10)
	{
		steeringSet -= 10;
	}
	else if(steeringSet <  steering - 10)
	{
		steeringSet += 10;
	}
	else
	{
		steeringSet = steering;
	}
	api_set_STEERING_control(pca9685, steeringSet);
}

//=========================================================================================================//

// MAIN PROGRAM ===========================================================================================//
int main( int argc, char* argv[] ) 
{
// LOAD PARAMETERS ========================================================================================//
	static int countStop = 0;
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

	if(value.size() == 29)
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
		par.test_delta_x = value[25];
		par.Kp = value[26];
		par.Ki = value[27];
		par.Kd = value[28];
		
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
		cout << par.test_delta_x << endl;
		cout << par.Kp <<endl;
		cout << par.Ki <<endl;
		cout << par.Kd <<endl;
		cout << endl << "Loaded all parameters" << endl << endl;
	}
	else 
	{
		cout << "Missing Parameters" << endl;
		return 0;
	}
	//================================================================================================================//

// LOAD SVM DETECTORS =====================================================================================//
	cout << "Loading SVM detectors..." << endl;

	signs.push_back(TrafficSign("turn-left-ahead", "svm_detectors/turn-left-ahead.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("turn-right-ahead", "svm_detectors/turn-right-ahead.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("stop-give-way", "svm_detectors/stop-give-way.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("turn-left-ahead", "svm_detectors/left2.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("turn-right-ahead", "svm_detectors/right2.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("stop-give-way", "svm_detectors/stop2.svm", rgb_pixel(0,0,255)));	

	// signs.push_back(TrafficSign("maximum-speed", "svm_detectors/maximum-speed.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("minimum-speed", "svm_detectors/minimum-speed.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("minimum-speed-end", "svm_detectors/minimum-speed-end.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("national-speed-limit", "svm_detectors/national-speed-limit.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("no-entry", "svm_detectors/no-entry.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("no-left-turn", "svm_detectors/no-left-turn.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("no-right-turn", "svm_detectors/no-right-turn.svm", rgb_pixel(0,0,255)));

	// signs.push_back(TrafficSign("left", "svm_detectors/left.svm", rgb_pixel(0,0,255)));
	// signs.push_back(TrafficSign("right", "svm_detectors/right.svm", rgb_pixel(0,0,255)));
	for (int i = 0; i < signs.size(); i++) {
		object_detector<image_scanner_type> detector;
		deserialize(signs[i].svm_path) >> detector;
		detectors.push_back(detector);
	}
	//==============================================================================================================//

// GPIO INIT ==============================================================================================//
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
// LCD INIT ===============================================================================================//
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
// CAMERA INIT ============================================================================================//
	//Init OpenNI2
	bool rs = OpenNI2::Instance() -> init();
	if(!rs)
		return -1;
// LOG FILE INIT ==========================================================================================//
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
// PCA9685 DRIVER INIT ====================================================================================//
	//Init PCA9685 driver
	api_pwm_pca9685_init( pca9685 );
	if (pca9685->error >= 0)
	//api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
	api_set_FORWARD_control( pca9685,throttle_val);

	//Init MSAC vanishing point library
	MSAC msac;
	cv::Rect roi1 = cv::Rect(0, VIDEO_FRAME_HEIGHT*3/4,VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT/4);
	api_vanishing_point_init( msac );
// START INIT =============================================================================================//
	//Init direction and ESC speed 
	throttle_val = 0;
	theta = 0;
	if(argc == 2 ) set_throttle_val = atoi(argv[1]);
	fprintf(stderr, "Initial throttle: %d\n", set_throttle_val);
	int frame_width = VIDEO_FRAME_WIDTH;
	int frame_height = VIDEO_FRAME_HEIGHT;
	lcd->LCDSetCursor(4,3);
	lcd->LCDPrintStr("WAITING ....");
// FILTER INIT ============================================================================================//
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
// END INIT ===============================================================================================//

	while (true)
	{
		Point center_point(0,0);
	 	st = getTickCount();
	 	key = getkey();
		unsigned int bt_status = 0;
		static int flag_wait = 0;
// BUTTON =================================================================================================//
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
					//fprintf(stderr, "Pause.\n");
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
		//START
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
					tsr_stt = TSR_NONE;
					flag_wait = 1;
					// STT_Machine = SM_RUN;
					//fprintf(stderr, "Wait Run.\n");
					//Exec
					lcd->LCDClear();
					lcd->LCDSetCursor(5,0);
					lcd->LCDPrintStr("WAIT RUN");
				}
				else
				{
					STT_Machine = SM_START;
					//fprintf(stderr, "Start.\n");
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
					//fprintf(stderr, "Run.\n");
					    //Exec
					lcd->LCDClear();
					lcd->LCDSetCursor(7,0);
					lcd->LCDPrintStr("RUNNING");
					flag_wait = 0;
				}
			}

			if(tsr_stt == TSR_NONE)
			{
				OpenNI2::Instance()->getData_new(colorImg, depthImg);
				new_depth_process(colorImg, depthImg, tsr_stt);
			}
			else
			{
				if(tsr_stt == TSR_TURN_LEFT)
				{
					lcd->LCDSetCursor(7,2);
					lcd->LCDPrintStr("LEFT");
				}
				else if(tsr_stt == TSR_TURN_RIGHT)
				{
					lcd->LCDSetCursor(7,2);
					lcd->LCDPrintStr("RIGHT");
				}
			}
		}
// KEY BOARD ==============================================================================================//
		switch(key) 
		{
			case 's':
				//Status
			preSTT_Machine = STT_Machine;
			STT_Machine = SM_START;
			//fprintf(stderr, "Start.\n");
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
			//fprintf(stderr, "Run.\n");
				//Exec
			lcd->LCDClear();
			lcd->LCDSetCursor(7,0);
			lcd->LCDPrintStr("RUNNING");
			break;
			case 'p':
				//Status
			preSTT_Machine = STT_Machine;
			STT_Machine = SM_PAUSE;
			//fprintf(stderr, "Pause.\n");
				//Exec
			lcd->LCDClear();
			lcd->LCDSetCursor(7,0);
			lcd->LCDPrintStr("PAUSE");
			theta = 0;
			throttle_val = 0;
			api_set_FORWARD_control(pca9685,throttle_val);
			break;
			case '1':
			tsr_stt = 1;
			break;
			case '2':
			tsr_stt = 2;
			break;
		}

// STATE MACHINE ==========================================================================================//
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
			if (tsr_stt == TSR_STOP) 
			{	
				countStop++;
				if(countStop > 2)
				{
					preSTT_Machine = STT_Machine;
					STT_Machine = SM_PAUSE;
				}
				lcd->LCDSetCursor(7,3);
				lcd->LCDPrintStr("STOP ");
			}
			else if (tsr_stt == TSR_TURN_LEFT)
			{
				lcd->LCDSetCursor(7,3);
				lcd->LCDPrintStr("LEFT ");
			}
			else if (tsr_stt == TSR_TURN_RIGHT)
			{
				lcd->LCDSetCursor(7,3);
				lcd->LCDPrintStr("RIGHT");
			}
			break;
			case SM_PAUSE:
			SM_Pause();
			countStop = 0;
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
// IS SHOW CAM ============================================================================================//
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
