//Uy_180318
/**
    This code runs our car automatically and log video, controller (optional)
    Line detection method: Canny
    Targer point: vanishing point
    Control: pca9685
    
    You should understand this code run to image how we can make this car works from image processing coder's perspective.
    Image processing methods used are very simple, your task is optimize it.
    Besure you set throttle val to 0 before end process. If not, you should stop the car by hand.
    In our experience, if you accidental end the processing and didn't stop the car, you may catch it and switch off the controller physically or run the code again (press up direction button then enter).
**/
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
//==============================================Uy=================================================//
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
using namespace std;
using namespace dlib;
using namespace framework;
using namespace signDetection;
using namespace SignRecognition;

signDetection::SignDetection detection;

struct TrafficSign {
	string name;
	string svm_path;
	rgb_pixel color;
	TrafficSign(string name, string svm_path, rgb_pixel color) :
	name(name), svm_path(svm_path), color(color) {};
};


ushort l_th = 600, h_th = 26000;//old: 600-2000
cv::Mat depthImg1, disparity;
//===============================================================================================//
using namespace openni;
using namespace EmbeddedFramework;
using namespace cv;
#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
#define VIDEO_FRAME_WIDTH 640	
#define VIDEO_FRAME_HEIGHT 480

#define CALIB_TRI

#define SW1_PIN	160
#define SW2_PIN	161
#define SW3_PIN	163
#define SW4_PIN	164
#define SENSOR	166

///////////////////////////////////////////////////////////////////////////////
enum StateMachine{
	SM_START = 0,
	SM_STOP,
	SM_RUN,
	SM_PAUSE, 
	SM_NONE
};

typedef struct {
	cv::Point x[11];
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
	double angRatioR;
	double throttleRatio;
	int mindl;
	int mindr;
}Parameters;
///////////////////////////////////////////////////////////////////////////////
cv::Point preMidLane(0,0);
///////////////////////////////////////////////////////////////////////////////
LCDI2C *lcd = new LCDI2C();
int preSTT_Machine = SM_NONE, STT_Machine =SM_NONE;
/// Init openNI ///
Status rc;
Device device;

VideoStream depth, color;
VideoFrameRef frame_depth, frame_color;
VideoStream* streams[] = {&depth, &color};

/// Init video writer and log files ///   
bool is_save_file = true; // set is_save_file = true if you want to log video and i2c pwm coeffs.
VideoWriter depth_videoWriter;  
VideoWriter color_videoWriter;
VideoWriter gray_videoWriter;

// string gray_filename = "gray.avi";
string color_filename = "color.avi";
string depth_filename = "depth.avi";

cv::Mat depthImg, colorImg, grayImage;

FILE *thetaLogFile; // File creates log of signal send to pwm control

int dir = 0, throttle_val = 0;
int set_throttle_val = 0;
double theta = 0;
int current_state = 0;

PCA9685 *pca9685 = new PCA9685() ;

bool is_show_cam = true;
int count_s,count_ss;
int frame_id = 0;
std::vector<cv::Vec4i> lines;

double avgThelta_L=0, avgThelta_R=0, avgXs_L=0, avgXs_R=0, avgXe_L=0, avgXe_R=0;

CenterPoint_t CenterPoint; 

double st = 0, et = 0, fps = 0;
double freq;

bool isCalibDone = false;
std::vector<cv::Point> Lp4_, Rp4_;
///////////////////////////////////////////////////////////////////////////////
cv::Mat remOutlier(const cv::Mat &gray) 
{
	int esize = 1;
	cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
		cv::Size( 2*esize + 1, 2*esize+1 ),
		cv::Point( esize, esize ) );
	cv::erode(gray, gray, element);
	std::vector< std::vector<cv::Point> > contours, polygons;
	std::vector< cv::Vec4i > hierarchy;
	cv::findContours(gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	for (size_t i = 0; i < contours.size(); ++i) {
		std::vector<cv::Point> p;
		cv::approxPolyDP(cv::Mat(contours[i]), p, 2, true);
		polygons.push_back(p);
	}
	cv::Mat poly = cv::Mat::zeros(gray.size(), CV_8UC3);
	for (size_t i = 0; i < polygons.size(); ++i) {
		cv::Scalar color = cv::Scalar(255, 255, 255);
		cv::drawContours(poly, polygons, i, color, CV_FILLED);
	}
	return poly;
}
char analyzeFrame(const VideoFrameRef& frame_depth,const VideoFrameRef& frame_color,Mat& depth_img, Mat& color_img) 
{
	DepthPixel* depth_img_data;
	RGB888Pixel* color_img_data;

	int w = frame_color.getWidth();
	int h = frame_color.getHeight();

	depth_img = Mat(h, w, CV_16U);
	color_img = Mat(h, w, CV_8UC3);

	depth_img_data = (DepthPixel*)frame_depth.getData();

	memcpy(depth_img.data, depth_img_data, h*w*sizeof(DepthPixel));

	color_img_data = (RGB888Pixel*)frame_color.getData();

	memcpy(color_img.data, color_img_data, h*w*sizeof(RGB888Pixel));

	cvtColor(color_img, color_img, COLOR_RGB2BGR);

	return 'c';
}

/// Return angle between veritcal line containing car and destination point in degree
double getTheta(Point car, Point dst) 
{
	if (dst.x == car.x) return 0;
	if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
	double pi = acos(-1.0);
	double dx = dst.x - car.x;
    double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}

bool exists_files (const std::string& name) 
{
	struct stat buffer;   
	return (stat (name.c_str(), &buffer) == 0); 
}

cv::Point ObstacleDectect(cv::Mat &depthImg, cv::Rect &intersect, cv::Rect &center_rect)
{
    cv::Point Obstacle_OffsetPoint;
    
    cv::Rect roi1(0, 132, 640, 238);
    std::vector< Rect > output_boxes;
    center_rect = Rect(320 - 200/2, 320, 200, 50);
    static int ObjectSide = 0;// 0:None, 1: Right, -1: Left 

    const float scaleFactor = 1.0f/257;
    depthImg.convertTo( depthImg, CV_8UC1, scaleFactor );
                    
    api_kinect_cv_get_obtacle_rect( depthImg, output_boxes, roi1, 40, 60);
    //Mat binImg = Mat::zeros(depthImg.size(), CV_8UC1);
                    
    //cv::rectangle( binImg, center_rect, Scalar( 128) );
                    
    for( int i = 0; i< output_boxes.size(); i++ )
    {
        //cout << "boxes size: " << output_boxes.size() << endl;
        intersect = output_boxes[i] & center_rect;
        if( intersect.area() != 0 )
        {   
            if (intersect.br().x <= 360)
            {
                //cv::rectangle( binImg, intersect, Scalar( 255) );
                cout<< endl<< "Object on the LEFT side " << flush;
                ObjectSide = -1;
            }
            else if (intersect.tl().x >= 360)
            {
                //cv::rectangle( binImg, intersect, Scalar( 255) );
                cout<< endl<< "Object on the RIGHT side " << flush;
                ObjectSide = 1;
            }
        }
        if (/*(intersect.br().x > 320) &&*/ (ObjectSide == -1))
        {
            ObjectSide = 0;
            Obstacle_OffsetPoint.x = 30;
            return Obstacle_OffsetPoint;
        }
        else if (/*(intersect.tl().x < 400) &&*/ (ObjectSide == 1))
        {
            ObjectSide = 0;
            Obstacle_OffsetPoint.x = -30;
            return Obstacle_OffsetPoint;
        }
        else
        {
        	Obstacle_OffsetPoint.x = 0;
        	return Obstacle_OffsetPoint;
        } 
    }
                    
    //if(!binImg.empty())
    //imshow( "BoundingRect", binImg );
}

cv::Point FilteredMidlanePoint(cv::Point pointMidLane, CenterPoint_t *CenterPoint)
{	
	cv::Point OffsetDistance;
	cv::Point PredictedPoint;
	cv::Point TempPoint;
	cv::Point FilteredPoint(0,0);
	static cv::Point prePointMidLane;

	for(int i = 0; i < 11; i++)
	{
		FilteredPoint += CenterPoint->x[i];
	}
	
	FilteredPoint /= 11;

	OffsetDistance = (CenterPoint->x[10] - CenterPoint->x[0])/10;
	PredictedPoint = FilteredPoint + OffsetDistance;

	//if ((pointMidLane.x > CenterPoint->MinRatioRange*PredictedPoint.x) && (pointMidLane.x < CenterPoint->MaxRatioRange*PredictedPoint.x))
	if ((pointMidLane.x > -(int)(CenterPoint->MinRatioRange) + PredictedPoint.x) && (pointMidLane.x < (int)(CenterPoint->MaxRatioRange) + PredictedPoint.x) && (pointMidLane.y != 0))
	{
		TempPoint = pointMidLane;
	}
	else TempPoint = PredictedPoint;

	FilteredPoint = FilteredPoint + CenterPoint->AlphaLPF*(TempPoint - FilteredPoint);

    //std::cout << pointMidLane << std::endl << PredictedPoint << std::endl << FilteredPoint << std::endl << std::endl;

	for(int i = 0; i < 10; i++)
	{
		CenterPoint->x[i] = CenterPoint->x[i+1];
	}
	CenterPoint->x[10] = FilteredPoint;

	prePointMidLane = pointMidLane;

	return FilteredPoint;
}

/*void TSR(std::vector<TrafficSign> *signs, std::vector<object_detector<image_scanner_type> > *detectors)
{
    OpenNI2::Instance()->getData(colorImg, depthImg1, grayImage, disparity);
    //cv::imshow("color", colorImg);
    //cv::imshow("depth", converted_depth);
    Mat colorTemp = colorImg.clone();
	std::vector<Rect> boxes;
	std::vector<int> labels;
	cv::Mat result;
		
	detection.objectLabeling(boxes, labels, depthImg1, colorImg, result, l_th, h_th, 1000, 4000, 36, 200, 1.2);
	if(result.empty())
	    cout<<"No object"<<endl;	
	else
	{  
		cout<<"Object detected"<<endl;
		imshow("result",result);
		Mat src;
	    src = result.clone();
	    resize(src,src,Size(80,80),INTER_LANCZOS4);// vi dieu
        //imshow("ROI",src);
           				
        cv_image<bgr_pixel> images_HOG(src);
        std::vector<rect_detection> rects;
        evaluate_detectors(detectors, images_HOG, rects, 0.9);
        cout<<"Rect Size: "<<rects.size()<<endl;
        if(rects.size() > 0)
        {
            cout << "Traffic Sign Name: " << signs[rects[0].weight_index].name <<": " << rects[0].detection_confidence << endl;
      	}
	}
}*/

bool SM_Start(cv::Rect roi, int key, bool reset) {
    //Open camera - depth and color image
	int readyStream = -1;
	rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
	if (rc != STATUS_OK)
	{
		printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
		return 0;
	}

	depth.readFrame(&frame_depth);
	color.readFrame(&frame_color);
	frame_id ++;

	char recordStatus = analyzeFrame(frame_depth,frame_color, depthImg, colorImg);
	//cv::imshow("depth_test1",depthImg);
	flip(depthImg, depthImg, 1);
	flip(colorImg, colorImg, 1);
	cv::Mat im_raw = colorImg(roi).clone();
    
    //cv::Point Obstacle_OffsetPoint;
    cv::Rect roi1(0, 132, 640, 238);
    std::vector< Rect > output_boxes;
    cv::Rect intersect;
    cv::Rect center_rect(320 - 200/2, 320, 200, 50);
    static int ObjectSide = 0;// 0:None, 1: Right, -1: Left 
    
		std::vector<Rect> boxes;
		std::vector<int> labels;
		cv::Mat result;
		
		detection.objectLabeling(boxes, labels, depthImg, colorImg, result, l_th, h_th, 0, 200, 3, 20, 1.2);//1000-4000-36-200-1.2
		if(result.empty())
		    cout<<"No object"<<endl;	
		else
		{  
			cout<<"Object detected"<<endl;
			cv:imshow("result",result);
        }
    
    //cvtColor(depthImg, depthImg, CV_RGB2GRAY);
    /*const float scaleFactor = 1.0f/257;
    depthImg.convertTo( depthImg, CV_8UC1, scaleFactor );
    cv::imshow("depth_test2",depthImg);
    api_kinect_cv_get_obtacle_rect( depthImg, output_boxes, roi1, 35, 65);
    Mat binImg = Mat::zeros(depthImg.size(), CV_8UC1);
    
    cv::rectangle( binImg, center_rect, Scalar( 128) );
    
    for( int i = 0; i< output_boxes.size(); i++ )
    {
        //cout << "boxes size: " << output_boxes.size() << endl;
        intersect = output_boxes[i] & center_rect;
        if( intersect.area() != 0 )
        {   
            if (intersect.br().x <= 360)
            {
                cv::rectangle( binImg, intersect, Scalar( 255) );
                ObjectSide = -1;
                std::cout<< std::endl<< "Object on the LEFT side: " << intersect.br().x << std::flush;
            }
            else if (intersect.tl().x >= 360)
            {
                cv::rectangle( binImg, intersect, Scalar( 255) );
                std::cout<< std::endl<< "Object on the RIGHT side: " << intersect.tl().x << std::flush;
            }
            cv::rectangle( binImg, intersect, Scalar( 255) );
            std::cout<< std::endl<< "Object Detected: " << intersect.br().x << " - " << intersect.tl().x << std::flush;
   
        }
    }
        /*if ((intersect.area() < xxx) && (ObjectSide == -1))
        {
            ObjectSide = 0;
            return theta;
        }
        else if ((intersect.area() < xxx) && (ObjectSide == 1))
        {
            ObjectSide = 0;
            return theta;
        }
    }
    
    if(!binImg.empty())
        imshow( "BoundingRect", binImg );*/
    //Obstacle_OffsetPoint = ObstacleDectect(depthImg);
    //cout << "Obstacle Offset Point: " << Obstacle_OffsetPoint << endl;
//TODO: Thanh
            ////////// Detect Center Point ////////////////////////////////////
	if (recordStatus == 'c') 
	{
		cv::Mat grayImage;
		cvtColor(im_raw, grayImage, CV_BGR2GRAY);

		cv::Mat dst = keepLanes_new(grayImage);
		// cv::imshow("dst", dst);

		cv::Mat lane;
		std::vector<cv::Point> Lp4corner, Rp4corner;

#ifdef CALIB_TRI
		int numContour;

		int a;
		a = calibLanes(dst.size(), dst, Lp4corner, Rp4corner, &numContour, key, reset);

		if((a == 0) || (a == 3))
			isCalibDone = true;
		if((key == 1) || (key == 2))
			isCalibDone = false;

		char buf[30];

		if(isCalibDone)
		{
			lcd->LCDSetCursor(0,1);      
			sprintf(buf,"READY");
			lcd->LCDPrintStr(buf);
		}
		else
		{
			lcd->LCDSetCursor(0,1);      
			sprintf(buf, "CALIB");
			lcd->LCDPrintStr(buf);
		}

		lcd->LCDSetCursor(15,1);      
		sprintf(buf, "CT: %1d", numContour);
		lcd->LCDPrintStr(buf);

		//cout << a << " " << Lp4corner << "" << Rp4corner << endl;

		if ((a == 0) || (a == 1) || (a == 3))
		{
			Lp4_ = Lp4corner;
			lcd->LCDSetCursor(0,2);
			sprintf(buf, "L: %3d %3d %3d %3d", Lp4corner[3].x, Lp4corner[3].y, Lp4corner[1].x, Lp4corner[1].y);
			lcd->LCDPrintStr(buf);
		}

		if ((a == 0) || (a == 2) || (a == 3))
		{  
			Rp4_ = Rp4corner;
			lcd->LCDSetCursor(0,3);
			sprintf(buf, "R: %3d %3d %3d %3d", Rp4corner[2].x, Rp4corner[2].y, Rp4corner[0].x, Rp4corner[0].y);
			lcd->LCDPrintStr(buf);
		}
                //else lcd->LCDPrintStr("R: -1 ");
#else
/*
		cv::Point MidLanePoint = twoLaneMostLanes_th(dst.size(), dst, lane, 
			Lp4corner, Rp4corner, false);

		char buf[30];
		lcd->LCDSetCursor(0,2);
		if (!Lp4corner.empty())
		{        
			sprintf(buf, "L: %3d", Lp4corner[3].x);
			lcd->LCDPrintStr(buf);
		}
		else lcd->LCDPrintStr("L: -1 ");

		lcd->LCDSetCursor(0,3);
		if (!Rp4corner.empty())
		{  
			sprintf(buf, "R: %3d", Rp4corner[2].x);
			lcd->LCDPrintStr(buf);
		}
		else lcd->LCDPrintStr("R: -1 ");

		if (MidLanePoint.x == -2)
		{
			lcd->LCDSetCursor(0,1);
			lcd->LCDPrintStr("ERR: LEFT Y ");
			std::cout << "ERR: LEFT Y" << std::endl;
		}
		else if (MidLanePoint.x == -3)
		{
			lcd->LCDSetCursor(0,1);
			lcd->LCDPrintStr("ERR: LEFT X ");
			std::cout << "ERR: LEFT X" <<std::endl;
		}
		else if (MidLanePoint.x == -4)
		{
			lcd->LCDSetCursor(0,1);
			lcd->LCDPrintStr("ERR: RIGHT Y");
			std::cout << "ERR: RIGHT Y" <<std::endl;
		}
		else if (MidLanePoint.x == -5)
		{
			lcd->LCDSetCursor(0,1);
			lcd->LCDPrintStr("ERR: RIGHT X");
			std::cout << "ERR: RIGHT X" <<std::endl;
		}
		else if (MidLanePoint.x == -6)
		{
			isCalibDone = true;
			lcd->LCDSetCursor(0,1);
			lcd->LCDPrintStr("CALIB DONE !");
			std::cout << "CALIB DONE!" << std::endl;

			Lp4_ = Lp4corner;
			Rp4_ = Rp4corner;
		}
		*/
#endif
	}
}

void SM_Stop(void) {

}

bool SM_Run(Parameters par) 
{
	cv::Point carPosition = cv::Point(par.roi.width/2 + par.car_offset, par.roi.height-1);
    //Check PCA9685 driver ////////////////////////////////////////////
	if (pca9685->error < 0)
	{
		cout<< endl<< "Error: PWM driver"<< endl<< flush;
		return 0;
	}
    //Open camera - depth and color image
	int readyStream = -1;
	rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
	if (rc != STATUS_OK)
	{
		printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
		return 0;
	}

	depth.readFrame(&frame_depth);
	color.readFrame(&frame_color);
	frame_id ++;
	char recordStatus = analyzeFrame(frame_depth,frame_color, depthImg, colorImg);
	flip(depthImg, depthImg, 1);
	flip(colorImg, colorImg, 1);

//TODO: Thanh
            ////////// Detect Center Point ////////////////////////////////////
	if (recordStatus == 'c') 
	{
		cv::Mat im_raw = colorImg(par.roi).clone();
		cv::Mat grayImage;
		cvtColor(im_raw, grayImage, CV_BGR2GRAY);

		cv::Mat dst = keepLanes_new(grayImage);
		// cv::imshow("dst", dst);

		cv::Mat lane;

		cv::Point pointMidLane = twoLaneMostLanes_th(dst.size(), dst, lane, par.yCalc, par.point_offset, par.mindl, par.mindr, Lp4_, Rp4_);

		cv::Point FilteredMidlanePoint1 = FilteredMidlanePoint(pointMidLane, &CenterPoint);

		///////////////////////////////////////////////////////////////////////////////////////////////
		static int Obstacle_FrameCnt = 0;
		static bool Obstacle_Detected = true;
		static cv::Point Obstacle_HoldingPoint;
		cv::Rect intersect, center;
		cv::Point Obstacle_OffsetPoint = ObstacleDectect(depthImg, intersect, center);
		
		if (Obstacle_Detected == true)
		{
			if ((Obstacle_OffsetPoint.x != 0) && (Obstacle_FrameCnt != 5))
			{
				FilteredMidlanePoint1 += Obstacle_OffsetPoint;
				Obstacle_FrameCnt++;
				Obstacle_Detected = true;	
				Obstacle_HoldingPoint = FilteredMidlanePoint1;
			}	
			if (Obstacle_FrameCnt == 5)
			{
				FilteredMidlanePoint1 = Obstacle_HoldingPoint;
			}
		}
		else 
		{
			Obstacle_Detected = false;
			Obstacle_FrameCnt = 0;
		}
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//lcd->LCDSetCursor(0,2);
		if (pointMidLane.y == 0)
		{
			std::cout<< "No Lane Detect" << std::endl;
			//lcd->LCDPrintStr("No Lane  ");
		}
		else
			//lcd->LCDPrintStr("Have Lane");

        lcd->LCDSetCursor(0,2);
        if( intersect.area() != 0 )
        {
            lcd->LCDPrintStr("Have XXX");
        }
        else
        {
            lcd->LCDPrintStr("No XXX  ");
        }

		circle(im_raw, pointMidLane, 2, Scalar(255,0,0), 5);
		circle(im_raw, FilteredMidlanePoint1, 2, Scalar(0,255,0), 5);
		circle(im_raw, carPosition, 2, Scalar(0,0,255), 5);
		circle(im_raw, Lp4_[1], 2, Scalar(0,255,255), 5);
		circle(im_raw, Lp4_[3], 2, Scalar(0,255,255), 5);
		circle(im_raw, Rp4_[0], 2, Scalar(0,255,255), 5);
		circle(im_raw, Rp4_[2], 2, Scalar(0,255,255), 5);
		cv::rectangle(colorImg, intersect, Scalar(0, 0, 255));
		//cv::rectangle(colorImg, center, Scalar(0, 0, 200));
		// imshow("raw", im_raw);
        // cv::imshow("two", two);
        //////////////////////////end edited////////////////////////////
		double angDiff = getTheta(carPosition, FilteredMidlanePoint1);
        //if(-10<angDiff&&angDiff<10) angDiff=0;
		if (angDiff < -45)
			theta = (angDiff*par.angRatioL1);
		else if (angDiff < 0)
			theta = (angDiff*par.angRatioL2);
		else theta = angDiff*par.angRatioR;
        //std::cout<<"angdiff "<<angDiff<<std::endl;

		if((theta > 75) || (theta < -75))
		{
			throttle_val = set_throttle_val*par.throttleRatio;
			if(throttle_val < 30)
				throttle_val = 30;
		}
		else
		{
			throttle_val = set_throttle_val;
		}
       // theta = (0.00);
		api_set_STEERING_control(pca9685,theta);

		if(is_save_file)
		{

			if (pointMidLane.y == 0)
			{
				fprintf(thetaLogFile, "No Lane Detect\n");
			}

			fprintf(thetaLogFile, "%d\n", frame_id);
			fprintf(thetaLogFile, "pwm: %d\n", throttle_val);
			fprintf(thetaLogFile, "theta: %f\n", theta);
			fprintf(thetaLogFile, "Mid lane Point: x = %d , y = %d\n\n", pointMidLane.x, pointMidLane.y ); 

			im_raw.copyTo(colorImg(Rect(0, 0, im_raw.cols, im_raw.rows)));
			Mat lane_s;
			cvtColor(lane, lane_s, CV_GRAY2BGR); 
			lane_s.copyTo(colorImg(Rect(0, im_raw.rows, lane_s.cols, lane_s.rows)));
			//Mat dst_s;
			//cvtColor(dst, dst_s, CV_GRAY2BGR); 
			//dst_s.copyTo(colorImg(Rect(0, im_raw.rows + lane_s.rows, dst_s.cols, dst_s.rows)));
			char buf[100];
			sprintf(buf, "Ang: %.1f - Theta: %d - Mid [%d, %d] - Filter [%d, %d]", angDiff, throttle_val, pointMidLane.x, pointMidLane.y, FilteredMidlanePoint1.x, FilteredMidlanePoint1.y);
			putText(colorImg, buf, cvPoint(10, 30), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255), 1, CV_AA);
			// imshow("raw", colorImg);
			if (!colorImg.empty())
				color_videoWriter.write(colorImg);
			char buf1[30];
			lcd->LCDSetCursor(0,1);
			sprintf(buf1, "A: %3.1f T: %2d", angDiff, throttle_val);
			lcd->LCDPrintStr(buf1);
		}
	}

	int pwm2 =  api_set_FORWARD_control( pca9685,throttle_val);
	et = getTickCount();
	fps = 1.0 / ((et-st)/freq);
	cerr << "FPS: "<< fps<< '\n';

	char buf1[30];

	lcd->LCDSetCursor(0,3);
	sprintf(buf1, "FPS: %2.1f", fps);
	lcd->LCDPrintStr(buf1);

    //Save file
	if (recordStatus == 'c' && is_save_file) 
	{
		// 'Center': target point
		// pwm2: STEERING coefficient that pwm at channel 2 (our steering wheel's channel)
		// fprintf(thetaLogFile, "Center: [%d, %d]\n", center_point.x, center_point.y);
		// fprintf(thetaLogFile, "pwm2: %d\n", pwm2);

		//if (!im_raw.empty())
		//    color_videoWriter.write(im_raw);

		if (!depthImg.empty())
		{
			Mat dep_img;
			cvtColor(depthImg, dep_img, CV_GRAY2BGR);               
            depth_videoWriter.write(dep_img);
		}
        // if (!grayImage.empty())
        //  gray_videoWriter.write(grayImage); 
	}
	if (recordStatus == 'd' && is_save_file) 
	{
        //if (!depthImg.empty())
            //depth_videoWriter.write(depthImg);
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

///////// utilitie functions  ///////////////////////////

int main( int argc, char* argv[] ) 
{
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

	if(value.size() == 16)
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
		par.angRatioR = value[12];
		par.throttleRatio = value[13];
		par.mindl = value[14];
		par.mindr = value[15];
		
		cout << par.roi << endl;
		cout << par.car_offset << endl;
		cout << par.yCalc << endl;
		cout << par.point_offset << endl;
		cout << par.MinRatioRange << endl;
		cout << par.MaxRatioRange << endl;
		cout << par.AlphaLPF << endl;
		cout << par.angRatioL1 << endl;
		cout << par.angRatioL2 << endl;
		cout << par.angRatioR << endl;
		cout << par.throttleRatio << endl;
		cout << par.mindl << endl;
		cout << par.mindr << endl;
		cout << endl << "Loaded all parameter" << endl << endl;
	}
	else 
	{
	    cout << "Missing Parameters" << endl;
	    return 0;
	}
//==============================================Uy=================================================//	
    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
    std::vector<object_detector<image_scanner_type> > detectors;
    std::vector<TrafficSign> signs;
    
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


	cout << "Loading Template images..." << endl;

	std::vector<Mat> tmp_img;

	Mat max_spd = imread("template/0.jpg");
    // resize(max_spd,max_spd,Size(80,80),INTER_LINEAR);
	tmp_img.push_back(max_spd);

	Mat min_spd = imread("template/2.jpg");
    // resize(min_spd,min_spd,Size(80,80),INTER_LINEAR);
	tmp_img.push_back(min_spd);

	Mat min_end = imread("template/1.jpg");
    // resize(min_end,min_end,Size(80,80),INTER_LINEAR);
	tmp_img.push_back(min_end);

	Mat no_ent = imread("template/4.jpg");
    // resize(no_ent,no_ent,Size(80,80),INTER_LINEAR);
	tmp_img.push_back(no_ent);

	Mat no_left = imread("template/5.jpg");
    // resize(no_left,no_left,Size(80,80),INTER_LINEAR);
	tmp_img.push_back(no_left);

	Mat stp_ga = imread("template/7.jpg");
    // resize(stp_ga,stp_ga,Size(80,80),INTER_LINEAR);
	tmp_img.push_back(stp_ga);

	Mat trn_left = imread("template/8.jpg");
    // resize(trn_left,trn_left,Size(80,80),INTER_LINEAR);
	tmp_img.push_back(trn_left);

	Mat trn_right = imread("template/9.jpg");
	tmp_img.push_back(trn_right);
	
	//OpenNI2::Instance() -> init();
	
//=================================================================================================//	

    //Peripherial inital
	GPIO *gpio = new GPIO();
	I2C *i2c_device = new I2C();
	
	int sw1_stat = 1;
	int sw2_stat = 1;
	int sw3_stat = 1;
	int sw4_stat = 1;
	int sensor = 0;
	
    // Setup input
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
    //init LCD//
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

/// Init openNI ///

	rc = OpenNI::initialize();
	if (rc != STATUS_OK) {
		printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
		return 0;
	}
	rc = device.open(ANY_DEVICE);
	if (rc != STATUS_OK) {
		printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
		return 0;
	}
	if (device.getSensorInfo(SENSOR_DEPTH) != NULL) {
		rc = depth.create(device, SENSOR_DEPTH);
		if (rc == STATUS_OK) {
			VideoMode depth_mode = depth.getVideoMode();
			depth_mode.setFps(30);
			depth_mode.setResolution(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT);
			depth_mode.setPixelFormat(PIXEL_FORMAT_DEPTH_100_UM);
			depth.setVideoMode(depth_mode);

			rc = depth.start();
			if (rc != STATUS_OK) {
				printf("Couldn't start the color stream\n%s\n", OpenNI::getExtendedError());
			}
		}
		else {
			printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
		}
	}

	if (device.getSensorInfo(SENSOR_COLOR) != NULL) {
		rc = color.create(device, SENSOR_COLOR);
		if (rc == STATUS_OK) {
			VideoMode color_mode = color.getVideoMode();
			color_mode.setFps(30);
			color_mode.setResolution(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT);
			color_mode.setPixelFormat(PIXEL_FORMAT_RGB888);
			color.setVideoMode(color_mode);

			rc = color.start();
			if (rc != STATUS_OK)
			{
				printf("Couldn't start the color stream\n%s\n", OpenNI::getExtendedError());
			}
		}
		else {
			printf("Couldn't create color stream\n%s\n", OpenNI::getExtendedError());
		}
	}

/// End of openNI init phase ///


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


	if(is_save_file) {
        // gray_videoWriter.open(gray_filename, codec, 8, output_size, false);
		color_videoWriter.open(color_filename, codec, 8, output_size, true);
    	depth_videoWriter.open(depth_filename, CV_FOURCC('M','J','P', 'G'), 8, output_size, true);
		thetaLogFile = fopen("thetaLog.txt", "w");
	}
	/// End of init logs phase ///

	char key = 0;

    //=========== Init  =======================================================

    ////////  Init PCA9685 driver   ///////////////////////////////////////////
	api_pwm_pca9685_init( pca9685 );

	if (pca9685->error >= 0)
       // api_pwm_set_control( pca9685, dir, throttle_val, theta, current_state );
		api_set_FORWARD_control( pca9685,throttle_val);
    /////////  Init UART here   ///////////////////////////////////////////////
    /// Init MSAC vanishing point library
	MSAC msac;
	cv::Rect roi1 = cv::Rect(0, VIDEO_FRAME_HEIGHT*3/4,
		VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT/4);

	api_vanishing_point_init( msac );

    ////////  Init direction and ESC speed  ///////////////////////////

	throttle_val = 0;
	theta = 0;

    // Argc == 2 eg ./test-autocar 27 means initial throttle is 27
	if(argc == 2 ) set_throttle_val = atoi(argv[1]);
	fprintf(stderr, "Initial throttle: %d\n", set_throttle_val);
	int frame_width = VIDEO_FRAME_WIDTH;
	int frame_height = VIDEO_FRAME_HEIGHT;

    //bool running = false, started = false, stopped = false, prestt = false;

	CenterPoint.x[0] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[1] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[2] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[3] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[4] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc); 
	CenterPoint.x[6] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[5] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);	
	CenterPoint.x[7] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[8] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[9] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.x[10] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
	CenterPoint.MinRatioRange = par.MinRatioRange; //150;
	CenterPoint.MaxRatioRange = par.MaxRatioRange; //150;
	CenterPoint.AlphaLPF = par.AlphaLPF; //0.9;

	freq = getTickFrequency();

	lcd->LCDSetCursor(4,3);
	lcd->LCDPrintStr("WAITING ....");

	while ( true )
	{
		Point center_point(0,0);

		st = getTickCount();
//==============================================Uy=================================================//
        //OpenNI2::Instance()->getData(colorImg, depthImg, grayImage, disparity);
        //cv::imshow("color", colorImg);
        //cv::imshow("depth", converted_depth);
        
        //Open camera - depth and color image
/*	    int readyStream = -1;
	    rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
	    if (rc != STATUS_OK)
	    {
	    	printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
	    	return 0;
	    }

	    depth.readFrame(&frame_depth);
	    color.readFrame(&frame_color);
	    frame_id ++;

	    char recordStatus = analyzeFrame(frame_depth,frame_color, depthImg, colorImg);
	    cv::imshow("depth_test",depthImg);
		Mat colorTemp = colorImg.clone();
		std::vector<Rect> boxes;
		std::vector<int> labels;
		cv::Mat result;
		
		detection.objectLabeling(boxes, labels, depthImg, colorImg, result, l_th, h_th, 1000, 4000, 36, 200, 1.2);
		if(result.empty())
		    cout<<"No object"<<endl;	
		else
		{  
			cout<<"Object detected"<<endl;
			imshow("result",result);
			Mat src;
		    src = result.clone();
		    resize(src,src,Size(80,80),INTER_LANCZOS4);// vi dieu
            //imshow("ROI",src);
            				
            cv_image<bgr_pixel> images_HOG(src);
            std::vector<rect_detection> rects;

            evaluate_detectors(detectors, images_HOG, rects, 0.9);
            cout<<"Rect Size: "<<rects.size()<<endl;
            if(rects.size() > 0)
            {
                cout << "Traffic Sign Name: " << signs[rects[0].weight_index].name <<": " << rects[0].detection_confidence << endl;
        	}
		}
        waitKey(10);*/
//=================================================================================================//	    
		key = getkey();
		unsigned int bt_status = 0;

        //Thu tu nut nhan - X(BT4) X(BT3) D(BT2) D(BT1)
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
                //Exec
				lcd->LCDClear();
				lcd->LCDSetCursor(7,0);
				lcd->LCDPrintStr("STOP");
				fprintf(stderr, "End process.\n");
				theta = 0;
				throttle_val = 0;
				api_set_FORWARD_control( pca9685,throttle_val);
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
						SM_Start(par.roi, 2, true);
					}
					else
						SM_Start(par.roi, 2, false);
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
						SM_Start(par.roi, 1, true);
					}
					else
						SM_Start(par.roi, 1, false);
				}
				else if(STT_Machine == SM_PAUSE)
				{
					SM_Start(par.roi, 0, true);
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

				if((STT_Machine == SM_START) && isCalibDone)
				{
					STT_Machine = SM_RUN;
					fprintf(stderr, "Run.\n");
	                //Exec
					lcd->LCDClear();
					lcd->LCDSetCursor(7,0);
					lcd->LCDPrintStr("RUNNING");
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

		gpio->gpioGetValue(SENSOR, &bt_status);
		if (!bt_status) 
		{
			if (bt_status != sensor) 
			{
                //Status
				sensor = bt_status;
			}
		} 
		else sensor = bt_status;

    	//cout << "IR: " << sw4_stat << endl;

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
			api_set_FORWARD_control( pca9685,throttle_val);
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
			api_set_FORWARD_control( pca9685,throttle_val);
			break;
		}

//TODO:Tri
		
		//State Machine

		switch(STT_Machine)
		{
			case SM_START:
			if(preSTT_Machine == SM_PAUSE)
			{
				SM_Start(par.roi, 0, true);
			}
			else
				SM_Start(par.roi, 0, false);
			break;
			case SM_STOP:
			SM_Stop();
			if(is_save_file)
			{
					// gray_videoWriter.release();
				color_videoWriter.release();
			    depth_videoWriter.release();
				fclose(thetaLogFile);
			}
			return 0;
			break;
			case SM_RUN:
			SM_Run(par);
			break;
			case SM_PAUSE:
			SM_Pause();
			CenterPoint.x[0] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[1] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[2] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[3] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[4] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc); 
			CenterPoint.x[6] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[5] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);	
			CenterPoint.x[7] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[8] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[9] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
			CenterPoint.x[10] = cv::Point(par.roi.width/2 + par.car_offset, par.yCalc);
                    //preMidLane = cv::Point(0,0);
			isCalibDone = false;
			break;
			default:

			break;
		}
		if(is_show_cam)
			waitKey(10);
	}
        //////////  Release //////////////////////////////////////////////////////
	if(is_save_file)
	{
        // gray_videoWriter.release();
		color_videoWriter.release();
    	depth_videoWriter.release();
        // fclose(thetaLogFile);
	}
	return 0;
}
