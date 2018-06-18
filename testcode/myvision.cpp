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
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace dlib;
using namespace cv;

const string nameWindow1 = "Original Image";
const string nameWindow2 = "Trackars";
const string nameWindow3 = "Blurred Image";
const string nameWindow4 = "EDGED Image";
const string nameWindow5 = "InRange Image";
const string nameWindow6 = "Result Image";
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "api_kinect_cv.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "openni2.h"
#include "../openni2/Singleton.h"
#include <unistd.h>
#include "../sign_detection/SignDetection.h"
#include <chrono>
//#include "signsRecognizer.h"
#include "extractInfo.h"
#include <stdlib.h>
#include "multilane.h"
#include "Hal.h"
#include "LCDI2C.h"
#include "api_i2c_pwm.h"

#include "../ObjectRecognition/SignRecognition.h"
using namespace openni;
using namespace framework;
using namespace signDetection;
using namespace EmbeddedFramework;
//using namespace SignRecognition;
#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms

#undef debug
#define debug false
#define SW1_PIN	160
#define SW2_PIN	161
#define SW3_PIN	163
#define SW4_PIN	164
#define SENSOR	165
#define LED		166
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int B_H_val = 100;
int B_H_max = 150;
int B_S_val = 80;
int B_S_max = 255;
int B_V_val = 110;
int B_V_max = 255;

int R_H_val = 150;
int R_H_max = 255;
int R_S_val = 50;
int R_S_max = 255;
int R_V_val = 50;
int R_V_max = 255;

int R2_H_val = 0;
int R2_H_max = 5;
int R2_S_val = 90;
int R2_S_max = 255;
int R2_V_val = 50;
int R2_V_max = 255;

int K_H_val = 105;
int K_H_max = 255;
int K_S_val = 40;
int K_S_max = 255;
int K_V_val = 30;
int K_V_max = 100;

int C_ths_val = 0;
int C_ths_max = 255;

RNG rng(12345);

bool bpause = false;
bool capture = false;

int ccpt = 0;
int croi = 0;

VideoWriter video;
	// Scalar red_min  = Scalar(0,36,0);
	// Scalar red_max = Scalar(25, 255, 255);

struct TrafficSign 
{
	string name;
	string svm_path;
	rgb_pixel color;
	TrafficSign(string name, string svm_path, rgb_pixel color) :
	name(name), svm_path(svm_path), color(color) {};
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if  ( event == EVENT_MBUTTONDOWN )
	{
		bpause ^= true;
	}
	else if(event== EVENT_LBUTTONDOWN)
	{
		capture = true;
	}
}

void CalcHistogram(Mat Image, int * Result_)
{
	int i = 0, j = 0, idx = 0, jdx = 0;
	int count = 0;
	for(idx = 0; idx < 4; idx++)
	{
		for(jdx = 0; jdx < 4; jdx++)
		{
			count = 0;
			for(i = 20*idx; i < (idx + 1)*20; i++)
			{
				for(j = 20*jdx; j < (jdx + 1)*20; j++)
				{
                    // cout << i << " " << j << " " << (int)Image.at<uchar>(i,j) << endl;
					if((int)Image.at<uchar>(i,j) > 128)
					{
						count++;
					}
				}
			}
			Result_[idx*4 + jdx] = count;
            // cout << idx*4 + jdx << " " << Result_[idx*4 + jdx]  << endl;
		}
	}
}

void CalcPercent(int * Src1, int * Src2, float * Kernel ,float * Des)
{
	float Des_[16];
	float Result_ = 0;
	int count_zero = 0;
	for (int i = 0; i < 16; i++)
	{
		if(Src2[i] == 0)
		{
			if(Src1[i] < 100)
			{
				count_zero++;
			}
			
			continue;
		}
		Des_[i] = Kernel[i] * (float)Src1[i] / Src2[i];
		Result_ += Des_[i];
        // cout << "Des_[" << i <<"] " << Des_[i] <<  " Result " << Result_ << endl;
	}
	*Des = Result_ / (16 - count_zero);
    // cout << "Result_ " << Result_ << " - Percent " << *Des << endl;
    // Des[0] = Kernel[0] * (float)Src1[0] / Src2[0];
    // Des[1] = Kernel[1] * (float)Src1[1] / Src2[1];
    // Des[2] = Kernel[2] * (float)Src1[2] / Src2[2];
    // Des[3] = Kernel[3] * (float)Src1[3] / Src2[3];
    // Des[4] = Kernel[4] * (float)Src1[4] / Src2[4];
    // Des[5] = Kernel[5] * (float)Src1[5] / Src2[5];
    // Des[6] = Kernel[6] * (float)Src1[6] / Src2[6];
    // Des[7] = Kernel[7] * (float)Src1[7] / Src2[7];
    // Des[8] = Kernel[8] * (float)Src1[8] / Src2[8];
    // Des[9] = Kernel[9] * (float)Src1[9] / Src2[9];
    // Des[10] = Kernel[10] * (float)Src1[10] / Src2[10];
    // Des[11] = Kernel[11] * (float)Src1[11] / Src2[11];
    // Des[12] = Kernel[12] * (float)Src1[12] / Src2[12];
    // Des[13] = Kernel[13] * (float)Src1[13] / Src2[13];
    // Des[14] = Kernel[14] * (float)Src1[14] / Src2[14];
    // Des[15] = Kernel[15] * (float)Src1[15] / Src2[15];
    // Des[16] = Kernel[16] * (float)Src1[16] / Src2[16];
    // Des[17] = Kernel[17] * (float)Src1[17] / Src2[17];
    // Des[18] = Kernel[18] * (float)Src1[18] / Src2[18];
    // Des[19] = Kernel[19] * (float)Src1[19] / Src2[19];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat remOutlier(const cv::Mat &gray) {
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
double getTheta(Point car, Point dst) {
    if (dst.x == car.x) return 0;
    if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
    double pi = acos(-1.0);
    double dx = dst.x - car.x;
    double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}
///////// utilitie functions  ///////////////////////////
SignRecognition::SignRecognize a;
signDetection::SignDetection detection;


/************************************************************* MAIN PROGRAM *********************************************************************************/ 
int main( int argc, char* argv[] ) 
{
	////// init videostream ///
	GPIO *gpio = new GPIO();
	I2C *i2c_device = new I2C();
	LCDI2C *lcd = new LCDI2C();
    	int sw1_stat = 1;
	int sw2_stat = 1;
	int sw3_stat = 1;
	int sw4_stat = 1;
	int sensor = 1;
	
	// Setup input
	gpio->gpioExport(SW1_PIN);
	gpio->gpioExport(SW2_PIN);
	gpio->gpioExport(SW3_PIN);
	gpio->gpioExport(SW4_PIN);
	gpio->gpioExport(SENSOR);
	gpio->gpioExport(LED);
	gpio->gpioSetDirection(SW1_PIN, INPUT);
	gpio->gpioSetDirection(SW2_PIN, INPUT);
	gpio->gpioSetDirection(SW3_PIN, INPUT);
	gpio->gpioSetDirection(SW4_PIN, INPUT);
	gpio->gpioSetDirection(SENSOR, INPUT);
	gpio->gpioSetDirection(LED, OUTPUT);
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
	lcd->LCDCursorOn();
	
	lcd->LCDSetCursor(3,1);
	lcd->LCDPrintStr("DRIVERLESS CAR");
	lcd->LCDSetCursor(5,2);
	lcd->LCDPrintStr("2017-2018");
	int dir = 0, throttle_val = 0;
	double theta = 0;
	int current_state = 0;
	char key = 0;

	//=========== Init  =======================================================
	////////  Init PCA9685 driver   ///////////////////////////////////////////
	PCA9685 *pca9685 = new PCA9685() ;
	api_pwm_pca9685_init( pca9685 );
	if (pca9685->error >= 0)api_set_FORWARD_control( pca9685,throttle_val);
	/// Init MSAC vanishing point library
	MSAC msac;
	api_vanishing_point_init( msac );
	int set_throttle_val = 0;
    	throttle_val = 0;
   	theta = 0;
    	if(argc == 2 ) set_throttle_val = atoi(argv[1]);
   	fprintf(stderr, "Initial throttle: %d\n", set_throttle_val);
   	int frame_width = VIDEO_FRAME_WIDTH;
   	int frame_height = VIDEO_FRAME_HEIGHT;
   	Point carPosition(frame_width / 2, frame_height);
  	Point prvPosition = carPosition;
	bool running = true, started = false, stopped = false;
	OpenNI2::Instance() -> init();
	//signsRecognizer recognizer = signsRecognizer("/home/ubuntu/data/new_templates/templates.txt");
	ushort l_th = 600, h_th = 2200;//old: 600-2000
	std::vector<std::vector<Point> > regs;
	Mat depthImg, colorImg, grayImage, disparity;

	/*****************************************************************************************************************************************************/
	cout<<"test";
	int img_count = 0;
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
	std::vector<object_detector<image_scanner_type> > detectors;

	cout << "Loading SVM detectors..." << endl;
	std::vector<TrafficSign> signs;

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
    // resize(trn_right,trn_right,Size(80,80),INTER_LINEAR);
	tmp_img.push_back(trn_right);

	namedWindow( nameWindow6 , WINDOW_NORMAL );
	setMouseCallback(nameWindow6, CallBackFunc, NULL);
	double ttime = 0;

	/**************************************************************************************************************************************************/
    	while ( true )
    	{	
		Point center_point(0,0);
		key = getkey();
       		unsigned int bt_status = 0;
		unsigned int sensor_status = 0;
		gpio->gpioGetValue(SW4_PIN, &bt_status);
		gpio->gpioGetValue(SENSOR, &sensor_status);
		//std::cout<<sensor_status<<std::endl;
		if (!bt_status) {
			if (bt_status != sw4_stat) {
				running = !running;
				sw4_stat = bt_status;
				throttle_val = set_throttle_val;
			}
		} else sw4_stat = bt_status;
	
        if( key == 's') {
			running = !running;
			throttle_val = set_throttle_val;
			
		}
       	if( key == 'f') {
			fprintf(stderr, "End process.\n");
        	theta = 0;
        	throttle_val = 0;
	    	api_set_FORWARD_control( pca9685,throttle_val);
        	break;
		}
		if( !running )
		{
		lcd->LCDClear();
		lcd->LCDSetCursor(3,1);
		lcd->LCDPrintStr("PAUSE");
		continue;
		}
		if( running ){
		lcd->LCDClear();
		lcd->LCDSetCursor(3,1);
		lcd->LCDPrintStr("RUNNING");
		
		if (!sensor_status) {
			if (sensor_status != sensor) {
				running = !running;
				sensor = sensor_status;
				throttle_val = 0;
			}
		} else sensor = sensor_status;
			if (pca9685->error < 0)
           	 {
                cout<< endl<< "Error: PWM driver"<< endl<< flush;
                break;
           	 }
			if (!started)
			{
    			fprintf(stderr, "ON\n");
			    started = true; stopped = false;
				throttle_val = set_throttle_val;
                api_set_FORWARD_control( pca9685,throttle_val);
			}
        auto st = chrono::high_resolution_clock::now();
		OpenNI2::Instance()->getData(colorImg, depthImg, grayImage, disparity);
		cv::imshow("color_clone", colorImg);
		//cv::imshow("c_depth", depthImg/30);
		Mat colorTemp = colorImg.clone();
		std::vector<Rect> boxes;
		std::vector<int> labels;
		cv::Mat object_result, img_HSV;
		cv::Mat pyrDown;
		detection.objectLabeling(boxes, labels, depthImg, colorImg, object_result, l_th, h_th, 1000, 8000, 50, 200, 1.5);
		/***********************************************************************************************************************************************/
		
		if(object_result.empty())
		cout<<"No object"<<endl;
		else
		// else cout<<"boxes size"<< boxes.size() <<endl;
		// for(size_t i=0;i<boxes.size();i++)
		// {	
		// 	cout<<"boxes"<<boxes[i] << endl;
		// Mat im_raw = colorImg(boxes[i]).clone();
		// Mat im_raw = result.clone();
		{
			cout<<"Object detected"<<endl;
			imshow("object_result_raw",object_result);
		}
		Mat img_result = object_result.clone();
		cvtColor(object_result, img_HSV, COLOR_BGR2HSV);

		img_result.convertTo(img_result, CV_8UC3);

        std::vector<std::vector<cv::Point> > contours;
        std::vector<Vec4i> hierarchy;

    	imshow("object_result", object_result);

        findContours(object_result, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        std::vector<Rect> boundRect( contours.size() );

        for(int idx = 0; idx < contours.size(); idx++)
        {
        	boundRect[idx] = boundingRect( Mat(contours[idx]) );
        	if ((boundRect[idx].width*boundRect[idx].height > 900) && (boundRect[idx].width*boundRect[idx].height < 50000))
        	{
        		if((boundRect[idx].tl().x != 0) && (boundRect[idx].tl().y != 0) && (boundRect[idx].br().x != result.cols) && (boundRect[idx].br().y != result.rows))
        		{
        			if ( ( (float)boundRect[idx].width/boundRect[idx].height > 0.5) && ( (float)boundRect[idx].width/boundRect[idx].height < 1.3 ) )
        			{
        				// drawContours( imgLaplacian, contours, idx, 255, CV_FILLED, 8, hierarchy );
        				Mat src = result(boundRect[idx]).clone();
 								
        				resize(src,src,Size(80,80),INTER_LANCZOS4);
                        // imshow("ROI",src);
        				croi++;
	                    // imwrite( "capture/"+ string(parser[1]) + "_" + to_string(ccpt++) + ".jpg" , image_roi );
        				cv_image<bgr_pixel> images_HOG(src);
        				std::vector<rect_detection> rects;

        				evaluate_detectors(detectors, images_HOG, rects, 0.9);

        				if(rects.size() > 0)
        				{

    						cout << "   				" << signs[rects[0].weight_index].name <<": " << rects[0].detection_confidence << endl;
    						cv::rectangle( img_result, boundRect[idx].tl(), boundRect[idx].br(), Scalar(255,0,0), 2, 8, 0 );
    						putText(img_result, signs[rects[0].weight_index].name, Point(boundRect[idx].br().x, boundRect[idx].tl().y), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255), 1, CV_AA);
    						save_template.push_back(rects[0].weight_index);
    					
        				}
        				else cv::rectangle( img_result, boundRect[idx].tl(), boundRect[idx].br(), Scalar(0,255,0), 2, 8, 0 );
                                            // waitKey(0);
        			}
        		}
        	}
        }

        for (int icout = 0; icout < save_template.size();icout++)
        {
        	tmp_img[save_template[icout]].copyTo(img_result(Rect(80*icout,400,tmp_img[save_template[icout]].cols,tmp_img[save_template[icout]].rows)));
        }

        namedWindow( nameWindow6 , WINDOW_NORMAL );
        imshow(nameWindow6,img_result);
		// vector<int> compression_params;
   		// compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    	// 	compression_params.push_back(9);
		// imwrite("alpha"+std::to_string(img_count)+".png", im_raw, compression_params);
		// namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        // 	imshow( "Display window", im_raw );              // Show our image inside it.
        // 	clock_t begin_time = clock();
        // 	int ans = a.recognizeSign(im_raw);
        //     	//std::cerr << 1 + (ans - 1) / 5  << '\n';
        // 	cout<< "Loai bien bao " << ans  << ' ' << float( clock () - begin_time ) /  CLOCKS_PER_SEC << '\n';  
		//waitKey(0);
		// }
		/***************************************************************************************************************************************/
		cv::pyrDown( colorTemp, pyrDown, cv::Size(colorTemp.cols/2, colorTemp.rows/2));
		cv::Rect roi1 = cv::Rect(0, 240*3/4, 320, 240/4);
		cvtColor(pyrDown, grayImage, CV_BGR2GRAY);
           	cv::Mat dst = keepLanes(grayImage, false);
            	//cv::imshow("dst", dst);
            	cv::Point shift (0, 3 * grayImage.rows / 4);
        	bool isRight = true;
		//api_get_vanishing_point( grayImage, roi1, msac, center_point, true,"Wavelet");
            	cv::Mat two = twoRightMostLanes(grayImage.size(), dst, shift, isRight);
           	//cv::imshow("two", two);
		Rect roi2(0,   3*two.rows / 4, two.cols, two.rows / 4); //c?t ?nh
		
		Mat imgROI2 = two(roi2);
		//cv::imshow("roi", imgROI2);
		int widthSrc = imgROI2.cols;
		int heightSrc = imgROI2.rows;
		std::vector<Point> pointList;
		//for (int y = 0; y < heightSrc; y++)
		//{
			for (int x = widthSrc; x >= 0; x--)
				{
				if (imgROI2.at<uchar>(30, x) == 255 )/////////////////25
				{
					pointList.push_back(Point(30, x));
					//break;
					}
				if(pointList.size() == 0){
					pointList.push_back(Point(30, 300));
				}
			
			}
		//}
		//std::cout<<"size"<<pointList.size()<<std::endl;
		int x = 0, y = 0;
		int xTam = 0, yTam = 0;
		for (int i = 0; i < pointList.size(); i++)
			{
				x = x + pointList.at(i).y;
				y = y + pointList.at(i).x;
			}
		xTam = (x / pointList.size());
		yTam = (y / pointList.size());
		xTam = xTam ;
		if(pointList.size()<=15&&pointList.size()>1)xTam = xTam - 70;
		yTam = yTam + 240 * 3 / 4;
		circle(grayImage, Point(xTam, yTam), 2, Scalar(255, 255, 0), 3);
		//imshow("result", grayImage);
            	//if(center_point.x == 0 && center_point.y == 0) center_point = prvPosition;
            	//prvPosition = center_point;
	    	center_point = Point(xTam, yTam);
            	double angDiff = getTheta(carPosition, center_point);
		//if(-20<angDiff&&angDiff<20)angDiff=0;
            	theta = (angDiff*2);
		//std::cout<<"angdiff"<<angDiff<<std::endl;
		// theta = (0.00);
		api_set_STEERING_control(pca9685,theta);
            	int pwm2 =  api_set_FORWARD_control( pca9685,throttle_val);
		auto et = chrono::high_resolution_clock::now();
		
		std::vector<int> signLabels;
		std::vector<string> names;
		// recognizer.labeling(boxes, labels, colorImg, signLabels, names);
		bt = chrono::high_resolution_clock::now();
		
		for (int i = 0; i < 0;i++) //names.size(); i++)
			{
				rectangle(colorImg, boxes[i], Scalar(255, 0, 0), 1, 8, 0);
				if (names[i] == "stop")
				{
					cout<<"dungxe";
					throttle_val = 0;
					theta = (0.00);	
					api_set_STEERING_control(pca9685,theta);
					api_set_FORWARD_control( pca9685,throttle_val);
					running = !running;
				}
				/*if (names[i] == "leftTurn")
				{
					cout<<"re trai";
					theta = (-60.00);	
					api_set_STEERING_control(pca9685,theta);
					api_set_FORWARD_control( pca9685,throttle_val);
					usleep(700000);//running = !running;
				}*/
				putText(colorImg, names[i], Point(boxes[i].x, boxes[i].y - 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
				
			}
		//cout<<"theta"<<theta<<endl;
		if (debug) printf("Sign_detection run in %.2fms\n", chrono::duration<double, milli> (et-bt).count());
			// imshow("color", colorImg);
			
			char ch = waitKey(10);
			if (ch == 'q')
				break;
			//////// End Detect traffic signs //////////////
    	}
    }
    return 0;
}


