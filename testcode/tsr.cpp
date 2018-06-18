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

#include "api_kinect_cv.h"
#include <opencv2/imgproc.hpp>
#include "openni2.h"
#include "../openni2/Singleton.h"
#include <unistd.h>
#include "../sign_detection/SignDetection.h"
#include <chrono>
#include "extractInfo.h"
#include "../ObjectRecognition/SignRecognition.h"
#include <sys/stat.h>

using namespace std;
using namespace dlib;
using namespace cv;

using namespace openni;
using namespace framework;
using namespace signDetection;
using namespace SignRecognition;

#define SAMPLE_READ_WAIT_TIMEOUT 2000
signDetection::SignDetection detection;
/// Init openNI ///
Status rc;
Device device;
VideoStream depth, color;
VideoFrameRef frame_depth, frame_color;
VideoStream* streams[] = {&depth, &color};
int frame_id = 0;

struct TrafficSign {
	string name;
	string svm_path;
	rgb_pixel color;
	TrafficSign(string name, string svm_path, rgb_pixel color) :
	name(name), svm_path(svm_path), color(color) {};
};

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

int main(int argc, char** argv)
{
    /// Init openNI ///

	/*rc = OpenNI::initialize();
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
			depth_mode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
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
	}*/

/// End of openNI init phase ///
    double st = 0, et = 0, fps = 0;
    double freq;
    freq = getTickFrequency();
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
	tmp_img.push_back(trn_right);
		
	OpenNI2::Instance() -> init();
	ushort l_th = 600, h_th = 2000;//old: 600-2000
	std::vector<std::vector<Point> > regs;
	Mat depthImg, colorImg, grayImage, disparity;
	
	while (1) 
	{
	    //Open camera - depth and color image
	/*int readyStream = -1;
	rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
	if (rc != STATUS_OK)
	{
		printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
		return 0;
	}

	depth.readFrame(&frame_depth);
	color.readFrame(&frame_color);
	frame_id ++;

	char recordStatus = analyzeFrame(frame_depth,frame_color, depthImg, colorImg);*/
	
	//cv::Mat converted_depth;
	//const float scaleFactor = 1.0f/6;
    //depthImg.convertTo( converted_depth, CV_8UC1, scaleFactor );
    
      	OpenNI2::Instance()->getData(colorImg, depthImg, grayImage, disparity);
      	cv::Mat CutDepthImg = depthImg(cv::Rect(0,200,640,150)).clone();
        cv::imshow("color", colorImg);
        cv::imshow("depth", depthImg);
		Mat colorTemp = colorImg.clone();
		std::vector<Rect> boxes;
		std::vector<int> labels;
		cv::Mat result;
		
        double min, max;
        cv::minMaxLoc(depthImg, &min, &max);
        std::cout << "min: " << min << " - max: " << max << std::endl;
		
		st = getTickCount();
		
		detection.objectLabeling(boxes, labels, CutDepthImg, colorImg, result, l_th, h_th, 1000, 4000, 36, 200, 1.2);
		
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
        //FPS
        et = getTickCount();
        fps = 1.0 / ((et-st)/freq);
        cerr << "FPS: "<< fps<< '\n';
        waitKey(10);
    }
    return 1;
}
