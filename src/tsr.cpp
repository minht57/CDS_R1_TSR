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

	// int B_blur_val = 4;
	// int B_blur_max = 255;

int B_H_val = 100;
int B_H_max = 150;
int B_S_val = 50;
int B_S_max = 255;
int B_V_val = 50;
int B_V_max = 255;

int R_H_val = 150;
int R_H_max = 255;
int R_S_val = 50;
int R_S_max = 255;
int R_V_val = 50;
int R_V_max = 255;

int R2_H_val = 0;
int R2_H_max = 05;
int R2_S_val = 90;
int R2_S_max = 255;
int R2_V_val = 50;
int R2_V_max = 255;

int C_ths_val = 0;
int C_ths_max = 255;

RNG rng(12345);
	// Scalar red_min  = Scalar(0,36,0);
	// Scalar red_max = Scalar(25, 255, 255);

struct TrafficSign {
	string name;
	string svm_path;
	rgb_pixel color;
	TrafficSign(string name, string svm_path, rgb_pixel color) :
	name(name), svm_path(svm_path), color(color) {};
};

void Init() {
	namedWindow(nameWindow2, 0);
	    //createTrackbar("Blur size", nameWindow2, &B_blur_val, B_blur_max);

	createTrackbar("B H MIN val",nameWindow2,&B_H_val, 255);
	createTrackbar("B H MAX val",nameWindow2,&B_H_max, 255);

	createTrackbar("B S MIN val",nameWindow2,&B_S_val, 255);
	createTrackbar("B S MAX val",nameWindow2,&B_S_max, 255);

	createTrackbar("B V MIN val",nameWindow2,&B_V_val, 255);
	createTrackbar("B V MAX val",nameWindow2,&B_V_max, 255);

	createTrackbar("R H MIN val",nameWindow2,&R_H_val, 255);
	createTrackbar("R H MAX val",nameWindow2,&R_H_max, 255);

	createTrackbar("R S MIN val",nameWindow2,&R_S_val, 255);
	createTrackbar("R S MAX val",nameWindow2,&R_S_max, 255);

	createTrackbar("R V MIN val",nameWindow2,&R_V_val, 255);
	createTrackbar("R V MAX val",nameWindow2,&R_V_max, 255);

	createTrackbar("R2 H MIN val",nameWindow2,&R2_H_val, 255);
	createTrackbar("R2 H MAX val",nameWindow2,&R2_H_max, 255);

	createTrackbar("R2 S MIN val",nameWindow2,&R2_S_val, 255);
	createTrackbar("R2 S MAX val",nameWindow2,&R2_S_max, 255);

	createTrackbar("R2 V MIN val",nameWindow2,&R2_V_val, 255);
	createTrackbar("R2 V MAX val",nameWindow2,&R2_V_max, 255);

	createTrackbar("C",nameWindow2,&C_ths_val, C_ths_max);
}

int main(int argc, char** argv)
{
	command_line_parser parser;
	Mat img_raw;
	VideoCapture vid;
	parser.add_option("h","Display this help message.");
	parser.add_option("v","Read video");
	parser.add_option("i","Read image");

	parser.parse(argc, argv);

	// fstream resfile;
	// resfile.open ("result/result.txt");
	// resfile << "0" << endl << endl;

	int cnt = 0;
	int idx_detect = 0;
	if (parser.option("h")) {
		cout << "Usage: " << argv[0] << " [options] <list of images>" << endl;
		parser.print_options();

		return EXIT_SUCCESS;
	}

	if (parser.number_of_arguments() == 0) {
		cout << "You must give a list of input files." << endl;
		cout << "\nTry the -h option for more information." << endl;
		return EXIT_FAILURE;
	}

	if(parser.option("i"))
	{
		cout << "Open Image" <<endl;
		string videofile = string(parser[0]);
		img_raw = imread(videofile);
	}
	if(parser.option("v"))
	{
		cout << "okv" <<endl;
		string videofile = string(parser[0]);
		cout << "Opening videofile: \"" + videofile + "\"" << endl;
		vid.open(videofile);

		if (!vid.isOpened())
		{
			cout << "Video isn't opened" << endl;
			return EXIT_FAILURE;
		}

	}

	cout << "Loading SVM detectors..." << endl;
	std::vector<TrafficSign> signs;

	signs.push_back(TrafficSign("object_detector", "svm_detectors/object_detector.svm", rgb_pixel(0,0,255)));

	signs.push_back(TrafficSign("oneway-exit", "resources/detectors/oneway-exit-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	signs.push_back(TrafficSign("crossing", "resources/detectors/crossing-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	signs.push_back(TrafficSign("give-way", "resources/detectors/give-way-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	signs.push_back(TrafficSign("main", "resources/detectors/main-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	signs.push_back(TrafficSign("parking", "resources/detectors/parking-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	signs.push_back(TrafficSign("stop", "resources/detectors/stop-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
	std::vector<object_detector<image_scanner_type> > detectors;

	for (int i = 0; i < signs.size(); i++) {
		object_detector<image_scanner_type> detector;
		deserialize(signs[i].svm_path) >> detector;
		detectors.push_back(detector);
	}

	    //Init();
	double ttime = 0;
	while (1) {

		double t = (double)getTickCount();

		Mat img_HSV, img_result, img_Grey, maskr, maskr1, maskr2, maskb, res, out, out1, edged;
		Mat masks1,masks2;

		if(parser.option("v"))
		{
			vid >> img_raw;
			cnt++;
		}
		if (img_raw.empty())
		{
			t = ((double)getTickCount() - t)/getTickFrequency();
			ttime += t;
			cout << fixed;
			cout << endl <<	"*********************************************************" <<	endl;
			cout << "*	***************Video  Info***************	*"  <<	endl;
			cout << "*	* Video time:		" << setprecision(2) << vid.get(CAP_PROP_FRAME_COUNT)/vid.get(CAP_PROP_FPS) << " s		*	*" << endl;
			cout << "*	* Number of frame:	" << (int)vid.get(CAP_PROP_FRAME_COUNT) << "		*	*" <<endl;
			cout << "*	* Frame size:		" << (int)vid.get(CV_CAP_PROP_FRAME_WIDTH) <<" x " << (int)vid.get(CAP_PROP_FRAME_HEIGHT) << "	*	*" <<endl;
			cout << "*	* FPS:			" << vid.get(CAP_PROP_FPS) << "		*	*" <<endl;
			cout << "*	**********Processing Info****************	*"  <<	endl;
			cout << "*	* Total time:		" <<setprecision(2) << ttime << " s" << "		*	*" <<endl;
			cout << "*	* Total frame:		" << cnt+1 << "		*	*" <<endl;
			cout << "*	* Frame size:		" << (int)vid.get(CV_CAP_PROP_FRAME_WIDTH) <<" x " << (int)vid.get(CAP_PROP_FRAME_HEIGHT) << "	*	*" <<endl;
			cout << "*	* FPS:			" << setprecision(2) << (double)cnt/ttime << "		*	*" <<endl;
			cout <<	"*********************************************************" <<	endl <<	endl;
			cout << "Video end" << endl;
			break;
		}

		img_result = img_raw.clone();

		cvtColor(img_result, img_HSV, COLOR_BGR2HSV);

		// cout << "image: " << cnt << endl;

		inRange(img_HSV, Scalar(B_H_val, B_S_val, B_V_val), Scalar(B_H_max, B_S_max, B_V_max), maskb);
		inRange(img_HSV, Scalar(R_H_val, R_S_val, R_V_val), Scalar(R_H_max, R_S_max, R_V_max), maskr1);
		inRange(img_HSV, Scalar(R2_H_val, R2_S_val, R2_V_val), Scalar(R2_H_max, R2_S_max, R2_V_max), maskr2);
		edged = maskr1 + maskr2 + maskb;

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

        // Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
        // dilate(imgLaplacian, imgLaplacian, kernel1);

        std::vector<std::vector<cv::Point> > contours;
        std::vector<Vec4i> hierarchy;

    imshow("imgLaplacian", edged);

        findContours(imgLaplacian, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        std::vector<Rect> boundRect( contours.size() );
        Mat drawing = Mat::zeros(imgLaplacian.size(), CV_8UC1);

        for(int idx = 0; idx < contours.size(); idx++)
        {
        	// drawContours( img_result, contours, idx, Scalar(255,0,0), 1 );

        	boundRect[idx] = boundingRect( Mat(contours[idx]) );

        	if ((boundRect[idx].width*boundRect[idx].height > 900) && (boundRect[idx].width*boundRect[idx].height < 50000))
        	{
        		if ( ( (float)boundRect[idx].width/boundRect[idx].height > 0.5) && ( (float)boundRect[idx].width/boundRect[idx].height < 1.3 ) )
        		{
        			drawContours( drawing, contours, idx, 255, CV_FILLED, 8, hierarchy );
        			// imshow("drawing" + to_string(idx), drawing);
#ifdef 	OFFSET_BOUNDRECT
        			int tl_x = boundRect[idx].tl().x;
        			int tl_y = boundRect[idx].tl().y;

        			int br_x = boundRect[idx].br().x;
        			int br_y = boundRect[idx].br().y;

        			int offset = 2;
        			if(tl_x > offset-1) tl_x = tl_x - offset;

        			if(tl_y > offset-1) tl_y = tl_y - offset;

        			if(br_x + offset-1< img_result.cols) br_x = br_x + offset;

        			if(br_y + offset-1< img_result.rows) br_y = br_y + offset;

        			Rect roi(tl_x, tl_y, br_x - tl_x, br_y - tl_y);

        			Mat roi_d = drawing(roi).clone();
        			Mat roi_s = img_raw(roi).clone();

#else
        			Mat crp_drw = drawing(boundRect[idx]).clone();
        			Mat src = img_raw(boundRect[idx]).clone();

        			int linewidth = 2;
					Scalar value;
        			value = Scalar(0, 0, 0);
        			Mat roi_d;
        			copyMakeBorder( crp_drw, roi_d, linewidth, linewidth, linewidth, linewidth, BORDER_CONSTANT, value );
        			Mat roi_s;
        			copyMakeBorder( src, roi_s, linewidth, linewidth, linewidth, linewidth, BORDER_CONSTANT, value );

#endif

        			Mat dist;
        			distanceTransform(roi_d, dist, CV_DIST_L2, 3);
        			normalize(dist, dist, 0, 1., NORM_MINMAX);
        			threshold(dist, dist, .4, 1., CV_THRESH_BINARY);

        			Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
        			dilate(dist, dist, kernel1);

        			Mat dist_8u;
        			dist.convertTo(dist_8u, CV_8U);

        			std::vector<std::vector<cv::Point> > contoursm;
        			std::vector<Vec4i> hierarchym;

        			findContours(dist_8u, contoursm, hierarchym, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        			Mat markers = Mat::zeros(dist.size(), CV_32SC1);
        			for(int isx = 0; isx < contoursm.size(); isx++) {
        				drawContours(markers, contoursm, static_cast<int>(isx), Scalar::all(static_cast<int>(isx)+1), -1);	
        			}

        			circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
		    // imshow("Markers", markers*10000);

        			// t = ((double)getTickCount() - t)/getTickFrequency();
        			// ttime += t;
        			// cout << "1: " << t << endl;
        			// t = (double)getTickCount();


        			watershed(roi_s, markers);


        			// t = ((double)getTickCount() - t)/getTickFrequency();
        			// ttime += t;
        			// cout << "3: " << t << endl;
        			// t = (double)getTickCount();

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

        			// bitwise_not(dst,dst);
        			dst = dst & roi_d;

      //   	namedWindow( "Dst", WINDOW_NORMAL );
   			// imshow("Dst", dst);
        			
        			std::vector<std::vector<cv::Point> > contoursw;
        			std::vector<Vec4i> hierarchyw;

        			findContours(dst, contoursw, hierarchyw, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

        			std::vector<Rect> boundRectw( contoursw.size() );

        			if(contoursw.size() > 0)
        			{
        				for(int iw = 0; iw < contoursw.size(); iw++) {

        					if (hierarchyw[iw][2]< 0)
        					{        						
        						boundRectw[iw] = boundingRect( Mat(contoursw[iw]) );
        						if ( ( (boundRectw[iw].width*boundRectw[iw].height) > 900) && ( (boundRectw[iw].width*boundRectw[iw].height)  < 50000) )
        						{        							
        							if ( ( (float)boundRectw[iw].width/boundRectw[iw].height < 1.3) && ( (float)boundRectw[iw].width/boundRectw[iw].height > 0.5) )
        							{
        								// drawContours(img_result, contoursw, iw, Scalar(255,0,0), 1);
#ifdef OFFSET_BOUNDRECT
        								Rect dstbound (tl_x + boundRectw[iw].tl().x, tl_y + boundRectw[iw].tl().y,boundRectw[iw].width, boundRectw[iw].height);
        								cv::rectangle( img_result, dstbound, Scalar(0,255,0), 2, 8, 0 );
#else
        								Rect dstbound (boundRectw[iw].tl().x + boundRect[idx].tl().x - 2, boundRectw[iw].tl().y + boundRect[idx].tl().y - 2,boundRectw[iw].width, boundRectw[iw].height);
        								// cv::rectangle( img_result, dstbound, Scalar(0,255,0), 2, 8, 0 );       						
#endif
        								Mat image_roi = img_result(dstbound);
        								cv_image<bgr_pixel> images_HOG(image_roi);
        								std::vector<rect_detection> rects;

        								evaluate_detectors(detectors, images_HOG, rects);
        								if(rects.size() > 0)
        									cv::rectangle( img_result, dstbound.tl(), dstbound.br(), Scalar(255,0,0), 2, 8, 0 );
        								else cv::rectangle( img_result, dstbound.tl(), dstbound.br(), Scalar(0,255,0), 2, 8, 0 );
        							}
        						}
        					}

        				}
        			}
        		}
        	}
        }

        namedWindow( nameWindow6 , WINDOW_NORMAL );
        imshow(nameWindow6,img_result);
        if(parser.option("v"))
        {
        	if(waitKey(1) == 27)
        		break;
        	t = ((double)getTickCount() - t)/getTickFrequency();
        	ttime += t;
        }
        if(parser.option("i"))
        {
        	if(waitKey(0))
        		break;
        }

        // t = ((double)getTickCount() - t)/getTickFrequency();
        // cout << "2: " << t << endl;

    }

    // resfile.close();
    // resfile.open ("result/result.txt");
    // resfile << idx_detect << endl;
    // resfile.close();

return 1;
}
