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
int B_S_val = 80;
int B_S_max = 255;
int B_V_val = 100;
int B_V_max = 255;

int R_H_val = 140;
int R_H_max = 255;
int R_S_val = 40;
int R_S_max = 255;
int R_V_val = 105;
int R_V_max = 255;

int R2_H_val = 0;
int R2_H_max = 5;
int R2_S_val = 90;
int R2_S_max = 255;
int R2_V_val = 60;
int R2_V_max = 255;

int W_H_val = 107;
int W_H_max = 155;
int W_S_val = 40;
int W_S_max = 80;
int W_V_val = 150;
int W_V_max = 255;

int C_ths_val = 0;
int C_ths_max = 255;

// Scalar red_min  = Scalar(0,36,0);
// Scalar red_max = Scalar(25, 255, 255);


void Init() {
    namedWindow(nameWindow2, 0);
    namedWindow("TrackarsWhite", 0);
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

    createTrackbar("W H MIN val","TrackarsWhite",&W_H_val, 255);
    createTrackbar("W H MAX val","TrackarsWhite",&W_H_max, 255);

    createTrackbar("W S MIN val","TrackarsWhite",&W_S_val, 255);
    createTrackbar("W S MAX val","TrackarsWhite",&W_S_max, 255);

    createTrackbar("W V MIN val","TrackarsWhite",&W_V_val, 255);
    createTrackbar("W V MAX val","TrackarsWhite",&W_V_max, 255);

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
  parser.add_option("d","Draw vector HOG");
	
	parser.parse(argc, argv);

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

  if(parser.option("d"))
  {
    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
    object_detector<image_scanner_type> detector;
    deserialize(string(parser[0])) >> detector;
    image_window hogwin(draw_fhog(detector), "Learned fHOG detector");
    cout << "Press any key to exit!" << endl;
    cin.get();
    return EXIT_SUCCESS;
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
    
    std::vector<std::vector<cv::Point> > contours;
    std::vector<Vec4i> hierarchy;

    //img_raw = imread("/home/duongthanh3327/CuocDuaSo/OpenCV/0473.jpg",1);
    int cnt = 0;

    Init();

    while (1) {
    	Mat img_HSV, img_result, img_Grey, maskr, maskr1, maskr2, maskb, maskw, res, out, out1, edged;
    	if(parser.option("v"))
    	{
    		vid >> img_raw;
    	}
    	cnt++;
        img_result = img_raw.clone();
        // imshow(nameWindow1,img_raw);
        blur(img_result, img_result, Size(2*2+1,2*2+1),Point(-1,-1),BORDER_DEFAULT);
       // imshow(nameWindow3, blurred_img);
       	cvtColor(img_result, img_HSV, COLOR_BGR2HSV);
        //imshow(nameWindow4,img_HSV);
       	//inRange(frame,Scalar(low_b,low_g,low_r), Scalar(high_b,high_g,high_r),frame_threshold);
       	// cout << R_H_val << R_S_val << R_V_val << endl;
       	inRange(img_HSV, Scalar(B_H_val, B_S_val, B_V_val), Scalar(B_H_max, B_S_max, B_V_max), maskb);
      	inRange(img_HSV, Scalar(R_H_val, R_S_val, R_V_val), Scalar(R_H_max, R_S_max, R_V_max), maskr1);
      	inRange(img_HSV, Scalar(R2_H_val, R2_S_val, R2_V_val), Scalar(R2_H_max, R2_S_max, R2_V_max), maskr2);
      	inRange(img_HSV, Scalar(W_H_val, W_S_val, W_V_val), Scalar(W_H_max, W_S_max, W_V_max), maskw);
       	maskr = maskr1 + maskr2 + maskb ;

       	imshow("After mask",maskr);

       	// maskr = maskw;
		// erode(maskr, maskr, Mat(), Point(-1, -1), 2, 1, 1);
       	// dilate(maskr, maskr, Mat(), Point(-1, -1), 2, 1, 1);

     //   	Size kernalSize (5,5);
   		// Mat element = getStructuringElement (MORPH_RECT, kernalSize, Point(2,2)  );
   		// dilate( maskr, maskr, element );
		  Size kernalSize4 (5,5);
   		Mat element4 = getStructuringElement (MORPH_RECT, kernalSize4, Point(-1,-1)  );
   		morphologyEx( maskr, maskr, MORPH_CLOSE, element4 );

   		// Size kernalSize2 (7,7);
   		// Mat element2 = getStructuringElement (MORPH_RECT, kernalSize2, Point(3,3)  );
   		// erode( maskr, maskr, element2 );

   		// Size kernalSize3 (5,5);
   		// Mat element3 = getStructuringElement (MORPH_RECT, kernalSize3, Point(2,2)  );
   		// dilate( maskr, maskr, element3 );

   		

       	// img_result.copyTo(res, maskr);

        // cvtColor(res, img_Grey, COLOR_BGR2GRAY);

        // imshow("Mask Filter",maskr);
        // imshow("Gray Filter",img_Grey);

        // GaussianBlur(img_Grey, img_Grey, Size(5, 5), 0);

        //Canny(maskr, edged, C_ths_val, C_ths_max);

        Mat drawing = Mat::zeros(maskr.size(), CV_8UC3);

        // imshow("out",out);
        // imshow("Edged",edged);

        findContours(maskr, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

        //*******************************************************//
        //*******************************************************//
        std::vector<Rect> boundRect( contours.size() );
        std::vector<std::vector<Point>> contours_poly( contours.size() );
        for (unsigned int i=0; i < contours.size(); i++) {

            // int area = contourArea(contours.at(i));

            // if (area > 256) {
            //     //RotatedRect rect = minAreaRect(contours.at(i));

            //     //putText(img_result, "RED OBJECT", rect.center, FONT_HERSHEY_COMPLEX, 0.8, Scalar(255,255,255), 1);
            //     //drawContours(img_result,contours,i,Scalar(255,255,255),1);
            // 	boundRect[i] = boundingRect( Mat(contours[i]) );
            // 	if ((((float)boundRect[i].width/boundRect[i].height) > 0.5) &&(((float)boundRect[i].width/boundRect[i].height) < 1.2))
            //     	cv::rectangle( img_result, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
            // }
	        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, false );
	        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        }

        for( int i = 0; i< contours.size(); i++ )
		{
			drawContours( img_result, contours_poly, i, Scalar(0,0,255), 1, 8, std::vector<Vec4i>(), 0, Point() );
			if (boundRect[i].width*boundRect[i].height > 256)
			{
				if ((((float)boundRect[i].width/boundRect[i].height) > 0.5) &&(((float)boundRect[i].width/boundRect[i].height) < 1.3))
				{
					
					cv::rectangle( img_result, boundRect[i].tl(), boundRect[i].br(), Scalar(255,0,0), 1, 8, 0 );
				}
				//else cout << (float)boundRect[i].width/boundRect[i].height << endl;
			}

		}

		// imshow("Drawing",drawing);
        imshow(nameWindow6,img_result);
        if (waitKey(10) == 'q')
            break;
    }
    return 1;
}
