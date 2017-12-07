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


// Scalar red_min  = Scalar(0,36,0);
// Scalar red_max = Scalar(25, 255, 255);


void Init() {
	namedWindow(nameWindow2, 0);
    //createTrackbar("Blur size", nameWindow2, &B_blur_val, B_blur_max);

	createTrackbar("H MIN val",nameWindow2,&B_H_val, 255);
	createTrackbar("H MAX val",nameWindow2,&B_H_max, 255);

	createTrackbar("S MIN val",nameWindow2,&B_S_val, 255);
	createTrackbar("S MAX val",nameWindow2,&B_S_max, 255);

	createTrackbar("V MIN val",nameWindow2,&B_V_val, 255);
	createTrackbar("V MAX val",nameWindow2,&B_V_max, 255);


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
		string videofile = string(parser[0]);
		cout << "Opening videofile: \"" + videofile + "\"" << endl;
		vid.open(videofile);

		if (!vid.isOpened())
		{
			cout << "Video isn't opened" << endl;
			return EXIT_FAILURE;
		}

	}

	int cnt = 0;

    // Init();

	while (1) {
        // Mat img_HSV, img_result, img_Grey, maskr, maskr1, maskr2, maskb, maskw, res, out, out1, edged;

        // std::vector<std::vector<cv::Point> > contours;
        // std::vector<Vec4i> hierarchy;

		if(parser.option("v"))
		{
			vid >> img_raw;
		}
        // cnt++;
        // img_result = img_raw.clone();

        // blur(img_result, img_result, Size(2*2+1,2*2+1),Point(-1,-1),BORDER_DEFAULT);

        // cvtColor(img_result, img_HSV, COLOR_BGR2HSV);


        // inRange(img_HSV, Scalar(B_H_val, B_S_val, B_V_val), Scalar(B_H_max, B_S_max, B_V_max), maskb);


        // namedWindow("mask", WINDOW_NORMAL);
        // imshow("mask",maskb);

        // findContours(maskb, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

        // for( int i = 0; i< contours.size(); i++ )
        // {
        //     drawContours( img_result, contours, i, Scalar(0,0,0), 2, 8, std::vector<Vec4i>(), 0, Point() );
        // }
        // imshow(nameWindow2,img_result);
        // if (waitKey(10) == 'q')
        //     break;
            // Check if everything was fine
		// if (!img_raw.data)
		// 	return -1;
    // Show source image
		imshow("Source Image", img_raw);
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
		for( int x = 0; x < img_raw.rows; x++ ) {
			for( int y = 0; y < img_raw.cols; y++ ) {
				if ( img_raw.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
					img_raw.at<Vec3b>(x, y)[0] = 0;
					img_raw.at<Vec3b>(x, y)[1] = 0;
					img_raw.at<Vec3b>(x, y)[2] = 0;
				}
			}
		}
    // Show output image
		// imshow("Black Background Image", img_raw);
    // Create a kernel that we will use for accuting/sharpening our image
		Mat kernel = (Mat_<float>(3,3) <<
			1,  1, 1,
			1, -8, 1,
            1,  1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
		Mat imgLaplacian;
    Mat sharp = img_raw; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    img_raw.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    // imshow( "New Sharped Image", imgResult );
    img_raw = imgResult; // copy back
    // Create binary image from source image
    Mat bw;
    cvtColor(img_raw, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    // bitwise_not(bw,bw);

    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    erode(bw, bw, kernel1);
    // dilate(bw, bw, kernel1);


    imshow("Binary Image", bw);


    // Perform the distance transform algorithm
//     Mat dist;
//     distanceTransform(bw, dist, CV_DIST_L2, 3);
//     // Normalize the distance image for range = {0.0, 1.0}
//     // so we can visualize and threshold it
//     normalize(dist, dist, 0, 1., NORM_MINMAX);
//     imshow("Distance Transform Image", dist);
//     // Threshold to obtain the peaks
//     // This will be the markers for the foreground objects
//     threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
//     // Dilate a bit the dist image
//     Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
//     dilate(dist, dist, kernel1);
//     imshow("Peaks", dist);
//     // Create the CV_8U version of the distance image
//     // It is needed for findContours()
//     Mat dist_8u;
//     dist.convertTo(dist_8u, CV_8U);
//     // Find total markers
//     std::vector<std::vector<Point> > contours;
//     findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//     // Create the marker image for the watershed algorithm
//     Mat markers = Mat::zeros(dist.size(), CV_32SC1);
//     // Draw the foreground markers
//     for (size_t i = 0; i < contours.size(); i++)
//         drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
//     // Draw the background marker
//     circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
//     imshow("Markers", markers*10000);
//     // Perform the watershed algorithm
//     watershed(img_raw, markers);
//     Mat mark = Mat::zeros(markers.size(), CV_8UC1);
//     markers.convertTo(mark, CV_8UC1);
//     bitwise_not(mark, mark);
// //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
//                                   // image looks like at that point
//     // Generate random colors
//     std::vector<Vec3b> colors;
//     for (size_t i = 0; i < contours.size(); i++)
//     {
//         int b = theRNG().uniform(0, 255);
//         int g = theRNG().uniform(0, 255);
//         int r = theRNG().uniform(0, 255);
//         colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
//     }
//     // Create the result image
    // Mat dst = Mat::zeros(markers.size(), CV_8UC3);
//     // Fill labeled objects with random colors
//     for (int i = 0; i < markers.rows; i++)
//     {
//         for (int j = 0; j < markers.cols; j++)
//         {
//             int index = markers.at<int>(i,j);
//             if (index > 0 && index <= static_cast<int>(contours.size()))
//                 dst.at<Vec3b>(i,j) = colors[index-1];
//             else
//                 dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
//         }
//     }
    // Visualize the final image
    // imshow("Final Result", dst);
    if(parser.option("v"))
    {
    	if(waitKey(1) == 27)
    		break;
        	// destroyAllWindows();
    }
    if(parser.option("i"))
    {
    	if(waitKey(0))
    		break;
    }

    
}
}