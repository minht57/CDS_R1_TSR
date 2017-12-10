#include <dlib/cmd_line_parser.h>

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

int H_val = 100;
int H_max = 150;
int S_val = 50;
int S_max = 255;
int V_val = 50;
int V_max = 255;

int C_ths_val = 0;
int C_ths_max = 255;


void Init() {
	namedWindow(nameWindow2, 0);

	createTrackbar("H MIN val",nameWindow2,&H_val, 255);
	createTrackbar("H MAX val",nameWindow2,&H_max, 255);

	createTrackbar("S MIN val",nameWindow2,&S_val, 255);
	createTrackbar("S MAX val",nameWindow2,&S_max, 255);

	createTrackbar("V MIN val",nameWindow2,&V_val, 255);
	createTrackbar("V MAX val",nameWindow2,&V_max, 255);
}

int main(int argc, char** argv)
{
	command_line_parser parser;
	Mat img_raw;
	parser.add_option("h","Display this help message.");
	parser.add_option("v","Read video");
	parser.add_option("i","Read image");
	
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

	if(parser.option("i"))
	{
		cout << "Open Image" <<endl;
		string videofile = string(parser[0]);
		img_raw = imread(videofile);
	}

	Init();

	while (1) {
		Mat img_HSV, img_result, mask;
		
		if (img_raw.empty())
		{
			cout << "Video end" << endl;
			break;
		}

		img_result = img_raw.clone();

		cvtColor(img_result, img_HSV, COLOR_BGR2HSV);

		inRange(img_HSV, Scalar(H_val, S_val, V_val), Scalar(H_max, S_max, V_max), mask);

		namedWindow( nameWindow5 , WINDOW_NORMAL );
		imshow(nameWindow5,mask);


		namedWindow( nameWindow6 , WINDOW_NORMAL );
		imshow(nameWindow6,img_result);

		if(waitKey(10) == 27)
			break;
	}
	return 1;
}
