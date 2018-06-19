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

struct TrafficSign {
	string name;
	string svm_path;
	rgb_pixel color;
	TrafficSign(string name, string svm_path, rgb_pixel color) :
	name(name), svm_path(svm_path), color(color) {};
};

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


int main(int argc, char** argv)
{
	command_line_parser parser;
	Mat img_raw;
	VideoCapture vid;
	parser.add_option("h","Display this help message.");
	parser.add_option("v","Read video");
	parser.add_option("i","Read image");
	parser.add_option("s","Save video");

	parser.parse(argc, argv);

	fstream resfile;
	// resfile.open ("result/result.txt");
	// resfile << "0" << endl << endl;

	int cnt = 0;
	int idx_frame = 0;
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
		// cout << "okv" <<endl;
		string videofile = string(parser[0]);
		cout << "Opening videofile: \"" + videofile + "\"" << endl;
		vid.open(videofile);

		if (!vid.isOpened())
		{
			cout << "Video isn't opened" << endl;
			return EXIT_FAILURE;
		}

	}
	if(parser.option("s"))	
		video.open(string(parser[1])+"_video.avi",CV_FOURCC('M','J','P','G'),60, Size(640, 480), 1); 

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

    //********************************Create Kernels***********************************************//


    // namedWindow( "Circle", WINDOW_NORMAL );
    // imshow("Circle",circle_);

	namedWindow( nameWindow6 , WINDOW_NORMAL );
	setMouseCallback(nameWindow6, CallBackFunc, NULL);
	double ttime = 0;

	while (1) {

		double t = (double)getTickCount();

		Mat img_HSV, img_result, maskr, maskr1, maskr2, maskb, maskk, edged;

		std::vector<char> save_template;

		if(parser.option("v"))
		{
			if (!bpause)
			{
				vid >> img_raw;
				cnt++;
			}
			if (capture)
			{
				cout << "capture image" << endl;
				imwrite( "capture/"+ string(parser[1]) + "-" + to_string(ccpt)+".jpg" , img_raw );
				ccpt++;
				capture = false;
			}
		}
		if (img_raw.empty())
		{
			t = ((double)getTickCount() - t)/getTickFrequency();
			break;
		}

		img_result = img_raw.clone();

		cvtColor(img_result, img_HSV, COLOR_BGR2HSV);

		// cout << "image: " << cnt << endl;

		inRange(img_HSV, Scalar(B_H_val, B_S_val, B_V_val), Scalar(B_H_max, B_S_max, B_V_max), maskb);
		inRange(img_HSV, Scalar(R_H_val, R_S_val, R_V_val), Scalar(R_H_max, R_S_max, R_V_max), maskr1);
		inRange(img_HSV, Scalar(R2_H_val, R2_S_val, R2_V_val), Scalar(R2_H_max, R2_S_max, R2_V_max), maskr2);
		inRange(img_HSV, Scalar(K_H_val, K_S_val, K_V_val), Scalar(K_H_max, K_S_max, K_V_max), maskk);
		edged = maskr1 + maskr2 + maskb;

		// imshow("Black threshold", maskk);
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

    imshow("imgLaplacian", imgLaplacian);

        findContours(imgLaplacian, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        std::vector<Rect> boundRect( contours.size() );
        Mat drawing = Mat::zeros(imgLaplacian.size(), CV_8UC1);

        for(int idx = 0; idx < contours.size(); idx++)
        {
        	// drawContours( img_result, contours, idx, Scalar(255,0,0), 1 );

        	boundRect[idx] = boundingRect( Mat(contours[idx]) );
        	if ((boundRect[idx].width*boundRect[idx].height > 900) && (boundRect[idx].width*boundRect[idx].height < 50000))
        	{
        		if((boundRect[idx].tl().x != 0)&&(boundRect[idx].tl().y != 0) && (boundRect[idx].br().x != img_raw.cols) && (boundRect[idx].br().y != img_raw.rows))
        		{
        			if ( ( (float)boundRect[idx].width/boundRect[idx].height > 0.5) && ( (float)boundRect[idx].width/boundRect[idx].height < 1.3 ) )
        			{
        				// drawContours( imgLaplacian, contours, idx, 255, CV_FILLED, 8, hierarchy );
        				Mat src = img_raw(boundRect[idx]).clone();
 								
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
        if(parser.option("s"))
        	video.write(img_result);
        if(parser.option("v"))
        {
        	if(waitKey(1) == 27)
        		break;
        	t = ((double)getTickCount() - t)/getTickFrequency();
        	ttime += t;
        	// destroyAllWindows();
        }
        if(parser.option("i"))
        {
        	if(waitKey(0))
        		break;
        }

        // t = ((double)getTickCount() - t)/getTickFrequency();
        // cout << "2: " << t << endl;

    }

    resfile.close();
    resfile.open ("result/result.txt");
    resfile << idx_detect << endl;
    resfile.close();

    return 1;
}