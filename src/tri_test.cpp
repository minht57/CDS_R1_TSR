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

int W_H_val = 0;
int W_H_max = 255;
int W_S_val = 0;
int W_S_max = 255;
int W_V_val = 200;
int W_V_max = 255;

int K_H_val = 100;
int K_H_max = 255;
int K_S_val = 0;
int K_S_max = 255;
int K_V_val = 58;
int K_V_max = 120;

int C_ths_val = 0;
int C_ths_max = 255;

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

    // createTrackbar("B H MIN val",nameWindow2,&B_H_val, 255);
    // createTrackbar("B H MAX val",nameWindow2,&B_H_max, 255);

    // createTrackbar("B S MIN val",nameWindow2,&B_S_val, 255);
    // createTrackbar("B S MAX val",nameWindow2,&B_S_max, 255);

    // createTrackbar("B V MIN val",nameWindow2,&B_V_val, 255);
    // createTrackbar("B V MAX val",nameWindow2,&B_V_max, 255);

    // createTrackbar("R H MIN val",nameWindow2,&R_H_val, 255);
    // createTrackbar("R H MAX val",nameWindow2,&R_H_max, 255);

    // createTrackbar("R S MIN val",nameWindow2,&R_S_val, 255);
    // createTrackbar("R S MAX val",nameWindow2,&R_S_max, 255);

    // createTrackbar("R V MIN val",nameWindow2,&R_V_val, 255);
    // createTrackbar("R V MAX val",nameWindow2,&R_V_max, 255);

    // createTrackbar("R2 H MIN val",nameWindow2,&R2_H_val, 255);
    // createTrackbar("R2 H MAX val",nameWindow2,&R2_H_max, 255);

    // createTrackbar("R2 S MIN val",nameWindow2,&R2_S_val, 255);
    // createTrackbar("R2 S MAX val",nameWindow2,&R2_S_max, 255);

    // createTrackbar("R2 V MIN val",nameWindow2,&R2_V_val, 255);
    // createTrackbar("R2 V MAX val",nameWindow2,&R2_V_max, 255);

    // createTrackbar("C",nameWindow2,&C_ths_val, C_ths_max);

    createTrackbar("W H MIN val",nameWindow2,&W_H_val, 255);
    createTrackbar("W H MAX val",nameWindow2,&W_H_max, 255);

    createTrackbar("W S MIN val",nameWindow2,&W_S_val, 255);
    createTrackbar("W S MAX val",nameWindow2,&W_S_max, 255);

    createTrackbar("W V MIN val",nameWindow2,&W_V_val, 255);
    createTrackbar("W V MAX val",nameWindow2,&W_V_max, 255);

    createTrackbar("K H MIN val",nameWindow2,&K_H_val, 255);
    createTrackbar("K H MAX val",nameWindow2,&K_H_max, 255);

    createTrackbar("K S MIN val",nameWindow2,&K_S_val, 255);
    createTrackbar("K S MAX val",nameWindow2,&K_S_max, 255);

    createTrackbar("K V MIN val",nameWindow2,&K_V_val, 255);
    createTrackbar("K V MAX val",nameWindow2,&K_V_max, 255);
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
            count_zero++;
            continue;
        }
        Des_[i] = Kernel[i] * (float)Src1[i] / Src2[i];
        Result_ += Des_[i];
        // cout << "Des_[" << i <<"] " << Des_[i] <<  " Result " << Result_ << endl;
    }
    *Des = Result_ / (16 - count_zero);
    // cout << "Result_ " << Result_ << " - Percent " << *Des << endl;
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

    fstream resfile;
    resfile.open ("result/result.txt");
    resfile << "0" << endl << endl;

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

    // if(parser.option("c"))
    // {
    //     // cout << "Open Camera" <<endl;
    //     // vid(0); // open the default camera
    //     // if(!vid.isOpened())  // check if we succeeded
    //     //     return -1;
    // }
    

    // image_window window;

    //img_raw = imread("/home/duongthanh3327/CuocDuaSo/OpenCV/0473.jpg",1);


    cout << "Loading SVM detectors..." << endl;
    std::vector<TrafficSign> signs;
    signs.push_back(TrafficSign("PARE", "svm_detectors/pare_detector.svm",
                                rgb_pixel(255,0,0)));
    signs.push_back(TrafficSign("LOMBADA", "svm_detectors/lombada_detector.svm",
                                rgb_pixel(255,122,0)));
    signs.push_back(TrafficSign("PEDESTRE", "svm_detectors/pedestre_detector.svm",
                                rgb_pixel(255,255,0)));
    signs.push_back(TrafficSign("Oneway-exit", "svm_detectors/object_detector.svm", rgb_pixel(0,0,255)));

    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
    std::vector<object_detector<image_scanner_type> > detectors;

    for (int i = 0; i < signs.size(); i++) {
      object_detector<image_scanner_type> detector;
      deserialize(signs[i].svm_path) >> detector;
      detectors.push_back(detector);
    }

    float kernel_circle[20] = {1, 1, 1, 1, 1, 1, 1, 1 ,1 ,1 ,
                               1, 1, 1, 1, 1, 1, 1, 1 ,1 ,1};
    Mat circle_ = Mat::zeros( 80, 80, CV_8UC1 );
    circle(circle_, Point(40,40), 40, 255, CV_FILLED, 8);

    // namedWindow( "Circle", WINDOW_NORMAL );
    // imshow("Circle",circle_);

    float kernel_triangle[20] = {1, 1, 1, 1, 1, 1, 1, 1 ,1 ,1 ,
                                 1, 1, 1, 1, 1, 1, 1, 1 ,1 ,1};

    Mat triangle_ = Mat::zeros( 80, 80, CV_8UC1 );

    Point rook_points[1][3];
    rook_points[0][0] = Point(40, 0);
    rook_points[0][1] = Point(0, 80);
    rook_points[0][2] = Point(80, 80);

    const Point* ppt[1] = { rook_points[0] };
    int npt[] = { 3 };

    fillPoly(triangle_, ppt, npt, 1, 255, 8);


    float kernel_triangle2[20] = {1, 1, 1, 1, 1, 1, 1, 1 ,1 ,1 ,
                                  1, 1, 1, 1, 1, 1, 1, 1 ,1 ,1};

    Mat triangle2_ = Mat::zeros( 80, 80, CV_8UC1 );

    rook_points[0][0] = Point(0, 0);
    rook_points[0][1] = Point(80, 0);
    rook_points[0][2] = Point(40, 80);

    const Point* ppt1[1] = { rook_points[0] };

    fillPoly(triangle2_, ppt1, npt, 1, 255, 8);

    // namedWindow( "Triangle", WINDOW_NORMAL );
    // imshow("Triangle",triangle_);
    // namedWindow( "Triangle 2", WINDOW_NORMAL );
    // imshow("Triangle 2",triangle2_);

    float kernel_square[20] = {1, 1, 1, 1, 1, 1, 1, 1 ,1 ,1 ,
                               1, 1, 1, 1, 1, 1, 1, 1 ,1 ,1};

    Mat square_ = Mat::zeros( 80, 80, CV_8UC1 );
    Point rook_points2[1][4];
    rook_points2[0][0] = Point(0, 0);
    rook_points2[0][1] = Point(80, 0);
    rook_points2[0][2] = Point(80, 80);
    rook_points2[0][3] = Point(0, 80);

    const Point* ppt2[1] = { rook_points2[0] };
    int npt2[] = { 4 };

    fillPoly(square_, ppt2, npt2, 1, 255, 8);

    // namedWindow( "Square", WINDOW_NORMAL );
    // imshow("Square",square_);

    Init();

    while (1) {
    	Mat img_HSV, img_result, img_Grey, maskr, maskr1, maskr2, maskb, res, out, out1, edged;
    	Mat masks1,masks2;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<Vec4i> hierarchy;

    	if(parser.option("v"))
    	{
    		vid >> img_raw;
    	}
        // if(parser.option("c"))
        // {
        //     vid >> img_raw;
        // }
        if (img_raw.empty())
        {
            cout << "Video end" << endl;
            break;
        }

        idx_frame++;

        img_result = img_raw.clone();
        // imshow(nameWindow1,img_raw);
        // blur(img_result, img_result, Size(2*2+1,2*2+1),Point(-1,-1),BORDER_DEFAULT);
       // imshow(nameWindow3, blurred_img);
       	cvtColor(img_result, img_HSV, COLOR_BGR2HSV);
        //imshow(nameWindow4,img_HSV);
       	//inRange(frame,Scalar(low_b,low_g,low_r), Scalar(high_b,high_g,high_r),frame_threshold);
       	// cout << R_H_val << R_S_val << R_V_val << endl;
       	inRange(img_HSV, Scalar(B_H_val, B_S_val, B_V_val), Scalar(B_H_max, B_S_max, B_V_max), maskb);
      	inRange(img_HSV, Scalar(R_H_val, R_S_val, R_V_val), Scalar(R_H_max, R_S_max, R_V_max), maskr1);
      	inRange(img_HSV, Scalar(R2_H_val, R2_S_val, R2_V_val), Scalar(R2_H_max, R2_S_max, R2_V_max), maskr2);
       	maskr = maskr1 + maskr2 + maskb;
       	// imshow(nameWindow5,mask);


        Mat mask_w, mask_k, mask_wk;
        /*------------White-------------*/
        inRange(img_HSV, Scalar(W_H_val, W_S_val, W_V_val), Scalar(W_H_max, W_S_max, W_V_max), mask_w);

        /*------------White-------------*/
        inRange(img_HSV, Scalar(K_H_val, K_S_val, K_V_val), Scalar(K_H_max, K_S_max, K_V_max), mask_k);

        mask_wk = mask_k + mask_w;

        // namedWindow( "Mask W", WINDOW_NORMAL );
        // namedWindow( "Mask K", WINDOW_NORMAL );
        namedWindow( "Mask WK", WINDOW_NORMAL );
        // imshow("Mask W",mask_w);
        // imshow("Mask K",mask_k);
        imshow("Mask WK",mask_wk);

        // maskr = maskr1 + maskr2 + maskb + mask_wk;

        // Mat result_w, result_k;
        // bitwise_and(img_result, mask_w, result_w);
        // bitwise_and(img_result, mask_k, result_k);
        // // result_k = img_result & mask_k;
        // // result_w = img_result & mask_w;

        // namedWindow( "Result W", WINDOW_NORMAL );
        // namedWindow( "Result K", WINDOW_NORMAL );
        // imshow("Result W",result_w);
        // imshow("Result K",result_k);
       	
		// dilate(maskr, masks1, Mat(), Point(-1, -1), 2, 1, 1);
		// erode(masks1, masks1, Mat(), Point(-1, -1), 2, 1, 1);

		
		// erode(maskr, masks2, Mat(), Point(-1, -1), 2, 1, 1);
		// dilate(masks2, masks2, Mat(), Point(-1, -1), 2, 1, 1);

		// masks1.copyTo(res, masks2);

        //namedWindow( "Mask Filter", WINDOW_NORMAL );
        // namedWindow( "Mask 1", WINDOW_NORMAL );
        //namedWindow( "Mask 2", WINDOW_NORMAL );
        // namedWindow( "Edged", WINDOW_NORMAL );
        //namedWindow( "Contours fill", WINDOW_NORMAL );

		//imshow("Mask Filter",maskr);
        // imshow("Mask 1",masks1);
        // imshow("Mask 2",masks2);
        // imshow("Mask r",res);

       	// img_result.copyTo(res, maskr);

        // cvtColor(res, img_Grey, COLOR_BGR2GRAY);

        // imshow("Mask Filter",maskr);
        // imshow("Gray Filter",img_Grey);

        // GaussianBlur(img_Grey, img_Grey, Size(5, 5), 0);

        // Canny(mask_wk, edged, C_ths_val, C_ths_max);

        // Create a kernel that we will use for accuting/sharpening our image
        Mat kernel = (Mat_<float>(3,3) <<
                1,  1, 1,
                1, -8, 1,
                1,  1, 1); // an approximation of second derivative, a quite strong kernel
        Mat imgLaplacian;
        Mat sharp = maskr; // copy source image to another temporary one
        filter2D(sharp, imgLaplacian, CV_32F, kernel);
        maskr.convertTo(sharp, CV_32F);
        Mat imgResult = sharp - imgLaplacian;
        // convert back to 8bits gray scale
        imgResult.convertTo(imgResult, CV_8UC3);
        imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

     //    ////////////////////White - Black////////////////
     //    Mat imgLaplacian1;
     //    sharp = mask_wk; // copy source image to another temporary one
     //    filter2D(sharp, imgLaplacian1, CV_32F, kernel);
     //    mask_wk.convertTo(sharp, CV_32F);
     //    // imgResult = sharp - imgLaplacian2;
     //    // convert back to 8bits gray scale
     //    imgResult.convertTo(imgResult, CV_8UC3);
     //    imgLaplacian1.convertTo(imgLaplacian1, CV_8UC3);

 
     //    // namedWindow( "canny", WINDOW_NORMAL );
     //    // imshow("Canny",edged);

     //    findContours(imgLaplacian1, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

     //    //*******************************************************//
     //    for(int idx = 0; idx < contours.size(); idx++)
     //    {
     //        // drawContours( img_result, contours, idx, Scalar(255,0,0), CV_FILLED, 8, hierarchy );
     //        drawContours( imgLaplacian1, contours, idx, 255, CV_FILLED, 8, hierarchy );
     //    }

     //    erode(imgLaplacian1, imgLaplacian1, Mat(), Point(-1, -1), 2, 1, 1);

     //    Mat imgLaplacian3;
     //    sharp = imgLaplacian1; // copy source image to another temporary one
     //    filter2D(sharp, imgLaplacian3, CV_32F, kernel);
     //    imgLaplacian1.convertTo(sharp, CV_32F);
     //    // convert back to 8bits gray scale
     //    imgResult.convertTo(imgResult, CV_8UC3);
     //    imgLaplacian3.convertTo(imgLaplacian3, CV_8UC3);

     //    namedWindow( "HPF", WINDOW_NORMAL );
     //    imshow("HPF",imgLaplacian1);
     //    namedWindow( "xxx", WINDOW_NORMAL );
     //    imshow("xxx",imgLaplacian3);

     //    // imshow("out",out);
        
     //    // namedWindow( "HPF", WINDOW_NORMAL );
     //    // imshow("HPF",imgLaplacian);

     //    // namedWindow( "imgResult", WINDOW_NORMAL );
     //    // imshow("imgResult",imgResult);


     //    findContours(imgLaplacian, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

     //    //*******************************************************//
	    // for(int idx = 0; idx < contours.size(); idx++)
	    // {
	    //     // drawContours( img_result, contours, idx, Scalar(255,0,0), CV_FILLED, 8, hierarchy );
     //        drawContours( imgLaplacian, contours, idx, 255, CV_FILLED, 8, hierarchy );
	    // }

     //    //erode(imgLaplacian, imgLaplacian, Mat(), Point(-1, -1), 2, 1, 1);

     //    Canny(imgLaplacian, edged, C_ths_val, C_ths_max);

	    // // imshow("Contours fill",img_result);
     //    // namedWindow( "Fill", WINDOW_NORMAL );
     //    // imshow("Fill",imgLaplacian);

     //    // bitwise_not(imgLaplacian)

        Mat img_Laplacian_test;

        img_Laplacian_test = imgLaplacian.clone();

        Mat imgLaplacian2;
        sharp = imgLaplacian; // copy source image to another temporary one
        filter2D(sharp, imgLaplacian2, CV_32F, kernel);
        imgLaplacian.convertTo(sharp, CV_32F);
        imgResult = sharp - imgLaplacian2;
        // convert back to 8bits gray scale
        imgResult.convertTo(imgResult, CV_8UC3);
        imgLaplacian2.convertTo(imgLaplacian2, CV_8UC3);

        // namedWindow( "Fill", WINDOW_NORMAL );
        // imshow("Fill",imgLaplacian);

        // imgLaplacian2 = imgLaplacian2 + imgLaplacian3;

        findContours(imgLaplacian2, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        // namedWindow( "imgLaplacian2", WINDOW_NORMAL );
        // imshow("imgLaplacian2",imgLaplacian2);

        //*******************************************************//
        std::vector<Rect> boundRect( contours.size() );

        for (unsigned int i=0; i < contours.size(); i++) {

            int area = contourArea(contours.at(i));
            // int peri = arcLength(contours.at(i), true);

            // cout << "Area: " << area << " Peri: " << peri << endl;

            std::vector<rect_detection> rects;

            if (area > 500) {
                //RotatedRect rect = minAreaRect(contours.at(i));

                //putText(img_result, "RED OBJECT", rect.center, FONT_HERSHEY_COMPLEX, 0.8, Scalar(255,255,255), 1);
                //drawContours(img_result,contours,i,Scalar(255,255,255),1);

            	boundRect[i] = boundingRect( Mat(contours[i]) );

                // cout << "ratio: " << f_test << endl;

            	if ((((float)boundRect[i].width/boundRect[i].height) > 0.5) &&(((float)boundRect[i].width/boundRect[i].height) < 1.2))
                {
                    // cout << "tl: " << boundRect[i].tl() <<" br: " << boundRect[i].br() <<endl;


                    // float f_test = (float)(area)/(boundRect[i].width * boundRect[i].height);
                    // if((f_test >=0.4) && (f_test <= 0.625))
                    // {
                    //     cout << "Tam giac " << f_test << " Area1: " << area << " Area2: " << boundRect[i].width * boundRect[i].height << endl;
                    //     continue;
                    // }
                    // else if((f_test > 0.625) && (f_test <= 0.875))
                    // {
                    //     cout << "Tron " << f_test << " Area1: " << area << " Area2: " << boundRect[i].width * boundRect[i].height << endl;
                    // }
                    // else if((f_test > 0.875) && (f_test <= 1.15))
                    // {
                    //     cout << "Vuong " << f_test << " Area1: " << area << " Area2: " << boundRect[i].width * boundRect[i].height << endl;
                    // }
                    // else
                    // {
                    //     continue;
                    // }


                    Rect roi2(boundRect[i].tl().x, boundRect[i].tl().y,
                        boundRect[i].br().x - boundRect[i].tl().x, boundRect[i].br().y - boundRect[i].tl().y);
                    Mat image_roi2 = img_Laplacian_test(roi2);

                    Size size2(80,80);
                    resize(image_roi2,image_roi2,size2);

                    // namedWindow( "Roi Image 2", WINDOW_NORMAL );
                    // imshow("Roi Image 2",image_roi2);

                    Mat image_test;

                    image_test = image_roi2.clone();
                    image_test = image_test & circle_;

                    // namedWindow( "X_Circle", WINDOW_NORMAL );
                    // imshow("X_Circle",image_test);


                    int Result[16];
                    int Result1[16], Result2[16];
                    CalcHistogram (circle_, Result1);
                    // cout  << Result1[0] << " " <<\
                    //          Result1[1] << " " <<\
                    //          Result1[2] << " " <<\
                    //          Result1[3] << " " <<\
                    //          Result1[4] << " " <<\
                    //          Result1[5] << " " <<\
                    //          Result1[6] << " " <<\
                    //          Result1[7] << " " <<\
                    //          Result1[8] << " " <<\
                    //          Result1[9] << " " <<\
                    //          Result1[10] << " " <<\
                    //          Result1[11] << " " <<\
                    //          Result1[12] << " " <<\
                    //          Result1[13] << " " <<\
                    //          Result1[14] << " " <<\
                    //          Result1[15] <<endl;

                    CalcHistogram (image_test, Result2);
                    // cout  << Result2[0] << " " <<\
                    //          Result2[1] << " " <<\
                    //          Result2[2] << " " <<\
                    //          Result2[3] << " " <<\
                    //          Result2[4] << " " <<\
                    //          Result2[5] << " " <<\
                    //          Result2[6] << " " <<\
                    //          Result2[7] << " " <<\
                    //          Result2[8] << " " <<\
                    //          Result2[9] << " " <<\
                    //          Result2[10] << " " <<\
                    //          Result2[11] << " " <<\
                    //          Result2[12] << " " <<\
                    //          Result2[13] << " " <<\
                    //          Result2[14] << " " <<\
                    //          Result2[15] <<endl;

                    float percent_circle, percent_trianle, percent_trianle2, percent_square;
                    CalcPercent(Result2, Result1, kernel_circle, &percent_circle);

                    image_test = image_roi2.clone();
                    image_test = image_test & triangle_;

                    // namedWindow( "X_Triangle", WINDOW_NORMAL );
                    // imshow("X_Triangle",image_test);

                    CalcHistogram (triangle_, Result1);
                    // cout  << Result1[0] << " " <<\
                    //          Result1[1] << " " <<\
                    //          Result1[2] << " " <<\
                    //          Result1[3] << " " <<\
                    //          Result1[4] << " " <<\
                    //          Result1[5] << " " <<\
                    //          Result1[6] << " " <<\
                    //          Result1[7] << " " <<\
                    //          Result1[8] << " " <<\
                    //          Result1[9] << " " <<\
                    //          Result1[10] << " " <<\
                    //          Result1[11] << " " <<\
                    //          Result1[12] << " " <<\
                    //          Result1[13] << " " <<\
                    //          Result1[14] << " " <<\
                    //          Result1[15] <<endl;

                    CalcHistogram (image_test, Result2);
                    // cout  << Result2[0] << " " <<\
                    //          Result2[1] << " " <<\
                    //          Result2[2] << " " <<\
                    //          Result2[3] << " " <<\
                    //          Result2[4] << " " <<\
                    //          Result2[5] << " " <<\
                    //          Result2[6] << " " <<\
                    //          Result2[7] << " " <<\
                    //          Result2[8] << " " <<\
                    //          Result2[9] << " " <<\
                    //          Result2[10] << " " <<\
                    //          Result2[11] << " " <<\
                    //          Result2[12] << " " <<\
                    //          Result2[13] << " " <<\
                    //          Result2[14] << " " <<\
                    //          Result2[15] <<endl;

                    CalcPercent(Result2, Result1, kernel_triangle, &percent_trianle);

                    // namedWindow( "X_Triangle", WINDOW_NORMAL );
                    // imshow("X_Triangle",image_test);

                    CalcHistogram (triangle2_, Result1);
                    // cout  << Result1[0] << " " <<\
                    //          Result1[1] << " " <<\
                    //          Result1[2] << " " <<\
                    //          Result1[3] << " " <<\
                    //          Result1[4] << " " <<\
                    //          Result1[5] << " " <<\
                    //          Result1[6] << " " <<\
                    //          Result1[7] << " " <<\
                    //          Result1[8] << " " <<\
                    //          Result1[9] << " " <<\
                    //          Result1[10] << " " <<\
                    //          Result1[11] << " " <<\
                    //          Result1[12] << " " <<\
                    //          Result1[13] << " " <<\
                    //          Result1[14] << " " <<\
                    //          Result1[15] <<endl;

                    CalcHistogram (image_test, Result2);
                    // cout  << Result2[0] << " " <<\
                    //          Result2[1] << " " <<\
                    //          Result2[2] << " " <<\
                    //          Result2[3] << " " <<\
                    //          Result2[4] << " " <<\
                    //          Result2[5] << " " <<\
                    //          Result2[6] << " " <<\
                    //          Result2[7] << " " <<\
                    //          Result2[8] << " " <<\
                    //          Result2[9] << " " <<\
                    //          Result2[10] << " " <<\
                    //          Result2[11] << " " <<\
                    //          Result2[12] << " " <<\
                    //          Result2[13] << " " <<\
                    //          Result2[14] << " " <<\
                    //          Result2[15] <<endl;

                    CalcPercent(Result2, Result1, kernel_triangle2, &percent_trianle2);

                    image_test = image_roi2.clone();
                    image_test = image_test & square_;

                    // namedWindow( "X_Square", WINDOW_NORMAL );
                    // imshow("X_Square",image_test);

                    CalcHistogram (square_, Result1);
                    // cout  << Result1[0] << " " <<\
                    //          Result1[1] << " " <<\
                    //          Result1[2] << " " <<\
                    //          Result1[3] << " " <<\
                    //          Result1[4] << " " <<\
                    //          Result1[5] << " " <<\
                    //          Result1[6] << " " <<\
                    //          Result1[7] << " " <<\
                    //          Result1[8] << " " <<\
                    //          Result1[9] << " " <<\
                    //          Result1[10] << " " <<\
                    //          Result1[11] << " " <<\
                    //          Result1[12] << " " <<\
                    //          Result1[13] << " " <<\
                    //          Result1[14] << " " <<\
                    //          Result1[15] <<endl;

                    CalcHistogram (image_test, Result2);
                    // cout  << Result2[0] << " " <<\
                    //          Result2[1] << " " <<\
                    //          Result2[2] << " " <<\
                    //          Result2[3] << " " <<\
                    //          Result2[4] << " " <<\
                    //          Result2[5] << " " <<\
                    //          Result2[6] << " " <<\
                    //          Result2[7] << " " <<\
                    //          Result2[8] << " " <<\
                    //          Result2[9] << " " <<\
                    //          Result2[10] << " " <<\
                    //          Result2[11] << " " <<\
                    //          Result2[12] << " " <<\
                    //          Result2[13] << " " <<\
                    //          Result2[14] << " " <<\
                    //          Result2[15] <<endl;

                    CalcPercent(Result2, Result1, kernel_square, &percent_square);
                    // cout <<"Circle " << percent_circle << "\t Triangle " << percent_trianle << " \t Square " << percent_square << endl;

                    if((percent_circle > 0.8) && (percent_trianle > 0.8) && (percent_square > 0.8))
                    {
                        cout << idx_frame << "\t Vuong\t" << percent_square << endl;
                    }
                    else if (percent_circle > 0.8)
                    {
                        cout << idx_frame << "\t Tron\t" << percent_circle << endl;
                    }
                    else if (percent_trianle > 0.8)
                    {
                        cout << idx_frame << "\t Tam giac xuoi\t" << percent_trianle << endl;
                    }
                    else if (percent_trianle2 > 0.8)
                    {
                        cout << idx_frame << "\t Tam giac nguoc\t" << percent_trianle2 << endl;
                    }
                    else
                    {
                        continue;
                    }

                    Rect roi(boundRect[i].tl().x, boundRect[i].tl().y,
                        boundRect[i].br().x - boundRect[i].tl().x, boundRect[i].br().y - boundRect[i].tl().y);
                    Mat image_roi = img_result(roi);

                    Size size(80,80);
                    resize(image_roi,image_roi,size);

                    // namedWindow( "Roi Image", WINDOW_NORMAL );
                    // imshow("Roi Image",image_roi);

                    // dlib::array<array2d<unsigned char> > images_HOG;
                    cv_image<bgr_pixel> images_HOG(image_roi);

                    evaluate_detectors(detectors, images_HOG, rects);

                    if(rects.size() > 0)
                    {
                        idx_detect++;
                        // cout << signs[rects[0].weight_index].name << " - tl: " << boundRect[i].tl() <<" br: " << boundRect[i].br() << endl;
                        cv::rectangle( img_result, boundRect[i].tl(), boundRect[i].br(), Scalar(255,0,0), 2, 8, 0 );
                        resfile << idx_frame << " " << signs[rects[0].weight_index].name << " " \
                        << boundRect[i].tl().x << " " << boundRect[i].tl().y << " "\
                        << boundRect[i].br().x << " " <<boundRect[i].br().y <<endl;
                    }
                    else
                    {
                        cv::rectangle( img_result, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
                    }
                
                }
            }
        }

        namedWindow( nameWindow6 , WINDOW_NORMAL );
        imshow(nameWindow6,img_result);
        if(parser.option("v"))
        {
            if(waitKey(10) == 27)
                break;
        }
        else
        {
            if(waitKey(10) == 27)
                break;
        }
    }

    resfile.close();
    resfile.open ("result/result.txt");
    resfile << idx_detect << endl;
    resfile.close();

    return 1;
}
