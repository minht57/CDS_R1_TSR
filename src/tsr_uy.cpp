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
            count_zero++;
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

	// fstream resfile;
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
  video.open(string(parser[1])+"_video.avi",CV_FOURCC('M','J','P','G'),10, Size(640, 480), 1); 

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

    //*******************************************************************************************//

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
                if(parser.option("s"))
                   video.write(img_raw);
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
     ttime += t;
     cout << fixed;
     cout << endl <<	"*********************************************************" <<	endl;
     cout << "*	***************Video  Info***************	*"  <<	endl;
     cout << "*	* Video time:		" << setprecision(2) << vid.get(CAP_PROP_FRAME_COUNT)/vid.get(CAP_PROP_FPS) << " s		*	*" << endl;
     cout << "*	* Number of frame:	" << (int)vid.get(CAP_PROP_FRAME_COUNT) << "		*	*" <<endl;
     cout << "*	* Frame size:		" << (int)vid.get(CV_CAP_PROP_FRAME_WIDTH) <<"x" << (int)vid.get(CAP_PROP_FRAME_HEIGHT) << "		*	*" <<endl;
     cout << "*	* FPS:			" << vid.get(CAP_PROP_FPS) << "		*	*" <<endl;
     cout << "*	**********Processing Info****************	*"  <<	endl;
     cout << "*	* Total time:		" <<setprecision(2) << ttime << " s" << "		*	*" <<endl;
     cout << "*	* Total frame:		" << cnt+1 << "		*	*" <<endl;
     cout << "*	* Frame size:		" << (int)vid.get(CV_CAP_PROP_FRAME_WIDTH) <<"x" << (int)vid.get(CAP_PROP_FRAME_HEIGHT) << "		*	*" <<endl;
     cout << "*	* FPS:			" << setprecision(2) << (double)cnt/ttime << "		*	*" <<endl;
     cout << "*	* Number of ROI:	" << croi << "		*	*" <<endl;
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

    // imshow("imgLaplacian", imgLaplacian);

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
                       drawContours( imgLaplacian, contours, idx, 255, CV_FILLED, 8, hierarchy );
        			// imshow("drawing" + to_string(idx), drawing);

                       Mat crp_drw = imgLaplacian(boundRect[idx]).clone();
                       Mat src = img_raw(boundRect[idx]).clone();
                        // FindBoundingline(src, crp_drw, detectors, boundRect[idx], Scalar(255,255,0), img_result);
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

        				// namedWindow("drawing" + to_string(idx),WINDOW_NORMAL);
        				// namedWindow("Dist" + to_string(idx) ,WINDOW_NORMAL);
        				// imshow("drawing" + to_string(idx), crp_drw);
        				// imshow("Dist" + to_string(idx), dist);
        			// namedWindow("drawing" + to_string(idx),WINDOW_NORMAL);
        			// namedWindow("Dist",WINDOW_NORMAL);
        			// imshow("drawing", crp_drw);
        			// imshow("Dist", dist);

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
		            // imshow("Markers" + to_string(idx), markers*10000);

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

                     findContours(dst, contoursw, hierarchyw, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

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
                                   Rect dstbound (boundRectw[iw].tl().x + boundRect[idx].tl().x - 2, boundRectw[iw].tl().y + boundRect[idx].tl().y - 2,boundRectw[iw].width, boundRectw[iw].height);
        								// cv::rectangle( img_result, dstbound, Scalar(0,255,0), 2, 8, 0 );       						
                                   Mat image_roi = img_result(dstbound);
                    //*****************************************************************************************************************//
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
                                   Mat image_draw = Mat::zeros(dst.size(), CV_8UC1);
                                   drawContours(image_draw, contoursw, iw, 255, CV_FILLED, 8, hierarchyw );
                                   imshow("A",image_draw);

                                   // Rect roi2(boundRect[idx].tl().x, boundRect[idx].tl().y,
                                   //  boundRect[idx].br().x - boundRect[idx].tl().x, boundRect[idx].br().y - boundRect[idx].tl().y);
                                   Mat image_roi2 = image_draw.clone();

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

                                   image_test = image_roi2.clone();
                                   image_test = image_test & triangle2_;

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

                    //*****************************************************************************************************************//


                                resize(image_roi,image_roi,Size(80,80),INTER_LANCZOS4);
                                        // imshow("ROI",image_roi);
                                croi++;
                                        // imwrite( "capture/"+ string(parser[1]) + "_" + to_string(ccpt++) + ".jpg" , image_roi );
                                cv_image<bgr_pixel> images_HOG(image_roi);
                                std::vector<rect_detection> rects;
        								// cout << to_string(idx) << " "  << boundRectw[iw].tl() << endl;


                                evaluate_detectors(detectors, images_HOG, rects, 0.9);

                                if(rects.size() > 0)
                                {
                                    cout << "   " << cnt << ":  " << signs[rects[0].weight_index].name <<": " << rects[0].detection_confidence << endl;
                                    cv::rectangle( img_result, dstbound.tl(), dstbound.br(), Scalar(255,0,0), 2, 8, 0 );
                                    putText(img_result, signs[rects[0].weight_index].name, Point(dstbound.br().x, dstbound.tl().y), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255), 1, CV_AA);
                                    save_template.push_back(rects[0].weight_index);

                                }
                                else cv::rectangle( img_result, dstbound.tl(), dstbound.br(), Scalar(0,255,0), 2, 8, 0 );
                                            // waitKey(0);
                            }
                        }
                    }
                }
            }
        }
    }
}
}

for (int icout = 0; icout < save_template.size();icout++)
{
   tmp_img[save_template[icout]].copyTo(img_result(Rect(80*icout,400,tmp_img[save_template[icout]].cols,tmp_img[save_template[icout]].rows)));
}
        /*
        std::vector<std::vector<cv::Point> > contoursk;
        std::vector<Vec4i> hierarchyk;

        findContours(maskk, contoursk, hierarchyk, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        std::vector<Rect> boundRectk( contoursk.size() );
        Mat drawk = Mat::zeros(maskk.size(), CV_8UC1);

        for(int idx = 0; idx < contoursk.size(); idx++)
        {
        	boundRectk[idx] = boundingRect( Mat(contoursk[idx]) );
        	int calc_area = boundRectk[idx].width * boundRectk[idx].height;
        	int areak = contourArea(contoursk[idx]);

            if (areak > 100)
            {
                if ( ((float)areak/calc_area < 0.5) && ((float)areak/calc_area>0.4) )
                {
                    cv::rectangle( img_result, boundRectk[idx].tl(), boundRectk[idx].br(), Scalar(0,0,255), 2, 8, 0 );
                }
            }
        }*/

namedWindow( nameWindow6 , WINDOW_NORMAL );
imshow(nameWindow6,img_result);
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

    // resfile.close();
    // resfile.open ("result/result.txt");
    // resfile << idx_detect << endl;
    // resfile.close();

return 1;
}
