/* HOG DETECTOR
 *
 */

#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/cmd_line_parser.h>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <time.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;
using namespace dlib;
using namespace cv;

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

int C_ths_val = 0;
int C_ths_max = 255;

RNG rng(12345);

struct TrafficSign {
  string name;
  string svm_path;
  rgb_pixel color;
  TrafficSign(string name, string svm_path, rgb_pixel color) :
    name(name), svm_path(svm_path), color(color) {};
};

int main(int argc, char** argv) {
	try {
		command_line_parser parser;

		parser.add_option("h","Display this help message.");
		parser.add_option("u", "Upsample each input image <arg> times. Each \
		                  upsampling quadruples the number of pixels in the image \
		                  (default: 0).", 1);
		parser.add_option("wait","Wait user input to show next image.");
		parser.add_option("f","Show frame have sign detected");
		parser.add_option("t","Show time executed, don't use with \"v\"");
		
		parser.set_group_name("format input file sub-options");
		parser.add_option("v","Read video");
		parser.add_option("i","Read image");
	
		parser.parse(argc, argv);
		parser.check_option_arg_range("u", 0, 8);

		const char* one_time_opts[] = {"h","u","wait"};
		parser.check_one_time_options(one_time_opts);

		// Display help message
		if (parser.option("h")) {
			cout << "Usage: " << argv[0] << " [options] <list of images>" << endl;
			parser.print_options();

			return EXIT_SUCCESS;
		}
		if(!(parser.option("i") ||  parser.option("v")))
		{
			cout << "Error:" << endl;
			cout << "Choose format input files and try again" << endl;
			cout << "Use \"-h\" to view option" << endl;
			return EXIT_FAILURE;
		}

		if (parser.number_of_arguments() == 0) {
			cout << "You must give a list of input files." << endl;
			cout << "\nTry the -h option for more information." << endl;
			return EXIT_FAILURE;
		}
		
		typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

		// Load SVM detectors
		cout << "Loading SVM detectors..." << endl;
		std::vector<TrafficSign> signs;
		//signs.push_back(TrafficSign("PARE", "svm_detectors/pare_detector.svm", rgb_pixel(255,0,0)));
		//signs.push_back(TrafficSign("LOMBADA", "svm_detectors/lombada_detector.svm", rgb_pixel(255,122,0)));
		//signs.push_back(TrafficSign("PEDESTRE", "svm_detectors/pedestre_detector.svm", rgb_pixel(255,255,0)));

		signs.push_back(TrafficSign("object_detector", "svm_detectors/object_detector.svm", rgb_pixel(0,0,255)));

		signs.push_back(TrafficSign("oneway-exit", "resources/detectors/oneway-exit-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	    signs.push_back(TrafficSign("crossing", "resources/detectors/crossing-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	    signs.push_back(TrafficSign("give-way", "resources/detectors/give-way-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	    signs.push_back(TrafficSign("main", "resources/detectors/main-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	    signs.push_back(TrafficSign("parking", "resources/detectors/parking-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));

	    signs.push_back(TrafficSign("stop", "resources/detectors/stop-detector.svm", rgb_pixel(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255))));
		
		std::vector<object_detector<image_scanner_type> > detectors;

		for (int i = 0; i < signs.size(); i++) {
			object_detector<image_scanner_type> detector;
			deserialize(signs[i].svm_path) >> detector;
			detectors.push_back(detector);
		}

		//Open txt file
		cout << "Initing result file..." << endl;

		ofstream resfile;
  		resfile.open ("result/result.txt");

		if(parser.option("i"))
		{
			dlib::array<array2d<unsigned char> > images;

			images.resize(parser.number_of_arguments());

			cv::Mat img, stop, hsv;
			for (unsigned long imx = 0; imx < images.size(); ++imx) {

				Mat img_HSV, img_result, img_Grey, maskr, maskr1, maskr2, maskb, res, out, out1, edged;
    			Mat masks1,masks2;
    			std::vector<std::vector<cv::Point> > contours;
    			std::vector<Vec4i> hierarchy;

				img = imread(parser[imx]);

				img_result = img.clone();
        		// imshow(nameWindow1,img_raw);
        		blur(img_result, img_result, Size(2*2+1,2*2+1),Point(-1,-1),BORDER_DEFAULT);
       			// imshow(nameWindow3, blurred_img);
       			cvtColor(img_result, img_HSV, COLOR_BGR2HSV);

       			cout <<"image: " << imx << endl;
       			resfile << "image: " << imx<< endl;

       			inRange(img_HSV, Scalar(B_H_val, B_S_val, B_V_val), Scalar(B_H_max, B_S_max, B_V_max), maskb);
      			inRange(img_HSV, Scalar(R_H_val, R_S_val, R_V_val), Scalar(R_H_max, R_S_max, R_V_max), maskr1);
      			inRange(img_HSV, Scalar(R2_H_val, R2_S_val, R2_V_val), Scalar(R2_H_max, R2_S_max, R2_V_max), maskr2);
       			maskr = maskr1 + maskr2 + maskb;
       	
				dilate(maskr, masks1, Mat(), Point(-1, -1), 2, 1, 1);
				erode(masks1, masks1, Mat(), Point(-1, -1), 2, 1, 1);

				Canny(masks1, edged, C_ths_val, C_ths_max);

				findContours(edged, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	    		for(int idx = 0; idx < contours.size(); idx++)
	    		{
	        	// drawContours( img_result, contours, idx, Scalar(255,0,0), CV_FILLED, 8, hierarchy );
            		drawContours( edged, contours, idx, 255, CV_FILLED, 8, hierarchy );
	    		}

       			Canny(edged, edged, C_ths_val, C_ths_max);

        		// imshow("Edged",edged);

        		findContours(edged, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        		std::vector<Rect> boundRect( contours.size() );
				for (unsigned int i=0; i < contours.size(); i++) {

		            int area = contourArea(contours.at(i));
		            std::vector<rect_detection> rects;

		            if (area > 256) {
		                //RotatedRect rect = minAreaRect(contours.at(i));

		                //putText(img_result, "RED OBJECT", rect.center, FONT_HERSHEY_COMPLEX, 0.8, Scalar(255,255,255), 1);
		                //drawContours(img_result,contours,i,Scalar(255,255,255),1);
		            	boundRect[i] = boundingRect( Mat(contours[i]) );
		                if ((((float)boundRect[i].width/boundRect[i].height) > 0.5) &&(((float)boundRect[i].width/boundRect[i].height) < 1.3))
		                {
		                    
		                    // cout << "tl: " << boundRect[i].tl() <<" br: " << boundRect[i].br() <<endl;
		                    int tl_x = boundRect[i].tl().x;
		                    int tl_y = boundRect[i].tl().y;

		                    int br_x = boundRect[i].br().x;
		                    int br_y = boundRect[i].br().y;

		                    if(tl_x > 10) tl_x = tl_x - 10;
		                    else tl_x = 0;

							if(tl_y > 10) tl_y = tl_y - 10;
		                    else tl_y= 0;

		                    if(img_result.cols - br_x > 11) br_x = br_x + 10;
		                    else br_x = img_result.cols - 1;

							if(img_result.rows - br_y > 11) br_y = br_y + 10;
		                    else br_y= img_result.rows - 1;


		                    Rect roi(tl_x, tl_y, br_x - tl_x, br_y - tl_y);

		                    Mat image_roi = img_result(roi);

		                    Size size(80,80);
		                    resize(image_roi,image_roi,size);

		                    namedWindow( to_string(i), WINDOW_NORMAL );
		                    imshow( to_string(i),image_roi);

		                    // dlib::array<array2d<unsigned char> > images_HOG;
		                    cv_image<bgr_pixel> images_HOG(image_roi);

		                    evaluate_detectors(detectors, images_HOG, rects);

		                    if(rects.size() > 0)
		                    {
		                    	for (unsigned long j = 0; j < rects.size(); ++j) {
									//window.add_overlay(rects[j].rect, signs[rects[j].weight_index].color, signs[rects[j].weight_index].name);
									// cout << signs[rects[j].weight_index].name << " " <<rects[j].rect.top() << ";" << rects[j].rect.left() << endl;
									cout << "	" << signs[rects[j].weight_index].name <<": " << rects[j].detection_confidence << endl;
									cv::rectangle( img, boundRect[i].tl(), boundRect[i].br(), Scalar(255,0,0), 2, 8, 0 );
									resfile <<"		" <<  signs[rects[j].weight_index].name <<": " << rects[j].detection_confidence << endl;
								}
		                    }
		                    else
		                    {
		                        cv::rectangle( img, boundRect[i].tl(), boundRect[i].br(), Scalar(0,0,255), 2, 8, 0 );
		                    }
		                
		                }
		            }
		        }
		        imshow(to_string(imx),img);
		        
		        if (parser.option("wait")) {
					waitKey(0);
					destroyAllWindows();
			  	}

			}
			cout << "Saving file in result/result.txt" << endl;
  			resfile.close();
			return EXIT_SUCCESS;
		}
		if(parser.option("v"))
		{
			VideoCapture vid;
			string videofile = string(parser[0]);
			cout << "Opening videofile: \"" + videofile + "\"" << endl;
			vid.open(videofile);
			
			if (!vid.isOpened())
			{
				cout << "Video isn't opened" << endl;
				return EXIT_FAILURE;
			}
			
			cout << "Video loaded" << endl;

			cv::Mat img, stop, hsv;

			//set start
			int cnt = 0;

			vid.set(CV_CAP_PROP_POS_FRAMES, cnt);

			cout << "Video set time" << endl;

			image_window window;
			
			while (true)
			{
				Mat img_HSV, img_result, img_Grey, maskr, maskr1, maskr2, maskb, res, out, out1, edged;
    			Mat masks1,masks2;
    			std::vector<std::vector<cv::Point> > contours;
    			std::vector<Vec4i> hierarchy;

				vid >> img;

				if (img.empty())
				{
					cout << "Video end" << endl;
					cout << "Saving file in result/result.txt" << endl;
  					resfile.close();
					break;
				}
					
				cnt++;

				img_result = img.clone();

        		blur(img_result, img_result, Size(2*2+1,2*2+1),Point(-1,-1),BORDER_DEFAULT);

       			cvtColor(img_result, img_HSV, COLOR_BGR2HSV);
       			inRange(img_HSV, Scalar(B_H_val, B_S_val, B_V_val), Scalar(B_H_max, B_S_max, B_V_max), maskb);
      			inRange(img_HSV, Scalar(R_H_val, R_S_val, R_V_val), Scalar(R_H_max, R_S_max, R_V_max), maskr1);
      			inRange(img_HSV, Scalar(R2_H_val, R2_S_val, R2_V_val), Scalar(R2_H_max, R2_S_max, R2_V_max), maskr2);
       			maskr = maskr1 + maskr2 + maskb;
       	
				dilate(maskr, masks1, Mat(), Point(-1, -1), 2, 1, 1);
				erode(masks1, masks1, Mat(), Point(-1, -1), 2, 1, 1);

				Canny(masks1, edged, C_ths_val, C_ths_max);

				findContours(edged, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	    		for(int idx = 0; idx < contours.size(); idx++)
	    		{
	        	// drawContours( img_result, contours, idx, Scalar(255,0,0), CV_FILLED, 8, hierarchy );
            		drawContours( edged, contours, idx, 255, CV_FILLED, 8, hierarchy );
	    		}

       			Canny(edged, edged, C_ths_val, C_ths_max);

        		findContours(edged, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        		std::vector<Rect> boundRect( contours.size() );
				for (unsigned int i=0; i < contours.size(); i++) {

		            int area = contourArea(contours.at(i));
		            std::vector<rect_detection> rects;

		            if (area > 256) {
		                //RotatedRect rect = minAreaRect(contours.at(i));

		                //putText(img_result, "RED OBJECT", rect.center, FONT_HERSHEY_COMPLEX, 0.8, Scalar(255,255,255), 1);
		                //drawContours(img_result,contours,i,Scalar(255,255,255),1);
		                boundRect[i] = boundingRect( Mat(contours[i]) );
		                if ((((float)boundRect[i].width/boundRect[i].height) > 0.5) &&(((float)boundRect[i].width/boundRect[i].height) < 1.2))
		                {
		                    
		                    // cout << "tl: " << boundRect[i].tl() <<" br: " << boundRect[i].br() <<endl;
		                    Rect roi(boundRect[i].tl().x, boundRect[i].tl().y,
		                        boundRect[i].br().x - boundRect[i].tl().x, boundRect[i].br().y - boundRect[i].tl().y);
		                    Mat image_roi = img_result(roi);

		                    Size size(80,80);
		                    resize(image_roi,image_roi,size);

		                    namedWindow( "Roi Image", WINDOW_NORMAL );
		                    imshow("Roi Image",image_roi);

		                    // dlib::array<array2d<unsigned char> > images_HOG;
		                    cv_image<bgr_pixel> images_HOG(image_roi);

		                    evaluate_detectors(detectors, images_HOG, rects);

		                    if(rects.size() > 0)
		                    {
		                    	for (unsigned long j = 0; j < rects.size(); ++j) {
									//window.add_overlay(rects[j].rect, signs[rects[j].weight_index].color, signs[rects[j].weight_index].name);
									cout << "	" << signs[rects[j].weight_index].name << "	" << rects[j].rect << endl;
									cv::rectangle( img, boundRect[i].tl(), boundRect[i].br(), Scalar(255,0,0), 2, 8, 0 );
									resfile <<"		" <<  signs[rects[j].weight_index].name <<": " << rects[j].detection_confidence << endl;
								}
		                    }
		                    else
		                    {
		                        cv::rectangle( img, boundRect[i].tl(), boundRect[i].br(), Scalar(0,0,255), 2, 8, 0 );
		                    }
		                
		                }
		            }
		        }       		

				imshow("Result",img);
		        if (waitKey(1) == 27)
		            break;
			}
		}
	}
	catch (exception& e) {
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}
