#include "api_kinect_cv.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "openni2.h"
#include "../openni2/Singleton.h"
#include <unistd.h>
#include "../sign_detection/SignDetection.h"
#include <chrono>
#include "signsRecognizer.h"
#include "extractInfo.h"
#include "SignRecognition.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdlib.h>
// #include "multilane.h"
// #include "Hal.h"
// #include "LCDI2C.h"
// #include "api_i2c_pwm.h"

#include "../ObjectRecognition/SignRecognition.h"
using namespace openni;
using namespace framework;
using namespace signDetection;
using namespace EmbeddedFramework;
using namespace cv;
using namespace std;
//using namespace SignRecognition;
#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms


char analyzeFrame(const VideoFrameRef& frame_depth,const VideoFrameRef& frame_color,Mat& depth_img, Mat& color_img) {
    DepthPixel* depth_img_data;
    RGB888Pixel* color_img_data;

    int w = frame_color.getWidth();
    int h = frame_color.getHeight();

    depth_img = Mat(h, w, CV_16U);
    color_img = Mat(h, w, CV_8UC3);
    Mat depth_img_8u;
	

            depth_img_data = (DepthPixel*)frame_depth.getData();

            memcpy(depth_img.data, depth_img_data, h*w*sizeof(DepthPixel));

            normalize(depth_img, depth_img_8u, 255, 0, NORM_MINMAX);

            depth_img_8u.convertTo(depth_img_8u, CV_8U);
            color_img_data = (RGB888Pixel*)frame_color.getData();

            memcpy(color_img.data, color_img_data, h*w*sizeof(RGB888Pixel));

            cvtColor(color_img, color_img, COLOR_RGB2BGR);
		
            return 'c';
}

// SignRecognition::SignRecognize a;
// signDetection::SignDetection detection;
int main( int argc, char** argv )
{
    SignDetection::Detection detec;
	SignRecognition::SignRecognize a;
    double st = 0, et = 0, fps = 0;
    double freq = getTickFrequency();

    signsRecognizer recognizer = signsRecognizer("/home/ubuntu/data/new_templates/templates.txt");
// /// Init openNI ///
//     Status rc;
//     Device device;
//     Mat depthImg, colorImg, grayImage;

//     VideoStream depth, color;
//     rc = OpenNI::initialize();
//     if (rc != STATUS_OK) {
//         printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
//         return 0;
//     }
//     rc = device.open(ANY_DEVICE);
//     if (rc != STATUS_OK) {
//         printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
//         return 0;
//     }

    
//   VideoFrameRef frame_depth, frame_color;
//     VideoStream* streams[] = {&depth, &color};
/// End of openNI init phase ///
    
    while ( true ) {
        st = getTickCount();
        if( argc != 2){
        cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
        }

        Mat image;
        image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

        if(! image.data ){
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        //     int readyStream = -1;
        //     rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
		//     if (rc != STATUS_OK)
		//     {
		//         printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
		//         break;
		//     }

		// depth.readFrame(&frame_depth);
		// color.readFrame(&frame_color);
		// char recordStatus = analyzeFrame(frame_depth,frame_color, depthImg, colorImg);
		// flip(depthImg, depthImg, 1);
		// flip(colorImg, colorImg, 1);
        Size size(60,60);
        Mat resize_img;
        resize(image,resize_img,size);
        
        namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        imshow( "Display window", resize_img );                   // Show our image inside it.
        clock_t begin_time = clock();
        int ans = a.recognizeSign(resize_img);
                //std::cerr << 1 + (ans - 1) / 5  << '\n';
        cout<<"Bien bao so "<< ans << ":))" << ' ' << float( clock () - begin_time ) /  CLOCKS_PER_SEC << '\n';  
        waitKey(0);	
    }                                
    return 0;
}



