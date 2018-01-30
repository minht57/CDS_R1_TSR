#include <iostream>
#include <stdexcept>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
using namespace cv;
using namespace std;
const char* keys =
{
    "{ help h      |   | print help message }"
    "{ image i     |   | specify input image}"
    "{ video v     |   | use video as input }"
};

bool bpause = false;
bool capture = false;


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if(event== EVENT_LBUTTONDOWN)
	{
		bpause ^= true;
	}
}

RNG rng(12345);

void skeleton(Mat& img, Mat& out)
{
	cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;		
	do
	{
	  cv::erode(img, eroded, element);
	  cv::dilate(eroded, temp, element); // temp = open(img)
	  cv::subtract(img, temp, temp);
	  cv::bitwise_or(skel, temp, skel);
	  eroded.copyTo(img);

	  done = (cv::countNonZero(img) == 0);
	} while (!done);
	out = skel;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        cout << "\nThis program demonstrates the use of the HoG descriptor using\n"
            " HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n";
        parser.printMessage();
        cout << "During execution:\n\tHit q or ESC key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n"
            "Note: camera device number must be different from -1.\n" << endl;
        return 0;
    }

    string filename = "";

    VideoCapture vid;
    Mat frame;
    bool isVideo = false;

	if (parser.has("image"))
    {
        filename = parser.get<string>("image");
        frame = imread(filename);
    }
    else if (parser.has("video"))
    {
    	isVideo = true;
        filename = parser.get<string>("video");
        vid.open(filename);
		if (!vid.isOpened())
		{
			cout << "Video isn't opened" << endl;
			return EXIT_FAILURE;
		}
    }

    namedWindow( "result" , WINDOW_AUTOSIZE); //WINDOW_NORMAL );
    setMouseCallback("result", CallBackFunc, NULL);

    for (;;)
    {
    	if (isVideo)
    	{
    		if(!bpause)
				vid >> frame;
			// cnt++;
    	}

    	if (frame.empty())
    	{
    		cout << "Errors read frame" << endl;
    		break;
    	}

    	resize(frame,frame,Size(640,480));

    	Mat image_result = frame.clone();

    	Rect ROI(99, 329, 440, 150);

    	cv::rectangle( image_result, ROI.tl(), ROI.br(), Scalar(0,255,0), 2, 8, 0 );

    	Mat src = frame(ROI).clone();

    	// imshow("src", src);
    	/*to do:*/
    	Mat gray;
    	cvtColor(src, gray, COLOR_BGR2GRAY);

    	Mat blur;
    	GaussianBlur(gray, blur, Size(7, 7), 1, 1 );

    	// Mat ad_th;
    	// adaptiveThreshold(blur, ad_th, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 5);

    	// Mat edge = ~ad_th;

    	double min, max;
		cv::minMaxLoc(blur, &min, &max);
		int th_val = (max-0.25*max);

    	Mat im_th;
    	threshold(blur, im_th, th_val, 255, CV_THRESH_BINARY);

    	Mat src_th;
    	gray.copyTo(src_th, im_th);

    	
    	Mat im_otsu = src_th > 0;
    	Mat im_rs;
    	threshold(im_otsu, im_rs, 0, 255, CV_THRESH_OTSU);

    	// Mat sk;
    	// skeleton(im_rs, sk);

    	// imshow("im_rs",sk);

    	Mat edge;
		Canny(im_rs, edge, 150, 300, 5);

		// Mat edge_inv;
		// threshold(edge,edge_inv,128,255,THRESH_BINARY_INV);

		imshow("edge", im_rs);

		
		vector<Point> points_right;
		vector<Vec4i> lines_left;

		Mat drawlines = Mat::zeros(edge.size(), CV_8UC1);

		std::vector<std::vector<cv::Point> > contours;
		std::vector<Vec4i> hierarchy;
		findContours(im_rs, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		if (contours.size() != 0)
		{
			for(int i = 0; i < contours.size(); i++) 
			{
				Vec4i lines;

				fitLine(Mat(contours[i]),lines,2,0,0.01,0.01);

				int x0 = lines[2 ];                       // a point on the line
				int y0 = lines[3 ];                    
				int x1 = x0 - 10 * lines[0 ];     // add a vector of length 200
				int y1 = y0 - 10 * lines[1 ];

				line( image_result, Point(x0+99, y0+329), Point(x1+99, y1+329), Scalar(0,0255), 1);

				// fitLine(Mat(contours[i]), line, CV_DIST_L2, 0, 0.01, 0.01);

				// int lefty = (-lines[2]*lines[1]/lines[0])+lines[3];
				// int righty = ((im_rs.cols-lines[2])*lines[1]/lines[0])+lines[3];

				// cv::line(drawlines,Point(im_rs.cols-1,righty),Point(0,lefty),255,1);
			}
		}

		// imshow("drawlines", drawlines);


		// HoughLinesP(edge, lines, 1, CV_PI/180, 33, 20, 200 ); //vote; min line; max gap

		// cout << "size: " << lines.size() << endl; 
		// for( size_t i = 0; i < lines.size(); i++ )
		// {
		//     Vec4i l = lines[i];
		//     // raito = float(l[1] / row_size );
		//     double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;

		    
	 //        // line( image_result, Point(l[0]+99, l[1]+329), Point(l[2]+99, l[3]+329), Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)), 1);
	        
	 //        //right side
	 //        if (angle>20 && angle<80)
	 //        {

	 //        	line( drawlines, Point(l[0], l[1]), Point(l[2], l[3]), 255, 1);
	 //        	// cout << "angle: " << angle << endl;
	 //        	// points_right.push_back(Point(l[0], l[1]));
	 //        	// points_right.push_back(Point(l[2], l[3]));
	 //        	// points_right.push_back(Point((l[0]+l[2])/2, (l[1]+l[3])/2));

	 //        	// circle(image_result, Point(l[0]+99, l[1]+329), 1, CV_RGB(255,0,0), -1);
	 //        	// circle(image_result, Point(l[2]+99, l[3]+329), 1, CV_RGB(255,0,0), -1);
	 //        	// circle(image_result, Point((l[0]+l[2])/2+99, (l[1]+l[3])/2+329), 1, CV_RGB(255,0,0), -1);
	 //        	// line( image_result, Point(l[0]+99, l[1]+329), Point(l[2]+99, l[3]+329), Scalar(0,0,255), 2);
	 //        }

	 //        //left side
	 //        if (angle>-80  && angle<-20) 
	 //        {
	 //        	lines_left.push_back(lines[i]);
	 //        	// line( image_result, Point(l[0]+99, l[1]+329), Point(l[2]+99, l[3]+329), Scalar(255,0,0), 2);
	 //        }
		// }

		// Mat leftside = drawlines & im_rs;

		// imshow("leftside", leftside);

		// fit line
		// Mat drawlines = Mat::zeros(edge.size(), CV_8UC1);
		// Vec4i line_right;
		// fitLine(Mat(points_right), line_right, CV_DIST_L2 , 0, 0.01, 0.01);

		// int x0 = line_right[2 ];                       // a point on the line
		// int y0 = line_right[3 ];                    
		// int x1 = x0 - 200 * line_right[0 ];     // add a vector of length 200
		// int y1 = y0 - 200 * line_right[1 ];

		// line( drawlines, Point(x0, y0), Point(x1, y1), 255, 1);
		// imshow("drawlines",drawlines);
    	/*end*/

    	// imshow("im_rs", im_rs);
    	// imshow("frame", frame);
    	// imshow("edge_inv", im_rs);
    	imshow("result", image_result);
    	int c = waitKey( vid.isOpened() ? 27 : 0 ) & 255;
    	if ( c == 'q' || c == 'Q' || c == 27)
            break;
    }
    return 0;
}