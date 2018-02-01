#include <iostream>
#include <stdexcept>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <array>
using namespace cv;
using namespace std;
const char* keys =
{
    "{ help h      |   | print help message }"
    "{ image i     |   | specify input image}"
    "{ video v     |   | use video as input }"
    "{ frame f     |   | specify frame of video}"
    "{ start s     |0  | start frame}"
};

bool bpause = false;
bool capture = false;

#define ROI_X 99
#define ROI_Y 329
#define ROI_W 440
#define ROI_H 150
#define Ys 150
#define Ye 0



void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if(event== EVENT_LBUTTONDOWN)
	{
		bpause ^= true;
	}
}

static std::array<int, 3> cross(const std::array<int, 3> &a, 
    const std::array<int, 3> &b)
{
    std::array<int, 3> result;
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}

static double point_to_line_distance(const cv::Point &p, const cv::Vec4i &line)
{
    std::array<int, 3> pa{ { line[0], line[1], 1 } };
    std::array<int, 3> pb{ { line[2], line[3], 1 } };
    std::array<int, 3> l = cross(pa, pb);
    return std::abs((p.x * l[0] + p.y * l[1] + l[2])) * 1.0 /
        std::sqrt(double(l[0] * l[0] + l[1] * l[1]));
}

static void movingAverage(double &avg, double newSample)
{
    
    if (avg == 0)
        avg = newSample;
    else
    {
        int N = 5;
        avg -= avg / N;
        avg += newSample / N;
    }
}

double getTheta(Point car, Point dst) {
    if (dst.x == car.x) return 0;
    if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
    double pi = acos(-1.0);
    double dx = dst.x - car.x;
    double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}

RNG rng(12345);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        cout << "\nThis program is Lane Detection\n";
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
    int start_frame;

	if (parser.has("image"))
    {
        filename = parser.get<string>("image");
        frame = imread(filename);
    }
    else if ((parser.has("video")) && (!parser.has("frame")))
    {
    	isVideo = true;
        filename = parser.get<string>("video");
        start_frame = parser.get<int>("start");
        vid.open(filename);
        vid.set( CAP_PROP_POS_FRAMES, start_frame );
		if (!vid.isOpened())
		{
			cout << "Video isn't opened" << endl;
			return EXIT_FAILURE;
		}
    }
    else if ((parser.has("video")) && (parser.has("frame")))
    {
        filename = parser.get<string>("video");
        start_frame = parser.get<int>("frame");
        vid.open(filename);
        vid.set( CAP_PROP_POS_FRAMES, start_frame );
        vid >> frame;
        if (!vid.isOpened())
        {
            cout << "Video isn't opened" << endl;
            return EXIT_FAILURE;
        }
    }

    namedWindow( "result" , WINDOW_AUTOSIZE); //WINDOW_NORMAL );
    setMouseCallback("result", CallBackFunc, NULL);
    int cnt=0 + start_frame;


    double avgThelta_L=0, avgThelta_R=0, avgXs_L=0, avgXs_R=0, avgXe_L=0, avgXe_R=0;
    for (;;)
    {
    	if (isVideo)
    	{
    		if(!bpause)
            {
                cout << cnt++ << endl;
                vid >> frame;
            }
			
    	}

    	if (frame.empty())
    	{
    		cout << "Errors read frame" << endl;
    		break;
    	}

    	resize(frame,frame,Size(640,480));

    	Mat im_result = frame.clone();

    	Rect ROI(ROI_X, ROI_Y, ROI_W, ROI_H);

    	// cv::rectangle( im_result, ROI.tl(), ROI.br(), CV_RGB(0,255,0), 2, 8, 0 );

    	Mat im_raw = frame(ROI).clone();

        
        /////////////////////////////////////////////////////////////////////////////
        //Start process
        /////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////
        //Convert to HSV colorspace and threshold
        /////////////////////////////////////////////////////////////////////////////
        Mat im_hsv, im_th_hsv_w;
        cvtColor(im_raw, im_hsv, COLOR_BGR2HSV);
        std::vector<Mat> hsv;
        split(im_hsv, hsv);
        // imshow("H", hls[0]);
        // imshow("S", hls[1]);
        // imshow("V", hls[2]);
        double min, max;
        cv::minMaxLoc(hsv[2], &min, &max);
        int sensity = 0.8*max;
        inRange(im_hsv, Scalar(0, 0, sensity), Scalar(255, 255-sensity, 255), im_th_hsv_w);
        // imshow("HLS", ~im_th_hsv_w);

        /////////////////////////////////////////////////////////////////////////////
        //Convert to Grayscale
        ///////////////////////////////////////////////////////////////////////////// 
        Mat im_gray;
        cvtColor(im_raw, im_gray, COLOR_BGR2GRAY);
        Mat im_blur;
        GaussianBlur(im_gray, im_blur, Size(7, 7), 1, 1 );

        /////////////////////////////////////////////////////////////////////////////
        //adaptiveThreshold
        /////////////////////////////////////////////////////////////////////////////
        Mat im_ad;
        adaptiveThreshold(im_blur, im_ad, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 55, 1.0);
        Mat im_median;
        medianBlur(im_ad, im_median, 5);
        // imshow("~median", ~im_median);

        /////////////////////////////////////////////////////////////////////////////
        //half-dynamic threshold
        /////////////////////////////////////////////////////////////////////////////
        // double min, max;
        // cv::minMaxLoc(im_blur, &min, &max);
        // int th_val = (max-0.25*max);
        // Mat im_hdy;
        // threshold(im_blur, im_hdy, th_val, 255, CV_THRESH_BINARY);
        // imshow("half-dynamic", im_hdy);

        /////////////////////////////////////////////////////////////////////////////
        //Canny
        /////////////////////////////////////////////////////////////////////////////
        // Mat im_canny;
        // Canny(im_blur, im_canny, 50, 200, 3);
        // imshow("canny", im_canny);
        // Mat kernel = Mat::ones(5, 5, CV_8UC1);
        // dilate(im_canny, im_canny, kernel);
        // erode(im_canny, im_canny, kernel);

        /////////////////////////////////////////////////////////////////////////////
        // find the contours
        /////////////////////////////////////////////////////////////////////////////
        // vector< vector<Point> > contours;
        // findContours(im_canny, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        // // you could also reuse img1 here
        // Mat im_fill = Mat::zeros(im_canny.rows, im_canny.cols, CV_8UC1);
        // // CV_FILLED fills the connected components found
        // drawContours(im_fill, contours, -1, Scalar(255), CV_FILLED);

        /////////////////////////////////////////////////////////////////////////////
        //Sobel
        /////////////////////////////////////////////////////////////////////////////
        Mat im_gradx, im_abs_gradx, im_sobel;
        Sobel(im_blur, im_gradx, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs(im_gradx, im_abs_gradx);
        inRange(im_abs_gradx, 20, 100, im_sobel);
        // imshow("Sobel", im_sobel);

        /////////////////////////////////////////////////////////////////////////////
        //find yellow lane
        /////////////////////////////////////////////////////////////////////////////
        Mat im_hls, im_th_hls_y, im_th_bgr_y;
        cvtColor(im_raw, im_hls, COLOR_BGR2HLS);
        inRange(im_hls, Scalar(20, 80, 120), Scalar(45, 255, 200), im_th_hls_y);
        inRange(im_raw, Scalar(0, 80, 100), Scalar(170, 220, 220), im_th_bgr_y);
        // imshow("Yellow bgr", im_th_bgr_y);
        // imshow("Yellow hls", im_th_hls_y);
        Mat im_bw_y = im_th_hls_y & im_th_bgr_y;


        /////////////////////////////////////////////////////////////////////////////
        //Combined masks
        /////////////////////////////////////////////////////////////////////////////
        Mat im__ = im_th_hsv_w & ~(~im_median & im_sobel) | im_bw_y;
        vector< vector<Point> > contours;
        Mat im_fill = Mat::zeros(im__.rows, im__.cols, CV_8UC1);
        findContours(im__, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        for(int idx = 0; idx < contours.size(); idx++)
        {
            RotatedRect rr = minAreaRect(contours[idx]);
            if  (
                ((rr.size.height && rr.size.width) != 0) &&
                (
                    ((rr.size.height > 3 )  &&  (rr.size.height < 30 )  &&  (rr.size.width > rr.size.height))   || 
                    ((rr.size.height < 3 )  &&  (rr.size.width > 5*rr.size.height)) ||

                    ((rr.size.width > 3 )   &&  (rr.size.width < 30 )    &&  (rr.size.height > 2*rr.size.width)) ||
                    ((rr.size.width < 3 )   &&  (rr.size.height > 5*rr.size.width))
                )
                )
            {
                drawContours(im_fill, contours, idx, Scalar(255), CV_FILLED);
            }
        }
        Mat im_bw = im_fill;
        Mat im_cvt_rgb;

        cvtColor(im__, im_cvt_rgb, COLOR_GRAY2RGB);
        im_cvt_rgb.copyTo(im_result(Rect(ROI_X,0,ROI_W,ROI_H)));

        // cvtColor(im__, im_cvt_rgb, COLOR_GRAY2RGB);
        // im_cvt_rgb.copyTo(im_result(Rect(ROI_X,ROI_H,ROI_W,ROI_H)));

        /////////////////////////////////////////////////////////////////////////////
        //HoughLine
        /////////////////////////////////////////////////////////////////////////////
        vector<Vec4i> lines;
        HoughLinesP(im_bw, lines, 1, CV_PI/180, 20, 15, 50 ); //vote; min line; max gap

        /////////////////////////////////////////////////////////////////////////////
        //Init points
        /////////////////////////////////////////////////////////////////////////////
        int c_right=0, c_left=0;
        double angle_l=0, angle_r=0;
        Point StL_, EnL_;
        Point StR_, EnR_;
        int MaxL_ = 0, MaxR_ = 0;
        int MinL_ = ROI_W, MinR_ = ROI_W;

        /////////////////////////////////////////////////////////////////////////////
        //filter lines side
        /////////////////////////////////////////////////////////////////////////////
        Mat im_drawlines = Mat::zeros(im_bw.size(), CV_8UC1);
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            // raito = float(l[1] / row_size );
            double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
            double d = point_to_line_distance(Point(220,149), l);

            Point En_(l[0], l[1]);
            Point St_(l[2], l[3]);

            line(im_drawlines, St_, En_, 255, 2);

            //RIGHT side
            if (angle>20 && angle<80)
            {
                c_right++;
                angle_r += angle;
                if ((d < MinR_) && (St_.x > 140))
                {
                    MinR_ = d;
                    StR_ = St_;
                    EnR_ = En_;
                }             
            }

            //LEFT side
            if (angle>-80  && angle<-20) 
            {
                c_left++;
                angle_l += angle;
                if ((d < MinL_) && (St_.x < 300))
                {
                    MinL_ = d;
                    StL_ = St_;
                    EnL_ = En_;
                } 
            }
        }
        // Mat im_lines_rgb;
        // cvtColor(im_drawlines, im_lines_rgb, COLOR_GRAY2RGB);
        // im_lines_rgb.copyTo(im_result(Rect(ROI_X, ROI_H+4, ROI_W, ROI_H)));
        
        /////////////////////////////////////////////////////////////////////////////
        //Compute 2 point of RIGHT line
        /////////////////////////////////////////////////////////////////////////////
        if ((c_right > 0) && (MinR_ != ROI_W ))
        {
            int thelta = angle_r/c_right;
            movingAverage(avgThelta_R, thelta);
            // int Ys = 150;
            double Xs = StR_.x + (Ys - StR_.y) / tan(avgThelta_R * CV_PI / 180.0);
            // int Ye = 0;
            double Xe = EnR_.x + (Ye - EnR_.y) / tan(avgThelta_R * CV_PI / 180.0);
            movingAverage(avgXs_R, Xs);
            movingAverage(avgXe_R, Xe);

            ellipse(im_result, Point(ROI_X+ROI_W, ROI_Y), Size(20, 5), 0 , 0, 360, CV_RGB(0,200,0), -1, 8, 0);
        }
        else 
        {
            movingAverage(avgThelta_R, avgThelta_R);
            movingAverage(avgXs_R, avgXs_R);
            movingAverage(avgXe_R, avgXe_R);

            ellipse(im_result, Point(ROI_X+ROI_W, ROI_Y), Size(20, 5), 0 , 0, 360, CV_RGB(200,0,0), -1, 8, 0);
        }

        /////////////////////////////////////////////////////////////////////////////
        //Compute 2 point of LEFT line
        /////////////////////////////////////////////////////////////////////////////
        if ((c_left > 0) && (MinL_ != ROI_W))
        {
            int thelta = angle_l/c_left;
            movingAverage(avgThelta_L, thelta);
            // int Ys = 150;
            double Xs = StL_.x + (Ys - StL_.y) / tan(avgThelta_L * CV_PI / 180.0);
            // int Ye = 0;
            double Xe = EnL_.x + (Ye - EnL_.y) / tan(avgThelta_L * CV_PI / 180.0);
            movingAverage(avgXs_L, Xs);
            movingAverage(avgXe_L, Xe);

            ellipse(im_result, Point(ROI_X, ROI_Y), Size(20, 5), 0 , 0, 360, CV_RGB(0,200,0), -1, 8, 0);
        }
        else 
        {
            movingAverage(avgThelta_L, avgThelta_L);
            movingAverage(avgXs_L, avgXs_L);
            movingAverage(avgXe_L, avgXe_L);
            
            ellipse(im_result, Point(ROI_X, ROI_Y), Size(20, 5), 0 , 0, 360, CV_RGB(200,0,0), -1, 8, 0);
        }

        /////////////////////////////////////////////////////////////////////////////
        //Draw poly
        /////////////////////////////////////////////////////////////////////////////
        Point rook_points[1][4];
        rook_points[0][0] = Point( avgXs_L+ROI_X, 479 );
        rook_points[0][1] = Point( avgXe_L+ROI_X, ROI_Y );
        rook_points[0][2] = Point( avgXe_R+ROI_X, ROI_Y );
        rook_points[0][3] = Point( avgXs_R+ROI_X, 479 );
        const Point* ppt[1] = { rook_points[0] };
        int npt[] = { 4 };

        Mat im_polly = Mat::zeros(im_result.size(), CV_8UC3);
        fillPoly( im_polly, ppt, npt, 1, CV_RGB( 10, 190, 255), 8 );

        /////////////////////////////////////////////////////////////////////////////
        //Draw lines
        /////////////////////////////////////////////////////////////////////////////
        line(im_polly, Point(avgXs_R + ROI_X, Ys + ROI_Y), Point(avgXe_R + ROI_X, Ye + ROI_Y), CV_RGB(255,125,0), 2);//right
        line(im_polly, Point(avgXs_L + ROI_X, Ys + ROI_Y), Point(avgXe_L + ROI_X, Ye + ROI_Y), CV_RGB(255,125,0), 2);//left

        addWeighted(im_polly, 0.25, im_result, 1, 0.75, im_result);

        /////////////////////////////////////////////////////////////////////////////
        //Compute center line
        /////////////////////////////////////////////////////////////////////////////
        int PR1_x = avgXs_R + (75 - 150) / tan(avgThelta_R * CV_PI / 180.0);
        int PL1_x = avgXs_L + (75 - 150) / tan(avgThelta_L * CV_PI / 180.0);
        int CT1_x = (PL1_x+PR1_x)/2;

        int PR2_x = avgXs_R + (100 - 150) / tan(avgThelta_R * CV_PI / 180.0);
        int PL2_x = avgXs_L + (100 - 150) / tan(avgThelta_L * CV_PI / 180.0);
        int CT2_x = (PL2_x+PR2_x)/2;

        // circle(im_result,Point(PL1_x + ROI_X, 75 + ROI_Y), 2, CV_RGB(255,0,0), -1);
        // circle(im_result,Point(PR1_x + ROI_X, 75 + ROI_Y), 2, CV_RGB(255,0,0), -1);
        // circle(im_result,Point(CT1_x + ROI_X, 75 + ROI_Y), 2, CV_RGB(255,0,0), -1);

        // circle(im_result,Point(PL2_x + ROI_X, 100 + ROI_Y), 2, CV_RGB(255,0,0), -1);
        // circle(im_result,Point(PR2_x + ROI_X, 100 + ROI_Y), 2, CV_RGB(255,0,0), -1);
        // circle(im_result,Point(CT2_x+ ROI_X, 100 + ROI_Y), 2, CV_RGB(255,0,0), -1);

        // line(im_result, Point(CT1_x + ROI_X, 75 + ROI_Y), Point(CT2_x + ROI_X, 100 + ROI_Y), CV_RGB(255,0,255), 2);
        // line(im_result, Point((avgXs_R + avgXs_L) /2+ ROI_X, Ys + ROI_Y), Point(CT2_x + ROI_X, 100 + ROI_Y), CV_RGB(255,0,255), 2);

        // line(im_result, Point(ROI_W/2 + ROI_X, 100 + ROI_Y), Point(CT2_x + ROI_X, 100 + ROI_Y), CV_RGB(0,255,0), 4);

        line(im_result, Point(ROI_W/2 + ROI_X, 479), Point(ROI_W/2 + ROI_X, 399), CV_RGB(0,255,0), 2);
        // line(im_result, Point(ROI_W/2 + ROI_X, ROI_H-1  + ROI_Y), Point((CT1_x + CT2_x)/2 + ROI_X,(100+75)/2  + ROI_Y), CV_RGB(0,255,0), 2);

        circle(im_result,Point((CT1_x + CT2_x)/2 + ROI_X,(100+75)/2  + ROI_Y), 2, CV_RGB(255,0,0), -1);
        double app_thelta = getTheta(Point(ROI_W/2,ROI_H-1), Point((CT1_x + CT2_x)/2,(100+75)/2));

        if (app_thelta > 0)
            putText(im_result, to_string(app_thelta) + "    LEFT", Point(15,15), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255), 1, CV_AA);
        else putText(im_result, to_string(app_thelta) + "   RIGHT", Point(15,15), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,255), 1, CV_AA);
        /////////////////////////////////////////////////////////////////////////////
        //Result
        /////////////////////////////////////////////////////////////////////////////
        imshow("result", im_result);
        int c = waitKey( (vid.isOpened() && isVideo) ? 1 : 0 ) & 255;
        if ( c == 'q' || c == 'Q' || c == 27)
            break;
    }
    return 0;
}














void BEV(Mat source, Mat& destination){
    int alpha_ = 90, beta_ = 45, gamma_ = 90;
    int f_ = 500, dist_ = 500;

    double focalLength, dist, alpha, beta, gamma; 
    alpha =((double)alpha_ -90) * CV_PI/180;
    beta =((double)beta_ -90) * CV_PI/180;
    gamma =((double)gamma_ -90) * CV_PI/180;
    focalLength = (double)f_;
    dist = (double)dist_;

    Size image_size = source.size();
    double w = (double)image_size.width, h = (double)image_size.height;

    // Projecion matrix 2D -> 3D
    Mat A1 = (Mat_<float>(4, 3)<< 
        1, 0, -w/2,
        0, 1, -h/2,
        0, 0, 0,
        0, 0, 1 );


    // Rotation matrices Rx, Ry, Rz

    Mat RX = (Mat_<float>(4, 4) << 
        1, 0, 0, 0,
        0, cos(alpha), -sin(alpha), 0,
        0, sin(alpha), cos(alpha), 0,
        0, 0, 0, 1 );

    Mat RY = (Mat_<float>(4, 4) << 
        cos(beta), 0, -sin(beta), 0,
        0, 1, 0, 0,
        sin(beta), 0, cos(beta), 0,
        0, 0, 0, 1  );

    Mat RZ = (Mat_<float>(4, 4) << 
        cos(gamma), -sin(gamma), 0, 0,
        sin(gamma), cos(gamma), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1  );


    // R - rotation matrix
    Mat R = RX * RY * RZ;

    // T - translation matrix
    Mat T = (Mat_<float>(4, 4) << 
        1, 0, 0, 0,  
        0, 1, 0, 0,  
        0, 0, 1, dist,  
        0, 0, 0, 1); 

    // K - intrinsic matrix 
    Mat K = (Mat_<float>(3, 4) << 
        focalLength, 0, w/2, 0,
        0, focalLength, h/2, 0,
        0, 0, 1, 0
        ); 

    Mat transformationMat = K * (T * (R * A1));

    warpPerspective(source, destination, transformationMat, image_size, INTER_CUBIC | WARP_INVERSE_MAP);
}

void transform(Mat& src, Mat &dst){
    //transform
    Point2f src_vertices[4];
    src_vertices[0] = Point(160, 0);
    src_vertices[1] = Point(280, 0);
    src_vertices[2] = Point(439, 100);
    src_vertices[3] = Point(0, 100);
    Point2f dst_vertices[4];
    dst_vertices[0] = Point(0, 0);
    dst_vertices[1] = Point(439, 0);
    dst_vertices[2] = Point(439, 149);
    dst_vertices[3] = Point(0, 149);

    Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
    cout << M << endl;
    Mat im_transform;
    // warpPerspective(im_bw, im_transform, M, im_transform.size(), INTER_LINEAR, BORDER_CONSTANT);
    // imshow("transform", im_transform);
}
