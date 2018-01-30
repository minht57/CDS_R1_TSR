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

RNG rng(12345);

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
    else if ((parser.has("video")) && (!parser.has("frame")))
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
    else if ((parser.has("video")) && (parser.has("frame")))
    {
        filename = parser.get<string>("video");
        int start_frame = parser.get<int>("frame");
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
    int cnt=0;
    for (;;)
    {
    	if (isVideo)
    	{
    		if(!bpause)
            {
				vid >> frame;
                cout << cnt++ << endl;
            }
			
    	}

    	if (frame.empty())
    	{
    		cout << "Errors read frame" << endl;
    		break;
    	}

    	resize(frame,frame,Size(640,480));

    	Mat im_result = frame.clone();

    	Rect ROI(99, 329, 440, 150);

    	// cv::rectangle( im_result, ROI.tl(), ROI.br(), Scalar(0,255,0), 2, 8, 0 );

        // line(im_result,Point(160+99, 0+329),Point(0+99, 100+329),Scalar(0,255,0),2);

        // line(im_result,Point(280+99, 0+329),Point(439+99, 100+329),Scalar(0,255,0),2);

    	Mat im_raw = frame(ROI).clone();

        //Start process

        //Convert to HSL colorspace and threshold
        // Mat im_hls, im_th_hls;
        // cvtColor(im_raw, im_hls, COLOR_BGR2HLS);
        // inRange(im_hls, Scalar(0, 155, 0), Scalar(255, 255, 255), im_th_hls); //s_channel

        //Convert to Grayscale and ot'su threshold
        Mat im_gray;
        cvtColor(im_raw, im_gray, COLOR_BGR2GRAY);
        Mat im_blur;
        GaussianBlur(im_gray, im_blur, Size(7, 7), 1, 1 );
        double min, max;
        cv::minMaxLoc(im_blur, &min, &max);
        int th_val = (max-0.25*max);
        Mat im_th_blur;
        threshold(im_blur, im_th_blur, th_val, 255, CV_THRESH_BINARY);
        Mat im_filter_th;
        im_gray.copyTo(im_filter_th, im_th_blur);
        Mat im_otsu = im_filter_th > 0;
        Mat im_th_otsu;
        threshold(im_otsu, im_th_otsu, 0, 255, CV_THRESH_OTSU);

        //Sobel
        Mat im_gradx, im_abs_gradx, im_th_gradx;

        Sobel(im_blur, im_gradx, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs(im_gradx, im_abs_gradx);
        inRange(im_abs_gradx, 20, 100, im_th_gradx);

        Mat im_bw = im_th_otsu & im_th_gradx;

        imshow("Binary image", im_bw);

        // Mat im_edge;
        // Canny(im_bw, im_edge, 150, 300, 5);
        // imshow("Canny", im_edge);

        //HoughLine
        Mat im_line_right = Mat::zeros(im_bw.size(), CV_8UC1);
        vector<Vec4i> lines;
        HoughLinesP(im_bw, lines, 1, CV_PI/180, 20, 20, 200 ); //vote; min line; max gap
        int c_right=0, c_left=0;
        double angle_l=0, angle_r=0;
        Point StL_, EnL_;
        Point StR_, EnR_;
        int MaxL_ =0, MaxR_ = 0;
        int MinL_ =440, MinR_ =440;
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            // raito = float(l[1] / row_size );
            double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
            double d = point_to_line_distance(Point(220,149), l);

            Point En_(l[0], l[1]);
            Point St_(l[2], l[3]);

            int Len;

            //right side
            if (angle>20 && angle<80)
            {
                c_right++;
                angle_r += angle;

                
                // cout << d << endl;
                if ((d < MinR_) && (St_.x > 140))
                {
                    MinR_ = d;
                    StR_ = St_;
                    EnR_ = En_;
                }             
            }

            //left side
            if (angle>-80  && angle<-20) 
            {
                c_left++;
                angle_l += angle;
                // cout << St_ << endl;
                if ((d < MinL_) && (St_.x < 300))
                {
                    MinL_ = d;
                    StL_ = St_;
                    EnL_ = En_;
                } 
            }
        }
        
        if ((c_right > 0) && (MinR_ != 440 ))
        {
            int thelta = angle_r/c_right;
            // cout << "Average angle: " << thelta << endl;
            int Ys = 149;
            int Xs = StR_.x + (Ys - StR_.y) / tan(thelta * CV_PI / 180.0);
            
            int Ye = 0;
            int Xe = EnR_.x + (Ye - EnR_.y) / tan(thelta * CV_PI / 180.0);

            line(im_result, Point(Xs+99, Ys+329), Point(Xe+99, Ye+329), Scalar(0,125,255), 2);
            // line(im_result, Point(StR_.x+99, StR_.y+329), Point(EnR_.x+99, EnR_.y+329), Scalar(0,0,255), 2);
        }

        if ((c_left > 0) && (MinL_ != 440))
        {
            int thelta = angle_l/c_left;
            // cout << "Average angle: " << thelta << endl;
            int Ys = 149;
            int Xs = StL_.x + (Ys - StL_.y) / tan(thelta * CV_PI / 180.0);
            
            int Ye = 0;
            int Xe = EnL_.x + (Ye - EnL_.y) / tan(thelta * CV_PI / 180.0);

            // cout << StL_ << endl;
            // cout << Xe << " " << Ye << endl;
            line(im_result, Point(Xs+99, Ys+329), Point(Xe+99, Ye+329), Scalar(0,125,255), 2);
        }
        
        // Point rook_points[1][4];
        // rook_points[0][0] = Point( w/4.0, 7*w/8.0 );
        // rook_points[0][1] = Point( 3*w/4.0, 7*w/8.0 );
        // rook_points[0][2] = Point( 3*w/4.0, 13*w/16.0 );
        // rook_points[0][3] = Point( 11*w/16.0, 13*w/16.0 );
        // const Point* ppt[1] = { rook_points[0] };
        // int npt[] = { 4 };
        // fillPoly( im_result, ppt, npt, 1, Scalar( 0, 0, 255 ), 8 );

        //Result
        imshow("result", im_result);
        int c = waitKey( (vid.isOpened() && isVideo) ? 1 : 0 ) & 255;
        if ( c == 'q' || c == 'Q' || c == 27)
            break;
    }
    return 0;
}














void BEV(Mat source, Mat& destination)
{
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