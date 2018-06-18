#ifndef MULTILANE_H
#define MUlTILANE_H

#include <vector>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef struct
{
    int key;
    bool Calib_done;
    int num_contour;
    int contourL;
    int contourR;
} Calib_t;

cv::Mat keepLanes(const cv::Mat &org, bool verbose);
cv::Mat twoRightMostLanes(const cv::Size &size, const cv::Mat &imgLane, cv::Point shift = cv::Point(0, 0), bool right = true);
cv::Mat keepLanes_new(const cv::Mat &img);
    
cv::Point twoLaneMostLanes_th(
	const cv::Size &size,
	const cv::Mat &imgLane,
	cv::Mat &ret,
	int yCalc,
	int offset_point,
    int mindl,
	int mindr,
	std::vector<cv::Point> &Lp4corner,
	std::vector<cv::Point> &Rp4corner) ;
    
int calibLanes(
    const cv::Size &size,
    const cv::Mat &imgLane,
    std::vector<cv::Point> &Lp4coner,
    std::vector<cv::Point> &Rp4coner,
    int *NumofContour,
    int key,//doi contour khac- X: trai - D: phai
    bool Restart);
    
int noLane_FindLanes(
	const cv::Size &size,
	const cv::Mat &imgLane,
	std::vector<cv::Point> &Lp4corner,
	std::vector<cv::Point> &Rp4corner);
	
int noLane_FindLanes_new(
	const cv::Size &size,
	const cv::Mat &imgLane,
	std::vector<cv::Point> &Lp4corner,
	std::vector<cv::Point> &Rp4corner,
	int pointMidLane_x);	

cv::Point removeOutlier_fl_new(const cv::Size &size,
							const cv::Mat &org,
							 cv::Mat &lane,
							int pointMidLane_x,
							int yCalc,
							int offset_point,
							int direct,
							int &lanepos,
							double &angle_lane);
							
cv::Point new_lane_process(const cv::Size &size,
							const cv::Mat &org,
							 cv::Mat &lane,
							int pointMidLane_x,
							int yCalc,
							int offset_point,
							int direct,
							int &lanepos,
							double &angle_lane,
							int priorityLane);

cv::Point new_lane_process_yu(const cv::Size &size,
							const cv::Mat &org,
							 cv::Mat &lane,
							int pointMidLane_x,
							int yCalc,
							int offset_point,
							int direct,
							int &lanepos,
							int &angle_lane);

cv::Point hiden_process(const cv::Size &size,
							const cv::Mat &org,
							 cv::Mat &ret,
							int pointMidLane_x,
							int yCalc,
							int offset_point,
							int direct,
							int &lanepos,
							double &angle_lane);
#endif // MULTILANE_H
