#include "multilane.h"

#define VIDEO_FRAME_WIDTH 640	
#define VIDEO_FRAME_HEIGHT 480
//#define OUTDOOR

const int MIN_LANE_AREA = 100	  ; // no . pixels
const int MAX_LANE_AREA = 1000000;
const double INF = 1e9;
const double MIN_ANG_DIFF = 50; // 20 degree
double MIN_X_DIFF = 40; // pixels
double MIN_Y_DIFF = 110; // pixels
double MIN_ACC_DIST = MIN_ANG_DIFF + MIN_X_DIFF * 3 + MIN_Y_DIFF * 5;
/////////////////////////////////////////////////////////////////////////////////////////////////////////
enum StateMachine{
	SM_START = 0,
	SM_STOP,
	SM_RUN,
	SM_PAUSE, 
	SM_NONE,
	CALIB_LEFT,
	CALIB_RIGHT,
	CALIB_NONE,
	CALIB_OK
};
enum CalibKey{
	LEFT = 0,
	RIGHT,
	NONE,
	OK
};
////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Vector {
	double x, y;
	Vector(double x = 0, double y = 0) : x(x), y(y){}
	double operator * (const Vector &that) const {
		return x * that.x + y * that.y;
	}
	double mod() const{
		return sqrt(x * x + y * y);
	}
    // lower angle difference between 2 vectors in degree
	double operator ^ (const Vector &that) const {
		double rad = acos(fabs((*this) * that) / this->mod() / that.mod());
		return rad * 180 / M_PI;
	}
};

struct Edge {
	cv::Point a, b;
	Edge(){}
	Edge(cv::Point a, cv::Point b) : a(a), b(b){}
};

struct Lane { // higher y -> nearer to eyes
	Edge h, l;
	Lane() {}
	Lane(Edge h, Edge l) : h(h), l(l){}
	void show() const {
		fprintf(stderr, "low: (%d, %d) -> (%d, %d)\n", this->l.a.x, this->l.a.y, this->l.b.x, this->l.b.y);
		fprintf(stderr, "hig: (%d, %d) -> (%d, %d)\n", this->h.a.x, this->h.a.y, this->h.b.x, this->h.b.y);
	}
};

bool cmpByY(const Lane &one, const Lane &two) {
	double oneHigMidY = 0.5 * (one.h.a.y + one.h.b.y);
	double twoHigMidY = 0.5 * (two.h.a.y + two.h.b.y);
	return (oneHigMidY > twoHigMidY);
}

double laneDist(const Lane &one, const Lane &two) {
    // assuming that one.l < two.l by cmpByY
	double oneLowMidY = 0.5 * (one.l.a.y + one.l.b.y);
	double twoHigMidY = 0.5 * (two.h.a.y + two.h.b.y);
    if (oneLowMidY < twoHigMidY) // overlap 
    	return 100000.0;
    double xDiff = 0.5 * fabs(two.h.a.x + two.h.b.x - one.l.a.x - one.l.b.x);
    double yDiff = 0.5 * fabs(two.h.a.y + two.h.b.y - one.l.a.y - one.l.b.y);
    if (xDiff > MIN_X_DIFF || yDiff > MIN_Y_DIFF)
    	return 200000.0;
    Vector oneLow(one.l.b.x - one.l.a.x, one.l.b.y - one.l.a.y);
    Vector twoHig(two.h.b.x - two.h.a.x, two.h.b.y - two.h.a.y);
    double angDiff = oneLow ^ twoHig;
    // printf("ang: %lf\n", angDiff);
    if (angDiff > MIN_ANG_DIFF)
    	return 300000.0;
    return angDiff + xDiff * 3 + yDiff * 5;
}

Lane joinTwoLane(const Lane &one, const Lane &two) {
	return Lane(one.h, two.l);
}

void addLanes(std::vector<cv::Point> poly, std::vector<Lane> &lanes) {
	if (poly.size() < 4) return;
	Edge l, h;
	double lsumY = 1e9, hsumY = -1e9;
	for (size_t i = 0; i < poly.size(); ++i) {
		size_t j = (i + 1) % poly.size();
		double sumY = poly[i].y + poly[j].y;
		if (lsumY > sumY) {
			lsumY = sumY;
			l = Edge(poly[i], poly[j]);
		}
		if (hsumY < sumY) {
			hsumY = sumY;
			h = Edge(poly[j], poly[i]);
		}
	}
	lanes.push_back(Lane(h, l));
}

bool joinLanes(std::vector<Lane> &lanes, std::vector< std::vector<cv::Point> > &joins) {
	double bestDist = MIN_ACC_DIST;
	int iOne = -1, iTwo = -1;
	for (int i = 0; i < (int)lanes.size() - 1; ++i)
		for (int j = i + 1; j < (int)lanes.size(); ++j) {
			double curDist = laneDist(lanes[i], lanes[j]);
			if (bestDist > curDist) {
				bestDist = curDist;
				iOne = i;
				iTwo = j;
			}
		}
		if (iOne != -1) {
			std::vector<Lane> tmpLane;
			for (int i = 0; i < (int)lanes.size(); ++i)
				if (i != iOne && i != iTwo)
					tmpLane.push_back(lanes[i]);
				tmpLane.push_back(joinTwoLane(lanes[iOne], lanes[iTwo]));
				std::vector<cv::Point> tmp;
				tmp.push_back(lanes[iOne].l.a);
				tmp.push_back(lanes[iOne].l.b);
				tmp.push_back(lanes[iTwo].h.b);
				tmp.push_back(lanes[iTwo].h.a);
				joins.push_back(tmp);
				lanes = tmpLane;
				return true;
			}
			return false;
		}

		cv::Mat removeOutlier(const cv::Mat &org, std::vector< std::vector< cv::Point> > &polys, bool verbose) {
			cv::Mat img_gray = org.clone();
	    // if (img.channels() == 3)
	    //     cv::cvtColor(img, img, CV_BGR2GRAY);
	    // Binarize image
			cv::Mat img_thr, img_ad;
			cv::threshold(img_gray, img_thr, 160, 255, CV_THRESH_BINARY);


			cv::Mat img = img_thr;

			if (verbose)
				cv::imshow("binary", img);
		    // Erode image -> remove small outliers
			int esize = 1;
			cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
				cv::Size( 2*esize + 1, 2*esize+1 ),
				cv::Point( esize, esize ) );
			cv::erode(img, img, element);
			if (verbose)
				cv::imshow("erosion", img);
		    // cv::dilate(img, img, element);
		    // get contours
			std::vector< std::vector<cv::Point> > contours;
			std::vector< cv::Vec4i > hierarchy;
			cv::findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		    // approximate contours as polygons
			polys.clear();
			for (size_t i = 0; i < contours.size(); ++i) {
				std::vector<cv::Point> p;
				cv::approxPolyDP(cv::Mat(contours[i]), p, 2, true);
				if (p.size() < 3) continue;
				if ( (cv::contourArea(p) < MIN_LANE_AREA) || (cv::contourArea(p) > MAX_LANE_AREA) ) continue;
				polys.push_back(p); 
			}
		    // Draw polygons
			cv::Mat rem = cv::Mat::zeros(img.size(), CV_8UC1);
			for (size_t i = 0; i < polys.size(); ++i)
				cv::drawContours(rem, polys, i, cv::Scalar(255), CV_FILLED);
			if (verbose)
				cv::imshow("remOutlier-result", rem);

		    // cv::imshow("remOutlier-result", rem);
			return rem;
		}

		cv::Mat keepLanes(const cv::Mat &img, bool verbose = false) {
			if (verbose)
				cv::imshow("orgKeepLanes", img);
			std::vector< std::vector< cv::Point> > polys;
			std::vector<Lane> lanes;
			cv::Mat rem = removeOutlier(img, polys, verbose);
			if (verbose)
				cv::imshow("remOutlier", rem);

			for (auto poly : polys)
				addLanes(poly, lanes);

			std::vector< std::vector< cv::Point> > joins;
			while (true) {
				sort(lanes.begin(), lanes.end(), cmpByY);
				if (!joinLanes(lanes, joins)) break;
			}

			cv::Mat draft = cv::Mat::zeros(rem.size(), CV_8UC1);
			for (size_t i = 0; i < polys.size(); ++i)
				cv::drawContours(draft, polys, i, cv::Scalar(255), CV_FILLED);
			for (size_t i = 0; i < joins.size(); ++i)
				cv::drawContours(draft, joins, i, cv::Scalar(255), CV_FILLED);
		    // if (verbose)
		      //  cv::imshow("lanes", draft);
		    // cv::imshow("lanes", draft);
			return draft;
		}

		cv::Mat twoRightMostLanes(const cv::Size &size, const cv::Mat &imgLane, cv::Point shift, bool right) {
	    // keep 2 lanes with highest average x values
	    //cv::imshow("lane", imgLane);
			std::vector< std::vector<cv::Point> > contours;
			std::vector< cv::Vec4i > hierarchy;
			cv::findContours(imgLane, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
			if (contours.size() <= 1) {
				cv::Mat none = cv::Mat::zeros(size, CV_8UC1);
				return none;
			}
    int sign = 1; //left
    if (right) sign = -1; // right
    fprintf(stderr, "sign = %d\n", sign);
    double x1Hig = 1e4 * sign, x2Hig = 1e4 * sign;
    int id1 = -1, id2 = -1;
    for (int i = 0; i < (int)contours.size(); ++i) {
    	if (cv::contourArea(contours[i]) < 20) continue;
    	double xMaxY = -1e9, maxY = -1e9;
    	for (auto p : contours[i])
    		if (maxY < p.y) {
    			maxY = p.y;
    			xMaxY = p.x;
    		}
	        // bam phai: xMaxY > x1Hig ... else xMaxY > x2Hig
	        // bam trai: xMaxY < x1Hig ... else xMaxY < x2Hig
	        // Khong sua cac dau <, > khac
    		if (xMaxY * sign < x1Hig * sign) {
    			x2Hig = x1Hig;
    			id2 = id1;
    			x1Hig = xMaxY;
    			id1 = i;
    		}
    		else
    			if (xMaxY * sign < x2Hig * sign) {
    				x2Hig = xMaxY;
    				id2 = i;
    			}
    		}
    		fprintf(stderr, "%lf %lf\n", x2Hig, x1Hig);

    		if (id1 == -1 || id2 == -1) {
    			cv::Mat none = cv::Mat::zeros(size, CV_8UC1);
    			return none;
    		}

    		cv::Mat ret = cv::Mat::zeros(size, CV_8UC1);
    		std::vector< std::vector< cv::Point> > tmp;
    		tmp.push_back(contours[id1]);
    		tmp.push_back(contours[id2]);
		    // printf("%u %u\n", tmp[0].size(), tmp[1].size());
		    // for (auto &c : tmp)
		    //     for (auto &p : c)
		    //         p.x += shift.x, p.y += shift.y;
    		for (size_t i = 0; i < tmp.size(); ++i)
    			cv::drawContours(ret, tmp, i, cv::Scalar(255), CV_FILLED);
    		return ret;
    	}

    	double getTheta_(cv::Point car, cv::Point dst) {
    		if (dst.x == car.x) return 0;
    		if (dst.y == car.y) return (dst.x < car.x ? -90 : 90);
    		double pi = acos(-1.0);
    		double dx = dst.x - car.x;
    double dy = car.y - dst.y; // image coordinates system: car.y > dst.y
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}


cv::Mat removeOutlier_new(const cv::Mat &org, std::vector< std::vector< cv::Point> > &polys) {
	cv::Mat img_gray = org.clone();

    //Adaptive Threshold and median filter
	cv::Mat img_erode, img_ad, img_md;
	cv::adaptiveThreshold(img_gray, img_ad, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 55, 1.0);
	cv::medianBlur(img_ad, img_md, 5);

    //Erode image -> remove small outliers
	int esize = 2;
	cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
		cv::Size( 2*esize + 1, 2*esize+1 ),
		cv::Point( esize, esize ) );
	cv::erode(img_md, img_erode, element);

    #ifdef OUTDOOR
	cv::Mat img = img_erode;
    #endif
    #ifndef OUTDOOR
	cv::Mat img_thr;
	double gmin, gmax, gthr;
	cv::minMaxLoc(img_gray, &gmin, &gmax);
	if (gmax < 150)
		gthr = gmax - (gmax-gmin)*0.5;
	else gthr = gthr = gmax - (gmax-gmin)*0.4;
	cv::threshold(img_gray, img_thr, gthr, 255, CV_THRESH_BINARY);
	cv::Mat img = img_erode & img_thr;
    #endif // OUTDOOR

    // cv::dilate(img, img, element);
    // get contours
    // cv::imshow("bw", img);
	std::vector< std::vector<cv::Point> > contours;
	std::vector< cv::Vec4i > hierarchy;
	cv::findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    // approximate contours as polygons
	polys.clear();
	for (size_t i = 0; i < contours.size(); ++i) {
		std::vector<cv::Point> p;
		cv::approxPolyDP(cv::Mat(contours[i]), p, 2, true);

		cv::Rect boundRect = boundingRect( cv::Mat(p) );

		if (p.size() < 3) continue;
        #ifdef OUTDOOR
		if ( ( (cv::contourArea(p) < MIN_LANE_AREA) || (cv::contourArea(p) > MAX_LANE_AREA) ) || (boundRect.height < 40) )
			continue; 
        #endif
        #ifndef OUTDOOR
		if ( (cv::contourArea(p) < MIN_LANE_AREA) || (cv::contourArea(p) > MAX_LANE_AREA) )
			continue;
        #endif
        //std::cout << boundRect.height << std::endl;
		polys.push_back(p); 
	}
    // Draw polygons
	cv::Mat rem = cv::Mat::zeros(img.size(), CV_8UC1);
	for (size_t i = 0; i < polys.size(); ++i)
		cv::drawContours(rem, polys, i, cv::Scalar(255), CV_FILLED);

	//cv::imshow("remOutlier-result", rem);
	return rem;
}

cv::Mat keepLanes_new(const cv::Mat &img) {

	std::vector< std::vector< cv::Point> > polys;
	std::vector<Lane> lanes;
	cv::Mat rem = removeOutlier_new(img, polys);

	for (auto poly : polys)
		addLanes(poly, lanes);

	std::vector< std::vector< cv::Point> > joins;
	while (true) {
		sort(lanes.begin(), lanes.end(), cmpByY);
		if (!joinLanes(lanes, joins)) break;
	}

	cv::Mat draft = cv::Mat::zeros(rem.size(), CV_8UC1);
	for (size_t i = 0; i < polys.size(); ++i)
		cv::drawContours(draft, polys, i, cv::Scalar(255), CV_FILLED);
	for (size_t i = 0; i < joins.size(); ++i)
		cv::drawContours(draft, joins, i, cv::Scalar(255), CV_FILLED);

	return draft;
}

std::vector<cv::Point> Export4corners(const cv::Size &size, std::vector< std::vector<cv::Point> > contours, int IoC) {
	std::vector<cv::Point> Res;
	Res.clear();
	cv::Point   pTL = cv::Point(-1,-1), pTR = cv::Point(-1,-1),
	pBL = cv::Point(-1,-1), pBR = cv::Point(-1,-1),
	pMaxL = cv::Point(1e4,-1), pMaxR = cv::Point(-1,-1),
	pML2 = cv::Point(1e4,-1), pMR2 = cv::Point(-1,-1);

	cv::Mat imgTmp = cv::Mat::zeros(size, CV_8UC1);
	cv::drawContours(imgTmp, contours, IoC, cv::Scalar(255), CV_FILLED);

	for (int y=0; y<imgTmp.rows; y++)
	{
		for (int x=0; x<imgTmp.cols; x++)
		{
			if (imgTmp.at<uchar>(y, x) == 255)
			{
				pTL = cv::Point(x,y);
				break;
			}
		}
		if (pTL.y != -1)
			break;
	}
	Res.push_back(pTL);

	for (int y=0; y<imgTmp.rows; y++)
	{
		for (int x=imgTmp.cols-1; x >= 0; x--)
		{
			if (imgTmp.at<uchar>(y, x) == 255)
			{
				pTR = cv::Point(x,y);
				break;
			}
		}
		if (pTR.y != -1)
			break;
	}
	Res.push_back(pTR);

	for (int y=imgTmp.rows-1; y>=0; y--)
	{
		for (int x=0; x<imgTmp.cols; x++)
		{
			if (imgTmp.at<uchar>(y, x) == 255)
			{
				pBL = cv::Point(x,y);
				break;
			}
		}
		if (pBL.y != -1)
			break;
	}
	Res.push_back(pBL);

	for (int y=imgTmp.rows-1; y>=0; y--)
	{
		for (int x=imgTmp.cols-1; x >= 0; x--)
		{
			if (imgTmp.at<uchar>(y, x) == 255)
			{
				pBR = cv::Point(x,y);
				break;
			}
		}
		if (pBR.y != -1)
			break;
	}
	Res.push_back(pBR);

	return Res;
}

cv::Point twoLaneMostLanes_th(
	const cv::Size &size,
	const cv::Mat &imgLane,
	cv::Mat &ret,
	int yCalc,
	int offset_point,
	int mindl,
	int mindr,
	std::vector<cv::Point> &Lp4corner,
	std::vector<cv::Point> &Rp4corner) 
{
	ret = cv::Mat::zeros(size, CV_8UC1);

	std::vector< std::vector<cv::Point> > contours;
	std::vector< cv::Vec4i > hierarchy;
	cv::findContours(imgLane, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	double x1Hig = 1e4, x2Hig = - 1e4, y1Hig = -1, y2Hig = -1;
	int id1 = -1, id2 = -1;
    
    int xminl = mindl;
    int xminr = mindr;
    
    double dist2;
    double dist3;
    
    std::vector<cv::Point> preLp4corner = Lp4corner;
    std::vector<cv::Point> preRp4corner = Rp4corner;

    for (int i = 0; i < (int)contours.size(); ++i) 
	{
		if(Lp4corner.empty() || Rp4corner.empty())
			return cv::Point(0,0);
		if (cv::contourArea(contours[i]) < MIN_LANE_AREA) 
			continue;

		std::vector<cv::Point> new_ = Export4corners(size, contours, i);
		
		dist3 = cv::norm(new_[3] - preLp4corner[3]);
		dist2 = cv::norm(new_[2] - preLp4corner[2]);
    	//follow left
		if ((dist3 < mindl) &&
		    (dist2 < xminl) &&
			(std::abs(new_[3].y - preLp4corner[3].y) < 20))
		{
			mindl = dist3;
			id1 = i;
			Lp4corner = new_;
		}
		
    	//follow right
    	dist3 = cv::norm(new_[3] - preRp4corner[3]);
		dist2 = cv::norm(new_[2] - preRp4corner[2]);
		if ((dist2 < mindr) &&
		    (dist3 < xminr) &&
			(std::abs(new_[2].y - preRp4corner[2].y) < 20))
		{
			mindr = dist2;
			id2 = i;
			Rp4corner = new_;   
		}
    }
	/*
	for (int i = 0; i < (int)contours.size(); ++i) 
	{
		if(Lp4corner.empty() | Lp4corner.empty())
			return cv::Point(0,0);

		if (cv::contourArea(contours[i]) < MIN_LANE_AREA) continue;

		std::vector<cv::Point> new_ = Export4corners(size, contours, i);
		//std::cout<< "new: " <<new_ << std::endl << std::endl;
        //So sanh' do^. doi` cua? line moi' so vs line cu? (Bot-Right, Top Right) doi^' vs line trai'
		if (id1 == -1)
		{
			if ((cv::norm(new_[3] - Lp4corner[3]) < 30) && (cv::norm(new_[2] - Lp4corner[2]) < 30))
			{
				id1 = i;
				Lp4corner = new_;
			}
		}

		if ((id2 == -1) && (i != id1))
		{
			if ((cv::norm(new_[2] - Rp4corner[2]) < 30) && (cv::norm(new_[3] - Rp4corner[3]) < 30))
			{
				id2 = i;
				Rp4corner = new_;
			}
		}
	}*/

	cv::Point errPoint = cv::Point(0,0);

	if (id1 == -1 && id2 == -1)
        return errPoint; //no line
    //debug
    if(id1 != -1)
    	cv::drawContours(ret, contours, id1, cv::Scalar(255), CV_FILLED);
    if(id2 != -1)
    	cv::drawContours(ret, contours, id2, cv::Scalar(255), CV_FILLED);
    //

    int sumY1 = 0, sumY2 = 0;
    int sumPoint1 = 0, sumPoint2 = 0;

    if (id1 != -1 && id2 != -1) 
    {
    	if (id1 != -1)
    	{
    		cv::Mat imgTmp1 = cv::Mat::zeros(size, CV_8UC1);
    		cv::drawContours(imgTmp1, contours, id1, cv::Scalar(255), CV_FILLED);
    		for (int x = imgTmp1.cols - 1; x >= 0; x--)
    		{
    			if (imgTmp1.at<uchar>(yCalc, x) == 255 )
    			{
    				sumY1 += x;
    				sumPoint1++;
    			}
    		}
    	}
    	if (id2 != -1)
    	{
    		cv::Mat imgTmp2 = cv::Mat::zeros(size, CV_8UC1);
    		cv::drawContours(imgTmp2, contours, id2, cv::Scalar(255), CV_FILLED);
    		for (int x = imgTmp2.cols - 1; x >= 0; x--)
    		{
    			if (imgTmp2.at<uchar>(yCalc, x) == 255 )
    			{
    				sumY2 += x;
    				sumPoint2++;
    			}
    		}
    	}
    	if (sumPoint1*sumPoint2 != 0)
    	{
    		if (((sumY1/sumPoint1 + sumY2/sumPoint2)/2 - (sumY1/sumPoint1)) < 100)
    			return errPoint;
    		else return cv::Point((sumY1/sumPoint1 + sumY2/sumPoint2)/2, yCalc);
    	}
    	else if (sumPoint1 != 0)
    	{
    	    if ((Lp4corner[1].x - Lp4corner[3].x) > 200)
    		    return cv::Point(sumY1/sumPoint1 + 1.5*offset_point, yCalc);
    		return cv::Point(sumY1/sumPoint1 + offset_point, yCalc);
    	}
    	else if (sumPoint2 != 0)
    	{
    	    if ((Rp4corner[2].x - Rp4corner[0].x) > 200)
    		    return cv::Point(sumY2/sumPoint2 - 1.7*offset_point , yCalc);
    		return cv::Point(sumY2/sumPoint2 - offset_point, yCalc);
    	}
    	return errPoint;
    }

    else if (id1 != -1)
    {
    	cv::Mat imgTmp1 = cv::Mat::zeros(size, CV_8UC1);
    	cv::drawContours(imgTmp1, contours, id1, cv::Scalar(255), CV_FILLED);
    	for (int x = imgTmp1.cols - 1; x >= 0; x--)
    	{
    		if (imgTmp1.at<uchar>(yCalc, x) == 255 )
    		{
    			sumY1 += x;
    			sumPoint1++;
    		}
    	}

    	if (sumPoint1 != 0)
    	{
    		if ((Lp4corner[1].x - Lp4corner[3].x) > 200)
    		    return cv::Point(sumY1/sumPoint1 + 1.5*offset_point, yCalc);
    		return cv::Point(sumY1/sumPoint1 + offset_point, yCalc);
    	}
    	return errPoint;
    }

    else if (id2 != -1)
    {
    	cv::Mat imgTmp2 = cv::Mat::zeros(size, CV_8UC1);
    	cv::drawContours(imgTmp2, contours, id2, cv::Scalar(255), CV_FILLED);
    	for (int x = imgTmp2.cols - 1; x >= 0; x--)
    	{
    		if (imgTmp2.at<uchar>(yCalc, x) == 255 )
    		{
    			sumY2 += x;
    			sumPoint2++;
    		}
    	} 
    	if (sumPoint2 != 0)
    	{
    	    if ((Rp4corner[2].x - Rp4corner[0].x) > 200)
    		    return cv::Point(sumY2/sumPoint2 - 1.7*offset_point , yCalc);
    		return cv::Point(sumY2/sumPoint2 - offset_point, yCalc);
    	}
    }
    return errPoint;
}

int calibLanes(
	const cv::Size &size,
	const cv::Mat &imgLane,
	std::vector<cv::Point> &Lp4corner,
	std::vector<cv::Point> &Rp4corner,
	int *NumofContour,
    int key,//doi contour khac- X: trai - D: phai
    bool Restart) {

	std::vector< std::vector<cv::Point> > contours;
	std::vector< cv::Vec4i > hierarchy;
	cv::findContours(imgLane, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	static std::vector<cv::Point>   Lp4corner_, Rp4corner_;
	static int LenghTop = 0, LenghBot = 0;
    //static std::vector<cv::Point>   Lp4corner_reject, Rp4corner_reject;

	double x1Hig = 1e4, x2Hig = - 1e4, y1Hig = -1, y2Hig = -1;
	int id1 = -1, id2 = -1;

	(*NumofContour) = contours.size();

	for (int i = 0; i < (int)contours.size(); ++i) 
	{
		if (cv::contourArea(contours[i]) < MIN_LANE_AREA) 
			continue;

		double xMaxY = -1e9, maxY = -1e9;
		double xMinY = 1e9, minY = 1e9;
        //double MaxX = -1e9, ymaxX = -1e9;
        //double MinX = 1e9, yminX = 1e9;

        //tim toa do y lon nhat tai tung contour size.width
		for (auto p : contours[i])
		{
			if (maxY < p.y)
			{
				maxY = p.y;
				xMaxY = p.x;
			}
			if (minY > p.y)
			{
				minY = p.y;
				xMinY = p.x;
			}
            /*if (maxX < p.x)
            {
                maxX = p.x;
                ymaxX = p.y;
            }
            if (MinX > p.x)
            {
                MinX = p.x;
                yminX = p.y;
            }*/
		}

    //follow left
		if ((xMaxY < x1Hig) &&  
			(xMaxY < size.width*0.5) &&
			(maxY > size.height - 100))
		{
			y1Hig = maxY;
			x1Hig = xMaxY;
			id1 = i;
		}
    //follow right
		if ((xMaxY > x2Hig) &&
			(xMaxY > size.width*0.5) &&
			(maxY > size.height - 100))
		{
			y2Hig =maxY;
			x2Hig = xMaxY;
			id2 = i;     
		}
	}

	if(key == 1)
	{
		Lp4corner_.clear();
		id1 = -1;
	}
	else if(key == 2)
	{
		Rp4corner_.clear();
		id2 = -1;
	}

    //tl tr bl br
	if(id1 != -1)
	{
		Lp4corner = Export4corners(size, contours, id1);
		if(Lp4corner_.empty())
		{
			Lp4corner_ = Lp4corner;
		}
		else
		{
			if((Lp4corner[1].x < Lp4corner_[1].x + 20) && (Lp4corner[1].x > Lp4corner_[1].x - 20) &&
				(Lp4corner[3].x < Lp4corner_[3].x + 40) && (Lp4corner[3].x > Lp4corner_[3].x - 40))
			{
				Lp4corner_ = Lp4corner;
			}
			else
			{
				id1 = -1;
				if(Restart && (id2 == -1))
				{
					Lp4corner_ = Lp4corner;
					Rp4corner = Lp4corner;
					Rp4corner[0].x += LenghTop;
					Rp4corner[1].x += LenghTop;
					Rp4corner[2].x += LenghBot;
					Rp4corner[3].x += LenghBot;
					Rp4corner_ = Rp4corner;
					id1 = 0;
				}
				else
					Lp4corner = Lp4corner_;
			}
		}
	}

	if(id2 != -1)
	{
		Rp4corner = Export4corners(size, contours, id2);
		if(Rp4corner_.empty())
		{
			Rp4corner_ = Rp4corner;
		}
		else
		{
			if((Rp4corner[1].x < Rp4corner_[1].x + 20) && (Rp4corner[1].x > Rp4corner_[1].x - 20) &&
				(Rp4corner[3].x < Rp4corner_[3].x + 40) && (Rp4corner[3].x > Rp4corner_[3].x - 40))
			{
				Rp4corner_ = Rp4corner;
			}
			else
			{
				id2 = -1;
				if(Restart && (id1 == -1))
				{
					Rp4corner_ = Rp4corner;
					Lp4corner = Rp4corner;
					Lp4corner[0].x -= LenghTop;
					Lp4corner[1].x -= LenghTop;
					Lp4corner[2].x -= LenghBot;
					Lp4corner[3].x -= LenghBot;
					Lp4corner_ = Lp4corner;
					id2 = 0;
				}
				else
					Rp4corner = Rp4corner_;
			}
		}
	}

	if((id1 != -1) && (id2 != -1))
	{
		LenghTop = Rp4corner_[1].x - Lp4corner_[1].x;
		LenghBot = Rp4corner_[3].x - Lp4corner_[3].x;
	}

	if((id1 != -1) && (id2 != -1))
		return 0;
	else if (id1 != -1)
	{
		Rp4corner = Rp4corner_;
		return 1;
	}
	else if (id2 != -1)
	{
		Lp4corner = Lp4corner_;
		return 2;
	}
	else if ((id1 == -1) && (id2 == -1) && (!Rp4corner_.empty()) && (!Lp4corner_.empty()))
	{
		Rp4corner = Rp4corner_;
		Lp4corner = Lp4corner_;
		return 3;
	}
	return -1;
}


int noLane_FindLanes_new(
	const cv::Size &size,
	const cv::Mat &imgLane,
	std::vector<cv::Point> &Lp4corner,
	std::vector<cv::Point> &Rp4corner,
	int pointMidLane_x)
{
	std::vector< std::vector<cv::Point> > contours;
	std::vector< cv::Vec4i > hierarchy;
	cv::findContours(imgLane, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    std::vector<cv::Point>   new_4corner;
	std::vector<cv::Point>   Lp4corner_ = Lp4corner;
	std::vector<cv::Point>   Rp4corner_ = Rp4corner;
	

	double x1Hig = 1e4, x2Hig = - 1e4, y1Hig = -1, y2Hig = -1;
	int id1 = -1, id2 = -1;

	for (int i = 0; i < (int)contours.size(); ++i) 
	{
		if (cv::contourArea(contours[i]) < MIN_LANE_AREA) 
			continue;
        std::vector<cv::Point> new_4corner_ = Export4corners(size, contours, i);
        if ((new_4corner_[0].y < 10) && (new_4corner_[2].y > 105)) 
        {
            new_4corner = new_4corner_;
            //follow left
		    if (new_4corner[0].x < x1Hig)
		    {
			    y1Hig = new_4corner[0].y;
			    x1Hig = new_4corner[0].x;
			    id1 = i;
		    }
            //follow right
		    if (new_4corner[0].x > x2Hig)
		    {
			    y2Hig = new_4corner[0].y;
			    x2Hig = new_4corner[0].x;
			    id2 = i;     
		    } 
        }
    }
    //tl tr bl br
    if ((id1 != -1) && (id2 != -1) && (id1 != id2))
    {
        Lp4corner = Export4corners(size, contours, id1);
        Rp4corner = Export4corners(size, contours, id2);
        return 0;
    }
    
    if ((id1 != -1) && (id2 != -1) && (id1 == id2))
    {
        if (((new_4corner[2].x - pointMidLane_x) < 0) && ((new_4corner[3].x - pointMidLane_x) < 0))
        {
            Lp4corner = Export4corners(size, contours, id1);
            if (std::abs(Lp4corner[3].x - Rp4corner_[2].x) < 350)
            {
                Rp4corner[0] = cv::Point(640,0);
                Rp4corner[1] = cv::Point(640,0);
                Rp4corner[2] = cv::Point(640,120);
                Rp4corner[3] = cv::Point(640,120);
            } 
            else Rp4corner = Rp4corner_;
            return 1;
        }
        else if (((new_4corner[2].x - pointMidLane_x) > 0) && ((new_4corner[3].x - pointMidLane_x) > 0))
        {
            Rp4corner = Export4corners(size, contours, id1);
            if (std::abs(Rp4corner[2].x - Lp4corner_[3].x) < 350)
            {
                Lp4corner[0] = cv::Point(0,0);
                Lp4corner[1] = cv::Point(0,0);
                Lp4corner[2] = cv::Point(0,120);
                Lp4corner[3] = cv::Point(0,120);
            } 
            else Lp4corner = Lp4corner_;
            return 2;
        }
        else if (((new_4corner[2].x - pointMidLane_x) < 0) && ((new_4corner[3].x - pointMidLane_x) > 0))
        {
            if (std::abs(new_4corner[2].x - pointMidLane_x) 
                > std::abs(new_4corner[3].x - pointMidLane_x))
            {
                Lp4corner = Export4corners(size, contours, id1);
                return 1;
            }
            else 
            {
                Rp4corner = Export4corners(size, contours, id1);
                return 2;
            }
        }
    } 
}

cv::Point removeOutlier_fl_new(const cv::Size &size,
							const cv::Mat &org,
							 cv::Mat &lane,
							int pointMidLane_x,
							int yCalc,
							int offset_point,
							int direct,
							int &lanepos,
							double &angle_lane)
{
	cv::Mat img_gray = org.clone();
	angle_lane = 0;
	double lane_ratio = 3;
	//Adaptive Threshold and median filter
	cv::Mat img_erode, img_ad, img_md;
	cv::adaptiveThreshold(img_gray, img_ad, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 55, 1.0);
	cv::medianBlur(img_ad, img_md, 5);

	// Erode image -> remove small outliers
	int esize = 2;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
												cv::Size(2 * esize + 1, 2 * esize + 1),
												cv::Point(esize, esize));
	cv::erode(img_md, img_erode, element);

#ifdef OUTDOOR
	cv::Mat img = img_erode;
#endif
#ifndef OUTDOOR
	cv::Mat img_thr;
	double gmin, gmax, gthr;
	cv::minMaxLoc(img_gray, &gmin, &gmax);
	if (gmax < 150)
		gthr = gmax - (gmax - gmin) * 0.5;
	else
		gthr = gthr = gmax - (gmax - gmin) * 0.4;
	cv::threshold(img_gray, img_thr, gthr, 255, CV_THRESH_BINARY);
	cv::Mat img = img_erode & img_thr;
#endif // OUTDOOR

	// cv::dilate(img, img, element);
	// get contours
	//cv::imshow("bw", img);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	contours.clear();
	hierarchy.clear();
	cv::findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	if (contours.size() < 1)
		std::cout << "NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO" << std::endl;
	std::cout << "YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS" << std::endl;
	lanepos = 0;

	double x1Hig = 1e4, x2Hig = -1e4, y1Hig = -1, y2Hig = -1;
	int id1 = -1, id2 = -1;

	for (int i = 0; i < (int)contours.size(); ++i)
	{
		std::vector<cv::Point> p;
		cv::approxPolyDP(cv::Mat(contours[i]), p, 2, true);

		if (p.size() < 3)
			continue;

		if ((cv::contourArea(p) < MIN_LANE_AREA) || (cv::contourArea(p) > MAX_LANE_AREA))
			continue;

		std::vector<cv::Point> new_4corner = Export4corners(size, contours, i);
		if ((new_4corner[0].y < 60) && (new_4corner[2].y > 105))
		{
			//follow left
			if (new_4corner[2].x < x1Hig)
			{
				y1Hig = new_4corner[2].y;
				x1Hig = new_4corner[2].x;
				id1 = i;
			}
			//follow right
			if (new_4corner[2].x > x2Hig)
			{
				y2Hig = new_4corner[2].y;
				x2Hig = new_4corner[2].x;
				id2 = i;
			}
		}
	}
	//std::cout << "id1 " << id1 << "id2 " << id2 << std::endl;
	//tl tr bl br
	cv::Mat ret = cv::Mat::zeros(size, CV_8UC1);
	if(id1 != -1)
	    cv::drawContours(ret, contours, id1, cv::Scalar(255), CV_FILLED);
    if(id2 != -1)
    	cv::drawContours(ret, contours, id2, cv::Scalar(125), CV_FILLED);
	//imshow("lane", ret);

	if ((id1 != -1) && (id2 != -1) && (id1 != id2))
	{
		// if ((direct == 1) || (direct == 0))
		// 	lanepos = 1;
		// else if (direct == 2)
		// 	lanepos = 2;
		lanepos = 1;
	}

	if ((id1 != -1) && (id2 != -1) && (id1 == id2))
	{
		std::vector<cv::Point> ln_corner = Export4corners(size, contours, id2);
		if (((ln_corner[2].x - pointMidLane_x) < 0) && ((ln_corner[3].x - pointMidLane_x) < 0))
		{
			lanepos = 1;
			angle_lane = getTheta_(ln_corner[3],ln_corner[1]);
		}
		else if (((ln_corner[2].x - pointMidLane_x) > 0) && ((ln_corner[3].x - pointMidLane_x) > 0))
		{
			lanepos = 2;
			angle_lane = getTheta_(ln_corner[2],ln_corner[0]);
		}
		else if (((ln_corner[2].x - pointMidLane_x) < 0) && ((ln_corner[3].x - pointMidLane_x) > 0))
		{
			if (std::abs(ln_corner[2].x - pointMidLane_x) > std::abs(ln_corner[3].x - pointMidLane_x))
			{
				lanepos = 1;
				angle_lane = getTheta_(ln_corner[3],ln_corner[1]);
			}
			else
			{
				lanepos = 2;
				angle_lane = getTheta_(ln_corner[2],ln_corner[0]);
			}
		}
	}
	cv::Mat imgTmp = cv::Mat::zeros(size, CV_8UC1);
	if (lanepos > 0)
	{
		int id;
		if (lanepos == 1)
			id = id1;
		else if (lanepos == 2)
			id = id2;
		cv::drawContours(imgTmp, contours, id, cv::Scalar(255), 1);
		//imshow("Lane", imgTmp);
		lane = imgTmp.clone();
		std::cout << "Lanepos: " << lanepos << std::endl;
		int sumY=0, sumPoint=0;
		for (int x = imgTmp.cols - 1; x >= 0; x--)
		{
			if (imgTmp.at<uchar>(yCalc, x) == 255)
			{
				sumY += x;
				sumPoint++;
			}
		}
		if (lanepos == 1)
		{
			if (angle_lane > 55.0)
				return cv::Point(sumY / sumPoint + offset_point + 60, yCalc);
			else
				return cv::Point(sumY / sumPoint + offset_point, yCalc);
		}
		else if (lanepos == 2)
		{
			if (angle_lane < -55.0)
				return cv::Point(sumY / sumPoint - offset_point - 60, yCalc);
			else
				return cv::Point(sumY / sumPoint - offset_point, yCalc);
		}
	}
	lane = imgTmp.clone();
	return cv::Point(-1, -1);
}

cv::Point new_lane_process(const cv::Size &size,
							const cv::Mat &org,
							 cv::Mat &ret,
							int pointMidLane_x,
							int yCalc,
							int offset_point,
							int direct,
							int &lanepos,
							double &angle_lane,
							int priorityLane)
{
	// double st1 = 0, et1 = 0, freq = cv::getTickFrequency();
	// st1 = cv::getTickCount();
	// cv::Mat img_gray = org.clone();
	//cv::imshow("img",img_gray);
	angle_lane = 0;
	double lane_ratio = 3;
		//Adaptive Threshold and median filter
	cv::Mat img_erode, img_ad, img_md;
	//cv::adaptiveThreshold(img_gray, img_ad, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 55, 1.0);
	// cv::medianBlur(img_ad, img_md, 5);

	// // Erode image -> remove small outliers
	// int esize = 2;
	// cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
	// 											cv::Size(2 * esize + 1, 2 * esize + 1),
	// 											cv::Point(esize, esize));
	// cv::erode(img_md, img_erode, element);

// #ifdef OUTDOOR
// 	cv::Mat img = img_erode;
// #endif
// #ifndef OUTDOOR

	cv::Mat img_gray;
	cv::cvtColor(org, img_gray, CV_BGR2GRAY);
	cv::Mat img_thr;
	double gmin, gmax, gthr;
	cv::minMaxLoc(img_gray, &gmin, &gmax);
	if (gmax < 150)
		gthr = gmax - (gmax - gmin) * 0.5;
	else
		gthr = gmax - (gmax - gmin) * 0.6;
	cv::threshold(img_gray, img_thr, 80, 255, CV_THRESH_BINARY);

	cv::Mat img_HSV, img_green;
	cv::cvtColor(org, img_HSV, CV_BGR2HSV);
	inRange(img_HSV, cv::Scalar(65, 100, 20), cv::Scalar(80, 255, 255), img_green);

	cv::Mat img = img_thr + img_green;

	int esize = 2;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
											cv::Size(2 * esize + 1, 2 * esize + 1),
											cv::Point(esize, esize));
	cv::dilate(img, img, element);
	cv::erode(img, img, element);

//#endif // OUTDOOR

	//cv::dilate(img, img, element);
	//get contours
	//cv::imshow("bw", img);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	contours.clear();
	hierarchy.clear();
	cv::findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	// if (contours.size() < 1)
	// std::cout << "YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS" << std::endl;
	// et1 = cv::getTickCount();
	// std::cout << "Find Lane 0\tTime\t" << (et1-st1)/freq << 	std::endl;
	double x1Hig = 1e4, x2Hig = -1e4, y1Hig = -1, y2Hig = -1;
	int id1 = -1, id2 = -1;
	lanepos = 0;

	for (int i = 0; i < (int)contours.size(); ++i)
	{
		std::vector<cv::Point> p;
		cv::approxPolyDP(cv::Mat(contours[i]), p, 2, true);

		if (p.size() < 3)
			continue;

		if ((cv::contourArea(p) < MIN_LANE_AREA) || (cv::contourArea(p) > MAX_LANE_AREA))
			continue;

		std::vector<cv::Point> new_4corner = Export4corners(size, contours, i);
		// if ((new_4corner[0].y < 100) && (new_4corner[2].y > 55))
		// {
			//follow left
			if (new_4corner[2].x < x1Hig)
			{
				y1Hig = new_4corner[2].y;
				x1Hig = new_4corner[2].x;
				id1 = i;
			}
			//follow right
			if (new_4corner[2].x > x2Hig)
			{
				y2Hig = new_4corner[2].y;
				x2Hig = new_4corner[2].x;
				id2 = i;
			}
		// }
	}
	// et1 = cv::getTickCount();
	// std::cout << "Find Lane 1\tTime\t" << (et1-st1)/freq << 	std::endl;
	//std::cout << "id1 " << id1 << "	id2 " << id2 << std::endl;
	//tl tr bl br
	ret = cv::Mat::zeros(size, CV_8UC1);
	if(id1 != -1)
	    cv::drawContours(ret, contours, id1, cv::Scalar(150), CV_FILLED);
    if(id2 != -1)
    	cv::drawContours(ret, contours, id2, cv::Scalar(100), CV_FILLED);
	//imshow("lane", ret);
	// et1 = cv::getTickCount();
	// std::cout << "Find Lane 2\tTime\t" << (et1-st1)/freq << 	std::endl;
	if ((id1 != -1) && (id2 != -1) && (id1 != id2))
	{
		if ((direct == 2))
			lanepos = 2;
		else if ((direct == 1) || (direct == 0))
			lanepos = 1;
		//lanepos = 1;
	}
	else if ((id1 != -1) && (id2 != -1) && (id1 == id2))
	{
		std::vector<cv::Point> ln_corner = Export4corners(size, contours, id2);
		if (((ln_corner[2].x - pointMidLane_x) < 0) && ((ln_corner[3].x - pointMidLane_x) < 0))
		{
			lanepos = 1;
			//angle_lane = getTheta_(ln_corner[3],ln_corner[1]);
		}
		else if (((ln_corner[2].x - pointMidLane_x) > 0) && ((ln_corner[3].x - pointMidLane_x) > 0))
		{
			lanepos = 2;
			//angle_lane = getTheta_(ln_corner[2],ln_corner[0]);
		}
		else if (((ln_corner[2].x - pointMidLane_x) < 0) && ((ln_corner[3].x - pointMidLane_x) > 0))
		{
			if (std::abs(ln_corner[2].x - pointMidLane_x) > std::abs(ln_corner[3].x - pointMidLane_x))
			{
				lanepos = 1;
				//angle_lane = getTheta_(ln_corner[3],ln_corner[1]);
			}
			else
			{
				lanepos = 2;
				//angle_lane = getTheta_(ln_corner[2],ln_corner[0]);
			}
		}
	}
	// et1 = cv::getTickCount();
	// std::cout << "Find Lane\tTime\t" << (et1-st1)/freq << 	std::endl;
	cv::Mat imgTmp = cv::Mat::zeros(size, CV_8UC1);
	if (lanepos > 0)
	{
		int id;
		if(priorityLane > 0)
		{
			if (lanepos == 2)
			{
				id = id2;
				std::vector<cv::Point> ln_corner = Export4corners(size, contours, id);
				angle_lane = getTheta_(ln_corner[2],ln_corner[0]);
			}
			else if (lanepos == 1)
			{
				id = id1;
				std::vector<cv::Point> ln_corner = Export4corners(size, contours, id);
				angle_lane = getTheta_(ln_corner[3],ln_corner[1]);
			}
		}
		else
		{
			if (lanepos == 1)
			{
				id = id1;
				std::vector<cv::Point> ln_corner = Export4corners(size, contours, id);
				angle_lane = getTheta_(ln_corner[3],ln_corner[1]);
			}
			else if (lanepos == 2)
			{
				id = id2;
				std::vector<cv::Point> ln_corner = Export4corners(size, contours, id);
				angle_lane = getTheta_(ln_corner[2],ln_corner[0]);
			}
		}
		cv::drawContours(imgTmp, contours, id, cv::Scalar(255), 1);
		//imshow("Lane", imgTmp);
		// lane = imgTmp.clone();
		cv::drawContours(ret, contours, id, cv::Scalar(255), 1);
		//std::cout << "Lanepos: " << lanepos << std::endl;
		int sumY=0, sumPoint=0;
		// for (int x = imgTmp.cols - 1; x >= 0; x--)
		// {
		// 	if (imgTmp.at<uchar>(yCalc, x) == 255)
		// 	{
		// 		sumY += x;
		// 		sumPoint++;
		// 	}
		// }
		int conditionAngleLanePositive = 50;
		int conditionAngleLaneNegative = -50;
		if(priorityLane > 0)
		{
			if (lanepos == 2)
			{
				for (int x = 0; x <= imgTmp.cols - 1; x++)
				{
					if (imgTmp.at<uchar>(yCalc, x) == 255)
					{
						sumY = x;
						break;
					}
				}
				// if ((angle_lane < conditionAngleLaneNegative) && (angle_lane >= (conditionAngleLaneNegative - 10)))
				// 	return cv::Point(sumY - offset_point - 60, yCalc);
				// else if ((angle_lane < (conditionAngleLaneNegative - 10)) && (angle_lane >= (conditionAngleLaneNegative - 20)))
				// 	return cv::Point(sumY - offset_point - 80, yCalc);
				// else if (angle_lane < (conditionAngleLaneNegative - 20))
				// 	return cv::Point(sumY - offset_point - 100, yCalc);
				// else
				// return cv::Point(sumY - offset_point - (std::abs(angle_lane) - 30)*lane_ratio, yCalc);
				return cv::Point(sumY - offset_point, yCalc);
			}
			else if (lanepos == 1)
			{
				for (int x = imgTmp.cols - 1; x >= 0; x--)
				{
					if (imgTmp.at<uchar>(yCalc, x) == 255)
					{
						sumY = x;
						break;
					}
				}
				// if ((angle_lane > conditionAngleLanePositive) && (angle_lane <= (conditionAngleLanePositive + 10)))
				// 	return cv::Point(sumY + offset_point + 60, yCalc);
				// else if ((angle_lane > (conditionAngleLanePositive + 10)) && (angle_lane <= (conditionAngleLanePositive + 20)))
				// 	return cv::Point(sumY + offset_point + 80, yCalc);
				// else if (angle_lane > (conditionAngleLanePositive + 20))
				// 	return cv::Point(sumY + offset_point + 100, yCalc);
				// else
				// 	return cv::Point(sumY + offset_point + (std::abs(angle_lane) - 30)*lane_ratio, yCalc);
				return cv::Point(sumY + offset_point, yCalc);
			}
		}
		else
		{
			if (lanepos == 1)
			{
				for (int x = imgTmp.cols - 1; x >= 0; x--)
				{
					if (imgTmp.at<uchar>(yCalc, x) == 255)
					{
						sumY = x;
						circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
						break;
					}
				}
				// if ((angle_lane > conditionAngleLanePositive) && (angle_lane <= (conditionAngleLanePositive + 10)))
				// 	return cv::Point(sumY + offset_point + 60, yCalc);
				// else if ((angle_lane > (conditionAngleLanePositive + 10)) && (angle_lane <= (conditionAngleLanePositive + 20)))
				// 	return cv::Point(sumY + offset_point + 80, yCalc);
				// else if (angle_lane > (conditionAngleLanePositive + 20))
				// 	return cv::Point(sumY + offset_point + 100, yCalc);
				// else
				// 	return cv::Point(sumY + offset_point + (std::abs(angle_lane) - 30)*lane_ratio, yCalc);
				return cv::Point(sumY + offset_point, yCalc);
			}
			else if (lanepos == 2)
			{
				for (int x = 0; x <= imgTmp.cols - 1; x++)
				{
					if (imgTmp.at<uchar>(yCalc, x) == 255)
					{
						sumY = x;
						circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
						break;
					}
				}
				// if ((angle_lane < -50.0) && (angle_lane >= -50.0))
				// 	return cv::Point(sumY - offset_point - 60, yCalc);
				// else if ((angle_lane < -60.0) && (angle_lane >= -70.0))
				// 	return cv::Point(sumY - offset_point - 80, yCalc);
				// else if (angle_lane < -70.0)
				// 	return cv::Point(sumY - offset_point - 100, yCalc);
				// else
				// 	return cv::Point(sumY - offset_point - (std::abs(angle_lane) - 30)*lane_ratio, yCalc);
				// if (angle_lane < -50.0)
				// 	return cv::Point(sumY - offset_point - 100, yCalc);
				return cv::Point(sumY - offset_point, yCalc);
			}			
		}
	}
	// lane = imgTmp.clone();
	return cv::Point(-1, -1);
}

cv::Point new_lane_process_yu(const cv::Size &size,
							const cv::Mat &org,
							 cv::Mat &ret,
							int pointMidLane_x,
							int yCalc,
							int offset_point,
							int direct,
							int &lanepos,
							int &angle_lane)
{
	angle_lane = 0;
		//Adaptive Threshold and median filter
	cv::Mat img_erode, img_ad, img_md;
	cv::Mat img_gray;
	cv::cvtColor(org, img_gray, CV_BGR2GRAY);
	cv::Mat img_thr;
	double gmin, gmax, gthr;
	cv::minMaxLoc(img_gray, &gmin, &gmax);
	if (gmax < 150)
		gthr = gmax - (gmax - gmin) * 0.5;
	else
		gthr = gmax - (gmax - gmin) * 0.6;
	cv::threshold(img_gray, img_thr, 180, 255, CV_THRESH_BINARY);

	cv::Mat img_HSV, img_green;
	cv::cvtColor(org, img_HSV, CV_BGR2HSV);
	inRange(img_HSV, cv::Scalar(65, 100, 20), cv::Scalar(80, 255, 255), img_green);

	cv::Mat img = img_thr;// + img_green;

	int esize = 2;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
											cv::Size(2 * esize + 1, 2 * esize + 1),
											cv::Point(esize, esize));
	cv::dilate(img, img, element);
	cv::erode(img, img, element);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	contours.clear();
	hierarchy.clear();
	cv::findContours(img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	double x1Hig = 1e4, x2Hig = -1e4, y1Hig = -1, y2Hig = -1;
	int id1 = -1, id2 = -1;
	lanepos = 0;

	for (int i = 0; i < (int)contours.size(); ++i)
	{
		std::vector<cv::Point> p;
		cv::approxPolyDP(cv::Mat(contours[i]), p, 2, true);

		if (p.size() < 3)
			continue;

		if ((cv::contourArea(p) < MIN_LANE_AREA) || (cv::contourArea(p) > MAX_LANE_AREA))
			continue;

		std::vector<cv::Point> new_4corner = Export4corners(size, contours, i);
		if ((new_4corner[0].y < 30) && (new_4corner[2].y > 105))
		{
			//follow left
			if (new_4corner[2].x < x1Hig) && (new_4corner[2].x < pointMidLane_x)
			{
				y1Hig = new_4corner[2].y;
				x1Hig = new_4corner[2].x;
				id1 = i;
			}
			//follow right
			if (new_4corner[2].x > x2Hig) && (new_4corner[2].x > pointMidLane_x)
			{
				y2Hig = new_4corner[2].y;
				x2Hig = new_4corner[2].x;
				id2 = i;
			}
		}

	}

	//tl tr bl br
	ret = cv::Mat::zeros(size, CV_8UC1);
	if(id1 != -1)
	    cv::drawContours(ret, contours, id1, cv::Scalar(150), CV_FILLED);
    if(id2 != -1)
    	cv::drawContours(ret, contours, id2, cv::Scalar(100), CV_FILLED);

	cv::Mat imgTmp = cv::Mat::zeros(size, CV_8UC1);
	int sumY=0;
	int sumM = 0, sumH = 0;	//middle, high
	bool flag_sumY = false, flag_sumM = false, flag_sumH = false;

	if ((id1 != -1) && (id2 != -1) && (id1 != id2))
	{
		// Set priority

		if ((direct == 2))
			lanepos = 2;
		else if ((direct == 1) || (direct == 0))
			lanepos = 1;
		// lanepos = 1;

		//[Tri] Check priority
		int id_lane = (lanepos == 1)?id1:id2;
		cv::drawContours(imgTmp, contours, id_lane, cv::Scalar(255), 1);
// TEST 1 //////////////////////////////////
		// if (lanepos == 1)
		// {
		// 	for (int x = imgTmp.cols - 1; x >= 0; x--)
		// 	{
		// 		if (imgTmp.at<uchar>(yCalc, x) == 255)
		// 		{
		// 			sumY = x;
		// 			circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
		// 			break;
		// 		}
		// 	}
		// }
		// else if (lanepos == 2)
		// {
		// 	for (int x = 0; x <= imgTmp.cols - 1; x++)
		// 	{
		// 		if (imgTmp.at<uchar>(yCalc, x) == 255)
		// 		{
		// 			sumY = x;
		// 			circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
		// 			break;
		// 		}
		// 	}
		// }
// TEST 2 //////////////////////////////////
		if (lanepos == 1)
		{
			for (int x = imgTmp.cols - 1; x >= 0; x--)
			{
				if ((imgTmp.at<uchar>(yCalc, x) == 255) && !flag_sumY)
				{
					sumY = x;
					circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
					flag_sumY = true;
				}
				if ((imgTmp.at<uchar>(10, x) == 255) && !flag_sumH)
				{
					sumH = x;
					circle(ret, cv::Point(x, 10), 2, cv::Scalar(255), 5);
					flag_sumH = true;
				}
				if ((imgTmp.at<uchar>(45, x) == 255) && !flag_sumM)
				{
					sumM = x;
					circle(ret, cv::Point(x, 45), 2, cv::Scalar(255), 5);
					flag_sumM = true;
				}
				if(flag_sumY & flag_sumM & flag_sumH)
				{
					break;
				}
			}			
			if(sumY == 0)
			{
				if((sumH > sumM) && (sumH > 20) && (sumM > 20))
				{
					sumY = sumM - (sumH - sumM);
				}
			}
		}
		else if (lanepos == 2)
		{
			for (int x = 0; x <= imgTmp.cols - 1; x++)
			{
				if ((imgTmp.at<uchar>(yCalc, x) == 255) && !flag_sumY)
				{
					sumY = x;
					circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
					flag_sumY = true;
				}
				if ((imgTmp.at<uchar>(10, x) == 255) && !flag_sumH)
				{
					sumH = x;
					circle(ret, cv::Point(x, 10), 2, cv::Scalar(255), 5);
					flag_sumH = true;
				}
				if ((imgTmp.at<uchar>(45, x) == 255) && !flag_sumM)
				{
					sumM = x;
					circle(ret, cv::Point(x, 45), 2, cv::Scalar(255), 5);
					flag_sumM = true;
				}
				if(flag_sumY & flag_sumM & flag_sumH)
				{
					break;
				}
			}
			if(sumY == 0)
			{
				if((sumH < sumM) && (sumH < 630) && (sumM < 630))
				{
					sumY = sumM + (-sumH + sumM);
				}
			}
		}
		cv::drawContours(ret, contours, id_lane, cv::Scalar(255), 1);
// END TEST //////////////////////////////////
	}
	else if ((id1 != -1) && (id2 != -1) && (id1 == id2))
	{
		std::vector<cv::Point> ln_corner = Export4corners(size, contours, id2);
		if (((ln_corner[2].x - pointMidLane_x) < 0) && ((ln_corner[3].x - pointMidLane_x) < 0))
		{
			lanepos = 1;
		}
		else if (((ln_corner[2].x - pointMidLane_x) > 0) && ((ln_corner[3].x - pointMidLane_x) > 0))
		{
			lanepos = 2;
		}
		else if (((ln_corner[2].x - pointMidLane_x) < 0) && ((ln_corner[3].x - pointMidLane_x) > 0))
		{
			if (std::abs(ln_corner[2].x - pointMidLane_x) > std::abs(ln_corner[3].x - pointMidLane_x))
			{
				lanepos = 1;
			}
			else
			{
				lanepos = 2;
			}
		}
		int id_lane = (lanepos == 1)?id1:id2;
		cv::drawContours(imgTmp, contours, id_lane, cv::Scalar(255), 1);
// TEST 1 //////////////////////////////////
		// if (lanepos == 1)
		// {
		// 	for (int x = imgTmp.cols - 1; x >= 0; x--)
		// 	{
		// 		if (imgTmp.at<uchar>(yCalc, x) == 255)
		// 		{
		// 			sumY = x;
		// 			circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
		// 			break;
		// 		}
		// 	}
		// }
		// else if (lanepos == 2)
		// {
		// 	for (int x = 0; x <= imgTmp.cols - 1; x++)
		// 	{
		// 		if (imgTmp.at<uchar>(yCalc, x) == 255)
		// 		{
		// 			sumY = x;
		// 			circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
		// 			break;
		// 		}
		// 	}
		// }
// TEST 2 //////////////////////////////////
		if (lanepos == 1)
		{
			for (int x = imgTmp.cols - 1; x >= 0; x--)
			{
				if ((imgTmp.at<uchar>(yCalc, x) == 255) && !flag_sumY)
				{
					sumY = x;
					circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
					flag_sumY = true;
				}
				if ((imgTmp.at<uchar>(10, x) == 255) && !flag_sumH)
				{
					sumH = x;
					circle(ret, cv::Point(x, 10), 2, cv::Scalar(255), 5);
					flag_sumH = true;
				}
				if ((imgTmp.at<uchar>(45, x) == 255) && !flag_sumM)
				{
					sumM = x;
					circle(ret, cv::Point(x, 45), 2, cv::Scalar(255), 5);
					flag_sumM = true;
				}
				if(flag_sumY & flag_sumM & flag_sumH)
				{
					break;
				}
			}
			if(sumY == 0)
			{
				if((sumH > sumM) && (sumH > 20) && (sumM > 20))
				{
					sumY = sumM - (sumH - sumM);
				}
			}
		}
		else if (lanepos == 2)
		{
			for (int x = 0; x <= imgTmp.cols - 1; x++)
			{
				if ((imgTmp.at<uchar>(yCalc, x) == 255) && !flag_sumY)
				{
					sumY = x;
					circle(ret, cv::Point(x, yCalc), 2, cv::Scalar(255), 5);
					flag_sumY = true;
				}
				if ((imgTmp.at<uchar>(10, x) == 255) && !flag_sumH)
				{
					sumH = x;
					circle(ret, cv::Point(x, 10), 2, cv::Scalar(255), 5);
					flag_sumH = true;
				}
				if ((imgTmp.at<uchar>(45, x) == 255) && !flag_sumM)
				{
					sumM = x;
					circle(ret, cv::Point(x, 45), 2, cv::Scalar(255), 5);
					flag_sumM = true;
				}
				if(flag_sumY & flag_sumM & flag_sumH)
				{
					break;
				}
			}
			if(sumY ==0)
			{
				if((sumH < sumM) && (sumH < 620) && (sumM < 620))
				{
					sumY = sumM + (-sumH + sumM);
				}
			}
		}
		cv::drawContours(ret, contours, id_lane, cv::Scalar(255), 1);
// END TEST //////////////////////////////////
	}

	angle_lane = (sumH - sumY);
	if (lanepos > 0)
	{
		// int id;
		// if (lanepos == 1)
		// {
		// 	id = id1;
		// 	std::vector<cv::Point> ln_corner = Export4corners(size, contours, id);
		// 	// angle_lane = getTheta_(ln_corner[3],ln_corner[1]);
		// }
		// else if (lanepos == 2)
		// {
		// 	id = id2;
		// 	std::vector<cv::Point> ln_corner = Export4corners(size, contours, id);
		// 	// angle_lane = getTheta_(ln_corner[2],ln_corner[0]);
		// }
		
		// cv::drawContours(ret, contours, id, cv::Scalar(255), 1);

		if (lanepos == 1)
		{
// TEST 2 //////////////////////////////////
			if((sumH > sumM) && (sumM > sumY) && (sumH > sumY + 80))
			{
				return cv::Point(sumY + offset_point + 10, yCalc);
				// return cv::Point(sumY + offset_point + angle_lane*0.5, yCalc);
			}
			else if ((sumY < 200) && (sumM == 0))
			{
				return cv::Point(-1, -1);
			}
// END TEST //////////////////////////////////

			return cv::Point(sumY + offset_point, yCalc);
		}
		else if (lanepos == 2)
		{
// TEST 2 //////////////////////////////////
			std::cout << "\tH: " << sumH << "\tM: " << sumM << "\tY: " << sumY << std::endl;
			if((sumH < sumM) && (sumM < sumY) && (sumH < sumY - 75))
			{
				return cv::Point(sumY - offset_point  - 50, yCalc);
				// return cv::Point(sumY - offset_point - angle_lane*0.8, yCalc);
			}
			else if ((sumY > 450) && (sumM == 0))
			{
				return cv::Point(-1, -1);
			}
 // END TEST //////////////////////////////////

			return cv::Point(sumY - offset_point + 15, yCalc);
		}			
	}
	return cv::Point(-1, -1);
}






































cv::Point hiden_process(const cv::Size &size,
							const cv::Mat &org,
							 cv::Mat &ret,
							int pointMidLane_x,
							int yCalc,
							int offset_point,
							int direct,
							int &lanepos,
							double &angle_lane)
{
	angle_lane = 0;
	double lane_ratio = 3;

	cv::Mat img_gray, img_thr;
	cv::cvtColor(org, img_gray, CV_BGR2GRAY);
	cv::threshold(img_gray, img_thr, 80, 255, CV_THRESH_BINARY);

	cv::Mat img_HSV, img_green;
	cv::cvtColor(org, img_HSV, CV_BGR2HSV);
	inRange(img_HSV, cv::Scalar(65, 100, 20), cv::Scalar(80, 255, 255), img_green);

	cv::Mat img = img_green + img_thr;

	int esize = 2;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
											cv::Size(2 * esize + 1, 2 * esize + 1),
											cv::Point(esize, esize));
	cv::dilate(img, img, element);
	cv::erode(img, img, element);


	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	contours.clear();
	hierarchy.clear();
	cv::findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	double x1Hig = 1e4, x2Hig = -1e4, y1Hig = -1, y2Hig = -1;
	int id1 = -1, id2 = -1;
	lanepos = 0;

	for (int i = 0; i < (int)contours.size(); ++i)
	{
		std::vector<cv::Point> p;
		cv::approxPolyDP(cv::Mat(contours[i]), p, 2, true);

		if (p.size() < 3)
			continue;

		if ((cv::contourArea(p) < MIN_LANE_AREA)) // || (cv::contourArea(p) > MAX_LANE_AREA))
			continue;

		std::vector<cv::Point> new_4corner = Export4corners(size, contours, i);
		// if ((new_4corner[0].y < 100) && (new_4corner[2].y > 55))
		// {
			//follow left
			if (new_4corner[2].x < x1Hig)
			{
				y1Hig = new_4corner[2].y;
				x1Hig = new_4corner[2].x;
				id1 = i;
			}
			//follow right
			if (new_4corner[2].x > x2Hig)
			{
				y2Hig = new_4corner[2].y;
				x2Hig = new_4corner[2].x;
				id2 = i;
			}
		// }
	}

	ret = cv::Mat::zeros(size, CV_8UC1);
	if(id1 != -1)
	    cv::drawContours(ret, contours, id1, cv::Scalar(150), CV_FILLED);
    if(id2 != -1)
    	cv::drawContours(ret, contours, id2, cv::Scalar(100), CV_FILLED);

	if ((id1 != -1) && (id2 != -1) && (id1 != id2))
	{
		if ((direct == 2) || (direct == 0))
			lanepos = 2;
		else if ((direct == 1))
			lanepos = 1;
	}

	if ((id1 != -1) && (id2 != -1) && (id1 == id2))
	{
		std::vector<cv::Point> ln_corner = Export4corners(size, contours, id2);
		if (((ln_corner[2].x - pointMidLane_x) < 0) && ((ln_corner[3].x - pointMidLane_x) < 0))
		{
			lanepos = 1;
		}
		else if (((ln_corner[2].x - pointMidLane_x) > 0) && ((ln_corner[3].x - pointMidLane_x) > 0))
		{
			lanepos = 2;
		}
		else if (((ln_corner[2].x - pointMidLane_x) < 0) && ((ln_corner[3].x - pointMidLane_x) > 0))
		{
			if (std::abs(ln_corner[2].x - pointMidLane_x) > std::abs(ln_corner[3].x - pointMidLane_x))
			{
				lanepos = 1;
			}
			else
			{
				lanepos = 2;
			}
		}
	}
	cv::Mat imgTmp = cv::Mat::zeros(size, CV_8UC1);

	//Calc point mid lane
	if (lanepos > 0)
	{
		cv::Mat imgTmp = cv::Mat::zeros(size, CV_8UC1);
		if (lanepos == 1)
		{
			cv::drawContours(imgTmp, contours, id1, cv::Scalar(255), 1);
			cv::drawContours(ret, contours, id1, cv::Scalar(255), 1);
			std::vector<cv::Point> ln_corner = Export4corners(size, contours, id1);
			angle_lane = getTheta_(ln_corner[3],ln_corner[1]);
			std::cout << "angle_lane 1:::::::::::::::::: " << angle_lane << std::endl;
			for (int x = imgTmp.cols - 1; x >= 0; x--)
			{
				if (imgTmp.at<uchar>(yCalc, x) == 255)
				{
					if(angle_lane > 75)
						return cv::Point(x + offset_point, -2);
					else if(angle_lane > 60)
						return cv::Point(x + offset_point, -3);
					else return cv::Point(x + offset_point, yCalc);
				}
			}
		}
		else if (lanepos == 2)
		{
			cv::drawContours(imgTmp, contours, id2, cv::Scalar(255), 1);
			cv::drawContours(ret, contours, id2, cv::Scalar(255), 1);
			std::vector<cv::Point> ln_corner = Export4corners(size, contours, id2);
			angle_lane = getTheta_(ln_corner[2],ln_corner[0]);
			std::cout << "angle_lane 2:::::::::::::::::: " << angle_lane << std::endl;
			for (int x = 0; x <= imgTmp.cols - 1; x++)
			{
				if (imgTmp.at<uchar>(yCalc, x) == 255)
				{
					if(angle_lane < -75)
						return cv::Point(x - offset_point, -5);
					else if(angle_lane < -60)
						return cv::Point(x - offset_point, -6);
					else return cv::Point(x - offset_point, yCalc);
				}
			}
		}
	}
	return cv::Point(-1, -1);
}