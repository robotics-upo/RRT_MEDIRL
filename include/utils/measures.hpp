#ifndef MEASURES_H
#define MEASURES_H


#include "opencv2/opencv.hpp"

#define FP_FAST_FMA  1 /* implementation-defined */
#define FP_FAST_FMAF 1 /* implementation-defined */
#define FP_FAST_FMAL 1 /* implementation-defined */

namespace Utils
{
namespace Measures
{
	float ComputeDist( const cv::Point2i& p1, const cv::Point2i& p2)
	{
		return hypotf( p1.x - p2.x, p1.y - p2.y);
	}

	cv::Point2i FindPointInLine(const float porcent, const cv::Point2i& start, const cv::Point2i& end)
	{
		return cv::Point2i( std::fma( porcent, (end.x - start.x ), start.x), 
							std::fma( porcent, (end.y - start.y ), start.y));
	}

	template<typename T, T EPS>
	cv::Point2i GetPointNear(const cv::Point2i& fixed_p, const cv::Point2i& p)
	{
		float ang = atan2(p.y - fixed_p.y , p.x - fixed_p.x );
		return cv::Point2i( fixed_p.x + EPS * cos(ang), fixed_p.y + EPS * sin(ang));
	}	
	
}
}



#endif
