#ifndef MISC_HPP
#define MISC_HPP
#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

Rect bbsToRect( int xmin, int xmax, int ymin, int ymax );

/*  resize the bounding box, based on the center--> doesn't change the Rect's center */
/*  mind result maybe less than zero ~~*/
Rect resizeBbox( const Rect &inrect, double h_ratio, double w_ratio );

/* crop image according to the rect, region will padded with boudary pixels if needed */
Mat cropImage( const Mat &input_image, const Rect &inrect );

void sampleRects( int howManyToSample, Size imageSize, Size objectSize, vector<Rect> &outputRects);

Rect resizeToFixedRatio( const Rect &inRect,				/* in : input boudingbox informtion */
					     double w_h_ratio,					/* in : target ratio */
						 int flag = 1 )	;						/* in : respect to width  = 0 
																	respect to height > 0 */
bool saveMatToFile( string path_name, const Mat & m);

bool colorEqu( const Mat &input_image, 
                Mat &output_image);


class reIllumination
{
    public:
    reIllumination();
    ~reIllumination();
    Mat  get_0_kernel( const Size target_size, int min_value, int max_value);
    Mat get_90_kernel( const Size target_size, int min_value, int max_value);
    Mat get_45_kernel( const Size target_size, int min_value, int max_value);
    Mat get_135_kernel( const Size target_size, int min_value, int max_value);

    void add_kernel( const Mat &kernel);

    void addIllumination( const Mat &input_image, vector<Mat> &output_images);

    private:
    /*  no copy */
    reIllumination( const reIllumination &rhs);
    reIllumination& operator=( reIllumination &rhs);

    vector<Mat> m_kernels;
};
#endif
