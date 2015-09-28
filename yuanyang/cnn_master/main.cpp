/*
 * =====================================================================================
 *
 *       Filename:  test_on_fly.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2015年05月11日 21时43分32秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YuanYang (), bengouawu@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>

#include "boost/filesystem.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cnn_master.hpp"

using namespace std;
using namespace cv;

double cosine_similarity( const Mat &sample1,
                          const Mat &sample2)
{
    if( sample1.empty() || sample2.empty() || 
        sample1.rows != 1 || sample2.rows!= 1||
        sample1.cols != sample2.cols)
        return 0;
    double a_m_b = 0;
    double a_norm = 0;
    double b_norm = 0;

    for( int c=0;c<sample1.cols;c++)
    {
        a_m_b += sample1.at<float>(0,c)*sample2.at<float>(0,c);
        a_norm += sample1.at<float>(0,c)*sample1.at<float>(0,c);
        b_norm += sample2.at<float>(0,c)*sample2.at<float>(0,c);
    }
    
    return a_m_b/( sqrt(a_norm) * sqrt(b_norm));
}


namespace bf=boost::filesystem;

int main( int argc, char **argv )
{
    /*  set paths for model */
    string model_deploy_file = string(argv[1]);   
    string model_binary_file = string(argv[2]);
    string model_mean_file = string(argv[3]);

    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);

    Mat img1 = imread(string(argv[4]));   
    if(img1.empty())
    {
        cout<<"input image is empty!"<<endl;
        return -1;
    }
    vector<Mat> imagelist;
    imagelist.push_back(img1);

    for(int c=0;c<imagelist.size();c++)
    {
        imshow("input",  imagelist[c]);
        cv::resize( imagelist[c], imagelist[c], Size(256,256), 0, 0 );
    }

    Mat output_feature;

    cnnfeature.extract_blob( "prob", imagelist, output_feature);
    cout<<"output feature size "<<output_feature.cols<<" "<<output_feature.rows<<endl;

    cout<<"feature 1 "<<output_feature.row(0)<<endl;


        waitKey(0);
    return 0;
}
