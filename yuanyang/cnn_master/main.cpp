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


namespace bf=boost::filesystem;

int main( int argc, char **argv )
{
    /*  set paths for model */
    string model_deploy_file = string(argv[1]);   
    string model_mean_file   = string(argv[2]);
    string model_binary_file = string(argv[3]);

    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);
    cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;
    cout<<"output dimension "<<cnnfeature.get_output_dimension("prob")<<endl;

    Mat img1 = imread("f1.jpg");   
    Mat img2 = imread("f2.jpg");   
    Mat img3 = imread("f3.jpg");   
    Mat img4 = imread("f4.jpg");   
    Mat img5 = imread("e.jpg");   
    
    vector<Mat> imagelist;
    imagelist.push_back(img1);
    imagelist.push_back(img2);
    imagelist.push_back(img3);
    imagelist.push_back(img4);
    imagelist.push_back(img5);


    for(int c=0;c<imagelist.size();c++)
        cv::resize( imagelist[c], imagelist[c], Size(256,256), 0, 0 );

    Mat output_feature;

    cnnfeature.extract_blob( "fc7", imagelist, output_feature);

    cout<<"f1 f2 dis "<<cv::norm( output_feature.row(0) - 
                                  output_feature.row(1))<<endl;

    cout<<"f1 f3 dis "<<cv::norm( output_feature.row(0) - 
                                  output_feature.row(2))<<endl;


    cout<<"f1 f4 dis "<<cv::norm( output_feature.row(0) - 
                                  output_feature.row(3))<<endl;


    cout<<"f1 e dis "<<cv::norm( output_feature.row(0) - 
                                  output_feature.row(4))<<endl;
    return 0;
}
