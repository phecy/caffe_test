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

#include "cnn_feature_extractor.hpp"

using namespace std;
using namespace cv;


namespace bf=boost::filesystem;

int main( int argc, char **argv )
{
    /*  set paths for model */
    string model_deploy_file = "../deploy.prototxt";   
    string model_binary_file = "../car_fine_tuning_iter_30000.caffemodel";
    string model_mean_file   = "../imagenet_mean.binaryproto";

    cnn_feature_extractor cnnfeature( model_deploy_file, model_mean_file, model_binary_file);
    cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;
    cout<<"output dimension "<<cnnfeature.get_output_dimension()<<endl;

    /*  test on the folder */
    string folder_root( argv[1] );
    bf::directory_iterator end_it;
    for( bf::directory_iterator file_iter( folder_root); file_iter !=end_it; file_iter ++)
    {
        vector<Mat> timgs;
        string pathname = file_iter->path().string();
        Mat input_img = imread(  pathname );
        timgs.push_back( input_img );
        imshow( "show ", input_img );
        Mat no_use_feature;
        cnnfeature.extract_cnn( timgs, no_use_feature );
        waitKey(0);
    }
    



    //vector<Mat> test_imgs;
    //Mat img1 = imread( argv[1] );
    //Mat img2 = imread( argv[2] );

    //test_imgs.push_back( img1 );
    //test_imgs.push_back( img2 );
    //
    //Mat no_ues_features;
    //cnnfeature.extract_cnn( test_imgs, no_ues_features );
    return 0;
}
