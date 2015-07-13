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

int main( int argc, char **argv )
{
    /*  set paths for model */
    string model_deploy_file = "../yunc_feature.prototxt";   
    string model_binary_file = "../yunc_feature.model";
    string model_mean_file   = "../face_mean.binaryproto";

    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);
    cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;
    cout<<"output dimension "<<cnnfeature.get_output_dimension("loss2/fc")<<endl;

    Mat input_img1  = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    Mat input_img2  = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

    vector<Mat> ttimgs2;
    ttimgs2.push_back( input_img1);
    ttimgs2.push_back( input_img2);

    Mat face_feas;
    cnnfeature.extract_blob( "pool5/7x7_s1", ttimgs2, face_feas );
    cout<<"feature dimension "<<face_feas.rows<<" "<<face_feas.cols<<endl;
    cout<<"Cos simi "<<cosine_similarity( face_feas.row(0), face_feas.row(1))<<endl;
    imshow("show", input_img1);
    waitKey(0);

    /*  test on positive pair  */
    /* --------------------------------------------------------------------------- */
    /*  test on the folder */
    //string folder_root = "/home/yuanyang/data/face_recognition/H_2/H_2/";
    //string folder_root2 = "/home/yuanyang/data/face_recognition/L_2/L_2/";

    //bf::directory_iterator end_it;
    //for( bf::directory_iterator file_iter( folder_root); file_iter !=end_it; file_iter++)
    //{
    //    string pathname = file_iter->path().string();
    //    string basename = bf::basename( *file_iter);
    //    string extname = bf::extension( *file_iter);
    //    
    //    vector<Mat> timgs;
    //    Mat input_img = imread(  pathname , CV_LOAD_IMAGE_GRAYSCALE);
    //    Mat input_img2 = imread(  folder_root2+basename+extname , CV_LOAD_IMAGE_GRAYSCALE);

    //    //cout<<"input_img size "<<input_img.rows<<" "<<input_img.cols<<endl;
    //    //cout<<"input_img2 size "<<input_img2.rows<<" "<<input_img2.cols<<endl;

    //    //cout<<"img1 path is "<<pathname<<endl;
    //    //cout<<"img2 path is "<<folder_root2+basename+extname<<endl;

    //    timgs.push_back( input_img );
    //    timgs.push_back( input_img2 );
    //    imshow( "show1 ", input_img );
    //    imshow( "show2 ", input_img2 );

    //    Mat no_use_feature;
    //    //cout<<"row 1 "<<endl;
    //    cnnfeature.extract_blob( "loss2/fc", timgs, no_use_feature );
    //    //cout<<"no_use_feature "<<no_use_feature.rows<<" "<<no_use_feature.cols<<endl;
    //    //cout<<no_use_feature<<endl;

    //    cout<<cosine_similarity( no_use_feature.row(0), no_use_feature.row(1))<<endl;
    //    //waitKey(0);
    //}
    /* --------------------------------------------------------------------------- */


    /* 2 test on negative pair */
    string folder_root = "/home/yuanyang/data/face_recognition/test_samples_old_2000/H_2/H_2/";

    vector<string> path_list1;
    vector<string> path_list2;

    ofstream ofs("neg_pair.txt"); 

    bf::directory_iterator end_it;
    for( bf::directory_iterator file_iter( folder_root); file_iter !=end_it; file_iter++)
    {
        path_list1.push_back( file_iter->path().string());
        path_list2.push_back( file_iter->path().string());
    }
    
    for( int i=0;i<path_list1.size();i++)
    {
        for( int j=0;j<path_list2.size();j++)
        {
            if( i==j)
                continue;

            //cout<<"path 1 is "<<path_list1[i]<<endl;
            //cout<<"path 1 is "<<path_list2[j]<<endl;

            Mat img1 = imread( path_list1[i], CV_LOAD_IMAGE_GRAYSCALE);
            Mat img2 = imread( path_list2[j], CV_LOAD_IMAGE_GRAYSCALE);

            //cout<<"img 1 is "<<img1.cols<<" "<<img1.rows<<endl;
            //cout<<"img 1 is "<<img2.cols<<" "<<img2.rows<<endl;

            vector<Mat> timgs;
            timgs.push_back( img1 );
            timgs.push_back( img2 );

            Mat no_use_feature;
            //cout<<"row 1 "<<endl;
            cnnfeature.extract_blob( "loss2/fc", timgs, no_use_feature );
            //cout<<"no_use_feature "<<no_use_feature.rows<<" "<<no_use_feature.cols<<endl;
            //cout<<no_use_feature<<endl;

            cout<<cosine_similarity( no_use_feature.row(0), no_use_feature.row(1))<<endl;
            ofs<<cosine_similarity( no_use_feature.row(0), no_use_feature.row(1))<<endl;
        }
    }
    ofs.close();
    return 0;
}
