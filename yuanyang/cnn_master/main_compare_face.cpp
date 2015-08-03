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
#include "boost/lambda/bind.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cnn_master.hpp"

using namespace std;
using namespace cv;

namespace bf=boost::filesystem;
namespace bl=boost::lambda;


struct pair_info
{
    string image1;
    string image2;
    int label;
};

bool read_in_file_list( const string &filelist,
                        vector<pair_info> &train_file)
{
    train_file.clear();
    std::ifstream in_file( filelist.c_str());
    if( !in_file.is_open())
    {
        LOG(WARNING)<<"Can not open file "<<filelist;
        return false;
    }
    
    string image1, image2;
    int label=0;

    while( in_file>>image1>>image2>>label )
    {
        pair_info t;
        t.image1 = image1;
        t.image2 = image2;
        t.label = label;
        train_file.push_back( t);
    }

    return true;
}

void saveMatToFile( const Mat& data,
                    const string &path)
{
    if( data.empty() || data.type()!=CV_32F)
    {
        cout<<"data is empty "<<endl;
        return;
    }

    ofstream ofs( path.c_str());
    
    for( int r=0;r<data.rows;r++)
    {
        for( int c=0;c<data.cols;c++)
        {
            ofs<<data.at<float>(r,c)<<" ";
        }
        ofs<<endl;
    }
    ofs.close();
}

size_t getNumberOfFilesInDir( string in_path )
{
    bf::path c_path(in_path);
    if( !bf::exists(c_path))
        return -1;
    if( !bf::is_directory(c_path))
        return -1;

    int cnt = std::count_if(
        bf::directory_iterator( c_path ),
        bf::directory_iterator(),
        bl::bind( static_cast<bool(*)(const bf::path&)>(bf::is_regular_file),
        bl::bind( &bf::directory_entry::path, bl::_1 ) ) );
    return cnt;
}


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

string get_folder_name( const string &fullpath)
{
	size_t pos1 = fullpath.find_last_of("/");
	string sub_str = fullpath.substr(0,pos1);
	size_t pos2 = sub_str.find_last_of("/");

	return fullpath.substr(pos2+1,pos1-pos2-1);
}

int main( int argc, char **argv )
{
    /*  set paths for model */
    string model_deploy_file = "sensetime.prototxt";   
    string model_binary_file = "sensetime.model";
    string model_mean_file   = "";

    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);

    cnnfeature.set_input_width(256);
    cnnfeature.set_input_height(256);
    cnnfeature.set_input_channel(3);

    cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;
    cout<<"output dimension "<<cnnfeature.get_output_dimension("output")<<endl;

    /* 2 test on negative pair */
    //string folder_root = "/home/yuanyang/data/face_recognition/verification/id_test/";
    //string folder_root = "/home/yuanyang/data/face_recognition/CASIA/casia_crop/";
    //string folder_root = "/home/yuanyang/data/face_recognition/lfw/lfw_crop/pos/";


    /* read in the list and compare */
    //vector<pair_info> train_file;
    //read_in_file_list( string(argv[1]), train_file);

    //for( unsigned long i=0;i<train_file.size();i++)
    //{
        cv::Mat img1  = cv::imread( argv[1] );
        cv::Mat img2  = cv::imread( argv[2] );

        cv::resize( img1, img1, Size(256,256), 0, 0);
        cv::resize( img2, img2, Size(256,256), 0, 0);

       // cout<<"label is "<<train_file[i].label<<endl;

        vector<Mat> imgs;
        imgs.push_back( img1);
        imgs.push_back(img2);

        Mat features;
        cnnfeature.extract_blob( "output", imgs, features);
        imshow("img1", img1);
        imshow("img2", img2);

        cout<<"feature dim "<<cnnfeature.get_output_dimension("output")<<endl;
        cout<<"feature : "<<features<<endl;
        //cout<<"feature dim "<<features.cols<<" "<<features.rows<<endl;
        cout<<"distance is "<<cosine_similarity( features.row(0), features.row(1))<<endl;
        waitKey(0);
    //}

    return 0;
}
