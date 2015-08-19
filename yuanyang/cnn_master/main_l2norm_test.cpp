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
    string model_deploy_file = "triplet_deploy.prototxt";   
    string model_binary_file = "triplet_deploy.caffemodel";
    string model_mean_file   = "";

    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);

    cnnfeature.set_input_width(144);
    cnnfeature.set_input_height(144);
    cnnfeature.set_input_channel(1);

    cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;
    cout<<"output dimension "<<cnnfeature.get_output_dimension("l2_norm")<<endl;


    /* 2 test on negative pair */
    string folder_root1 = "/home/yuanyang/data/face_recognition/verification/1/";
    string folder_root2 = "/home/yuanyang/data/face_recognition/verification/2/";

    //string folder_root1 = "/home/yuanyang/libs/caffe/yuanyang/cnn_master/build/pci_staff/real/";
    //string folder_root2 = "/home/yuanyang/libs/caffe/yuanyang/cnn_master/build/pci_staff/card/";

    //string folder_root1 = "/home/yuanyang/data/face_recognition/lfw/pos/1/";
    //string folder_root2 = "/home/yuanyang/data/face_recognition/lfw/pos/2/";

    int wrong_counter = 0;

    bf::directory_iterator end_it;
    for( bf::directory_iterator folder_it(folder_root1); folder_it != end_it;folder_it++ )
    {
        string pathname1 = folder_it->path().string();
        string basename = bf::basename( *folder_it);
        string extname = bf::extension( *folder_it);
        string pathname2 = folder_root2 + basename + extname;

        cout<<"reading image "<<pathname2<<endl;

        cv::Mat img1  = cv::imread( pathname1, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat img2  = cv::imread( pathname2, CV_LOAD_IMAGE_GRAYSCALE);

        if( img1.empty())
            return -1;

        cv::resize( img1, img1, Size(cnnfeature.get_input_width(),cnnfeature.get_input_height()), 0, 0);
        cv::resize( img2, img2, Size(cnnfeature.get_input_width(),cnnfeature.get_input_height()), 0, 0);


        vector<Mat> imgs;
        imgs.push_back( img1);
        imgs.push_back(img2);

        Mat features;
        cnnfeature.extract_blob( "l2_norm", imgs, features);

        //cout<<"feature dim "<<features.cols<<" "<<features.rows<<endl;
        float cos_dis =  cosine_similarity( features.row(0), features.row(1));
        float l2_dis = cv::norm( features.row(0) ,features.row(1));
        
        if(  l2_dis > 1.04)
        {
            wrong_counter++;
            cout<<"l2_dis "<<l2_dis<<endl;
            cout<<"cos_dis "<<cos_dis<<endl;
            imshow("img1", img1);
            imshow("img2", img2);
            waitKey(0);
        }

    }
    cout<<"counter is "<<wrong_counter<<endl;

    return 0;
}
