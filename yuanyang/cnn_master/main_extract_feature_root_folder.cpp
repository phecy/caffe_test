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
#include <fstream>

#include "boost/filesystem.hpp"
#include "boost/lambda/bind.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cnn_master.hpp"

using namespace std;
using namespace cv;

#define DO_MIRROR 0
#define DO_BLUR 0

namespace bf=boost::filesystem;
namespace bl=boost::lambda;

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
    //string model_deploy_file = "face_deploy.prototxt";   
    //string model_binary_file = "small_max_out.caffemodel";
    //string model_mean_file   = "";

    string model_deploy_file = "small_max_out.prototxt";   
    string model_binary_file = "small_max_out.caffemodel";
    string model_mean_file   = "";

    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);

    cnnfeature.set_input_width( 180 );
    cnnfeature.set_input_height(180) ;
    cnnfeature.set_input_channel(1);

    //cnnfeature.set_input_width( 144 );
    //cnnfeature.set_input_height(144) ;
    //cnnfeature.set_input_channel(1);

    cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;

    /* 2 test on negative pair */
    //string folder_root = "/home/yuanyang/data/face_recognition/celes_plus_diaosi/";
    //string folder_root = "/home/yuanyang/data/face_recognition/diaosi_crop/";
    //string folder_root = "veri_data/";

    string folder_root = "/home/yuanyang/data/face_recognition/lfw/pos/";
    //string folder_root = "/home/yuanyang/data/face_recognition/verification/";

    bf::directory_iterator end_it;
	for( bf::directory_iterator folder_iter( folder_root); folder_iter!=end_it; folder_iter++)
	{
		if( !bf::is_directory(*folder_iter))
			continue;

        int number_of_images = getNumberOfFilesInDir( folder_iter->path().string());
        cout<<"processing subfolder "<<folder_iter->path().string()<<endl;
        cout<<"   number of image is "<<number_of_images<<endl;

        /* get Feature */
        vector<Mat> input_imgs;
        Mat features;

        bool first_image = true;
        string folder_name;

        bf::directory_iterator end_it2;
	    for( bf::directory_iterator file_iter( folder_iter->path() ); file_iter!=end_it2; file_iter++)
	    {
	    	string pathname = file_iter->path().string();
	    	string basename = bf::basename( *file_iter);
	    	string extname  = bf::extension( *file_iter);

            if( first_image)
                folder_name = get_folder_name( pathname );

            if( extname != ".jpg" && extname != ".png" && extname != ".bmp")
                continue;

            Mat input_img = imread( pathname, CV_LOAD_IMAGE_GRAYSCALE);


            cv::resize( input_img, input_img, Size(cnnfeature.get_input_width(),cnnfeature.get_input_height()), 0, 0);

            //resize(input_img, input_img, Size(128,128), 0, 0);
            //cvtColor( input_img, input_img, CV_BGR2GRAY);
            input_imgs.push_back( input_img);
            if( DO_MIRROR)
            {
                Mat mirror_img;
                cv::flip( input_img, mirror_img, 1);
                input_imgs.push_back(mirror_img);
                //imshow("input", input_img);
                //imshow("flip", mirror_img);
                //waitKey(0);
            }
            if(DO_BLUR)
            {
                Mat block_norm;
                Mat gaussian_norm;
                Mat median_norm;
                Mat bilateral_norm;
                Mat s_back;
                cv::resize( input_img, s_back, Size(0,0), 0.2, 0.2 );
                cv::resize( s_back, s_back, Size(cnnfeature.get_input_width(), cnnfeature.get_input_height()), 0, 0 );

                int ker_size = 9; 
                cv::GaussianBlur(input_img, gaussian_norm, Size(ker_size,ker_size),0,0);
                input_imgs.push_back( s_back);
                input_imgs.push_back(gaussian_norm);
                //imshow("input", input_img);
                //imshow("s_back", s_back);
                //imshow("gaussian", gaussian_norm);
                //waitKey(0);
            }
        }
        
        cout<<"folder_name is "<<folder_name<<endl;
        cnnfeature.extract_blob( "eltwise11", input_imgs, features);
        cout<<"feature's size is "<<features.cols<<" "<<features.rows<<endl;
        saveMatToFile( features, "lfw_pos/"+folder_name+".mat");
	}
    return 0;
}
