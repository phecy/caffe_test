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
#include <map>
#include <sstream>

#include "boost/filesystem.hpp"
#include "boost/serialization/serialization.hpp"
#include "boost/serialization/map.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cnn_master.hpp"

using namespace std;
using namespace cv;


namespace bf=boost::filesystem;

bool save_wrong( int order_num,
                 const string &wrong_info,
                 const Mat &img)
{
    if( !bf::is_directory("./wrong"))
    {
        cout<<"Do not have folder ./wrong, skip"<<endl;
        return false;
    }
    
    stringstream ss;
    ss<<order_num;
    string extfix;
    ss>>extfix;

    string save_wrong_name = "./wrong/"+wrong_info+extfix+".jpg";
    imwrite(save_wrong_name, img);
}

template<class T> bool save_map( T &codebook,
               const string &path_to_save)
{
    ofstream ss(path_to_save.c_str());
    if( !ss.is_open() )
    {
        cout<<"can not open file "<<path_to_save<<endl;
        return false;
    }
    boost::archive::text_oarchive oarch(ss);
    oarch<<codebook;
    return true;
}

template< class T>
bool load_map( T &codebook,
            const string &path_of_file)
{
    ifstream ss( path_of_file.c_str());
    if( !ss.is_open())
    {
        cout<<"can not open file "<<path_of_file<<endl;
        return false;
    }
    boost::archive::text_iarchive iarch(ss);
    iarch>>codebook;
    return true;
}


bool read_in_list( const string &list_file,
                    vector<string> &file_paths,
                    vector<int> &labels)
{
    FILE *fp = fopen( list_file.c_str(), "r");
    if( fp == NULL)
    {
        cout<<"Can not open file "<<list_file<<endl;
        return false;
    }
    int r;
    char file_name_buffer[50];
    int file_label;
    while(1)
    {
        r = fscanf( fp, "%s %d\n", &file_name_buffer, &file_label);
        if( r == EOF)
            break;

        file_paths.push_back( string(file_name_buffer));
        labels.push_back( file_label);
    }
    return true;
}

int main( int argc, char **argv )
{
    vector<string> file_list;
    vector<int> file_labels;
    if( argc !=3 )
    {
        cout<<"Usage: ./test_license test_image_root_path test_list_file"<<endl;
        return -1;
    }
    string test_root_path(argv[1]);
    read_in_list( argv[2], file_list, file_labels);
    
    /*  load label ->class and class->label map */
    map<int,string> label_to_class;
    map<string,int> class_to_label;
    if(!load_map( label_to_class, "label_to_class_map.data") ||
       !load_map( class_to_label, "class_to_label_map.data"))
    {
        cout<<"can not load map data"<<endl;
        return -3;
    }

    /* wrong classify count, "R_to_B number_of_wrong" */
    map<string, int> wrong_count;

    /*  set paths for model */
    string model_deploy_file = "../deploy_license.prototxt";   
    string model_mean_file   = "../license_char_mean.binaryproto";
    string model_binary_file = "../license_char_model__iter_20000.caffemodel";

    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);
    cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;
    cout<<"output dimension "<<cnnfeature.get_output_dimension("fc6")<<endl;

    /*  test on the folder */
    //Mat predict_label;
    //cnnfeature.get_class_label( timgs, predict_label );
    //cout<<"label :"<<predict_label.at<float>(1,0)<<", confidence: "<<predict_label.at<float>(1,1)<<endl;

    /* test on folder */
    for(unsigned int c=0;c<file_list.size();c++)
    {
        vector<Mat> imagelists;
        string image_path = test_root_path+"/"+file_list.at(c);
        Mat input_image = imread( image_path, CV_LOAD_IMAGE_GRAYSCALE);

        /* scale */
        cv::resize( input_image, input_image, Size(0,0), 0.3, 0.3 );
        cv::resize( input_image, input_image, Size(48,48), 0, 0 );

        imagelists.push_back( input_image);

        Mat predict_label;

        cnnfeature.get_class_label( imagelists, predict_label);

        /* record wrong item */
        if( int(predict_label.at<float>(0,0)) != file_labels.at(c) )
        {
            string wrong_id = label_to_class[file_labels.at(c)]+"_to_"+label_to_class[predict_label.at<float>(0,0)];
            if( wrong_count.count( wrong_id) == 0 )
                wrong_count[wrong_id] = 1;
            else
                wrong_count[wrong_id] += 1;

            cout<<"wrong_id is "<<wrong_id<<endl;
            save_wrong( wrong_count[wrong_id], wrong_id, input_image);
        }

        /* show result */
        //cout<<"predict to label "<<predict_label.at<float>(0,0)<<" with confidence "<<predict_label.at<float>(0,1)<<endl;
        //cout<<"ground truth label is "<<file_labels.at(c)<<endl;
        //cout<<"convert the label to string -> "<<label_to_class[int(predict_label.at<float>(0,0))]<<endl;

        /*  the output of the new topk function */
        //vector<Mat> labelconfs;
        //cnnfeature.get_topk_label( imagelists, labelconfs, 3 );
        //for( unsigned int c=0;c<labelconfs.size();c++)
        //{
        //    cout<<"labelconf is "<<labelconfs[c]<<endl;
        //}

        //imshow("test", input_image);
        //waitKey(0);
    }


    ofstream output_wrongid("wrong_stat.txt");
    for( map<string, int>::iterator it=wrong_count.begin(); it!=wrong_count.end();it++)
    {
        output_wrongid<<it->first<<"  --> "<<it->second<<endl;
    }
    output_wrongid.close();
    return 0;
}

