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
#include "misc.hpp"

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
    char file_name_buffer[800];
    int file_label;
    while(1)
    {
        r = fscanf( fp, "%s %d\n", &file_name_buffer, &file_label);
        if( r == EOF)
            break;

        file_paths.push_back( string(file_name_buffer));
        labels.push_back( file_label);
    }
    cout<<"Scanning file done "<<endl;
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

    cout<<"Map loaded "<<endl;
    /* wrong classify count, "R_to_B number_of_wrong" */
    map<string, int> wrong_count;
    map<int, vector<int> > decision_list;

    /*  set paths for model */
    string model_deploy_file = "../../car_type/caffenet_model/deploy_car.prototxt";   
    string model_mean_file   = "../../car_type/car_type_mean.binaryproto";
    string model_binary_file = "../../car_type/pci_car_type_stage2__iter_30000.caffemodel";

    //string model_deploy_file = "deploy_wine.prototxt";   
    //string model_mean_file   = "wine_labels_mean.binaryproto";
    //string model_binary_file = "wine_model.caffemodel";
    
    cout<<"Loading model .."<<endl;
    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);

    cnnfeature.set_input_channel(3);
    cnnfeature.set_input_width(256);
    cnnfeature.set_input_height(256);

    cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;

    /* test on folder */
    for(unsigned int c=0;c<file_list.size();c++)
    {
        /* create the list for the first time */
        if( decision_list.count(file_labels.at(c)) == 0)
        {
            vector<int> empty_result;
            decision_list[file_labels.at(c)] = empty_result;
        }

        vector<Mat> imagelists;
        string image_path = test_root_path+"/"+file_list.at(c);
        Mat input_image = imread( image_path );
		//imshow("origin", input_image);


		cv::resize( input_image, input_image, Size(256, 256), 0, 0);

        if( input_image.channels() != cnnfeature.get_input_channels())
        {
            cout<<"input channels "<<input_image.channels()<<endl;
            cout<<"cnn_feature input channel "<<cnnfeature.get_input_channels()<<endl;
            cout<<"skip those who has wrong channel "<<image_path<<endl;
            continue;
        }

        imagelists.push_back( input_image);

        Mat predict_label;

        cnnfeature.get_class_label( imagelists, predict_label);

        /* record the decision */
        decision_list[file_labels.at(c)].push_back( int(predict_label.at<float>(0,0)));

        /* record wrong item */
        if( int(predict_label.at<float>(0,0)) != file_labels.at(c) )
        {
            string wrong_id = label_to_class[file_labels.at(c)]+"__to__"+label_to_class[predict_label.at<float>(0,0)];
            if( wrong_count.count( wrong_id) == 0 )
                wrong_count[wrong_id] = 1;
            else
                wrong_count[wrong_id] += 1;

            cout<<"wrong_id is "<<wrong_id<<endl;
            //save_wrong( wrong_count[wrong_id], wrong_id, input_image);

			//imshow("wrong", input_image);
        }

        /* show result */
        cout<<"predict to label "<<predict_label.at<float>(0,0)<<" with confidence "<<predict_label.at<float>(0,1)<<endl;
        cout<<"ground truth label is "<<file_labels.at(c)<<endl;
        cout<<"convert the label to string -> "<<label_to_class[int(predict_label.at<float>(0,0))]<<endl;

        /*    the output of the new topk function */
        vector<Mat> labelconfs;
        cnnfeature.get_topk_label( imagelists, labelconfs, 3 );
        for( unsigned int c=0;c<labelconfs.size();c++)
        {
            cout<<"labelconf is "<<labelconfs[c]<<endl;
        }

        //imshow("test", input_image);
        //waitKey(0);
    }

    /* record the decision error */
    ofstream output_wrongid("wrong_stat.txt");
    cout<<"Recording the error "<<endl;
    for( map<int, vector<int> >::iterator it=decision_list.begin(); it!=decision_list.end();it++)
    {
        int number_of_test_sample = it->second.size();
        int number_of_wrong = 0;
        for( int c=0;c<it->second.size();c++)
        {
            if( it->first != it->second.at(c))
                number_of_wrong++;
        }
        double acc_ratio = 1.0*(it->second.size()-number_of_wrong)/it->second.size();
        if( acc_ratio < 0.9)
            output_wrongid<<"-----> less than 0.9 <------"<<endl;

        output_wrongid<<"Class "<<it->first<<"\t\t("<<label_to_class[it->first]<<") \t\t, Total number:"<<it->second.size()<<
            ", Wrong number:"<<number_of_wrong<<", accuracy: "<<acc_ratio<<endl;

        for( int c=0;c<it->second.size();c++)
        {
            if( it->first != it->second.at(c))
                output_wrongid<<"\t from "<<label_to_class[it->first]<<" to "<<label_to_class[ it->second.at(c) ]<<endl;
        }
    }
    output_wrongid.close();
    return 0;
}

