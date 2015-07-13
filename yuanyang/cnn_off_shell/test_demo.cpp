#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>

#include "boost/filesystem.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "cnn_master.hpp"

#include "opencv_warpper_libsvm.h"

using namespace std;
namespace bf=boost::filesystem;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
        cout<<"using it like this: ./extract_feature path_of_img_folder"<<endl;
        return -1;
    }
    
    /*  Load the network model */
    string net_model_file = "./imagenet_val.prototxt";
    string trained_model_file = "./bvlc_reference_caffenet.caffemodel";
    string mean_file_path = "./imagenet_mean.binaryproto";

    cout<<"Load the network model "<<endl;
    cnn_master cnnfeature;
    cnnfeature.load_model( net_model_file, mean_file_path, trained_model_file);
    cout<<"input should have width : "<<cnnfeature.get_input_width()<<std::endl;
    cout<<"input should have height : "<<cnnfeature.get_input_height()<<std::endl;
    cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<std::endl;
    
    /* Load the svm model */
    opencv_warpper_libsvm svm_classifier;
    if( !svm_classifier.load_model( "people_car_svm.model","people_car_svm_info.xml") )
    {
        cout<<"Can not load the model "<<endl;
        return -1;
    }
    

    /*  test on folder */
    string temp_path( argv[1] );
    bf::path folder_path( temp_path);
    if(!bf::exists(folder_path))
    {
        cout<<"Path "<<folder_path<<" does not exist "<<endl;
        return -2;
    }
    
    bf::directory_iterator end_it;
    for( bf::directory_iterator file_iter(folder_path); file_iter != end_it; file_iter++)
    {
        string pathname = file_iter->path().string();
        string basename = bf::basename( *file_iter);
        string extname  = bf::extension( *file_iter);
        if( extname != ".jpg" &&  extname !=".png" )
            continue;
        
        vector<cv::Mat> input_mats;
        cv::Mat input_img = cv::imread( pathname );
        cv::resize( input_img , input_img, cv::Size(256,256), 0, 0);

        input_mats.push_back( input_img );

        cv::Mat cnn_features;
        cnnfeature.extract_blob( "fc7", input_mats, cnn_features);

        Mat predict_value;
        svm_classifier.predict( cnn_features, predict_value );

        cout<<"predict value is "<<predict_value<<endl;
        if( predict_value.at<float>(0,0) > 0)
        {
            cout<<"wrong .."<<endl;
            cv::imwrite("./wrong/"+basename+".jpg",  input_img);
        }

        cv::imshow("input_img", input_img);
        cv::waitKey(10);

    }

    return 0;
}

