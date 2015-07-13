#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>

#include "boost/filesystem.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cnn_master.hpp"

using namespace std;
namespace bf=boost::filesystem;

void read_data( const string &folder,
                vector<cv::Mat> &mats,
                vector<string> &paths)
{
    paths.clear();
    mats.clear();

    bf::path folder_path(folder);
    if(!bf::exists( folder_path) || !bf::is_directory(folder_path))
    {
        cout<<folder<<" is not a folder"<<endl;
        return;
    }
    
    /* iterate the folder */
    cout<<"reading image  names "<<endl;
    bf::directory_iterator end_it;
    for( bf::directory_iterator file_iter(folder_path); file_iter != end_it; file_iter++ )
    {
        string pathname = file_iter->path().string();
        string basename = bf::basename(*file_iter);
        string extname = bf::extension(*file_iter);

        if( extname != ".jpg" && extname != ".png")
            continue;
        paths.push_back( pathname );
    }

    /*  read images  */
    cout<<"reading images ..."<<endl;
    for( unsigned int c=0;c<paths.size();c++)
    {
        cv::Mat temp = cv::imread( paths[c] );
        cv::resize( temp, temp, cv::Size(256,256), 0, 0);
        if(temp.empty())
            continue;
        mats.push_back( temp );
    }
}


int main(int argc, char** argv) 
{
    if( argc != 3)
    {
        cout<<"using it like this: ./extract_feature path_of_img_folder output.xml"<<endl;
        return -1;
    }

    /*  paths about the networks  */
    string net_model_file = "./imagenet_val.prototxt";
    string trained_model_file = "./bvlc_reference_caffenet.caffemodel";
    string mean_file_path = "./imagenet_mean.binaryproto";

    cnn_master cnnfeature;
    cnnfeature.load_model( net_model_file, mean_file_path, trained_model_file);
    std::cout<<"input should have width : "<<cnnfeature.get_input_width()<<std::endl;
    std::cout<<"input should have height : "<<cnnfeature.get_input_height()<<std::endl;
    std::cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<std::endl;
    std::cout<<"output dimension "<<cnnfeature.get_output_dimension("fc7")<<std::endl;

    vector<cv::Mat> imgs;
    vector<string> paths;
    read_data( string(argv[1]), imgs, paths);

    std::cout<<"------------ extracting features -----------"<<std::endl;

    cv::Mat cnn_features;
    cnnfeature.extract_blob( "fc7", imgs, cnn_features );

    /*  save the feature */
    cv::FileStorage fs;
    fs.open( string(argv[2]), cv::FileStorage::WRITE);
    fs<<"cnn_features"<<cnn_features;
    fs.release();

    return 0;
}

