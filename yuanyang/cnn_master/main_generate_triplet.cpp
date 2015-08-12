#include <iostream>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/lambda/bind.hpp"
#include "boost/algorithm/string.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cnn_master.hpp"

using namespace std;
using namespace cv;

namespace bf=boost::filesystem;
namespace bl=boost::lambda;

void get_all_image_files( const string &folder, vector<string> &image_path)
{
    image_path.clear();
    if( !bf::is_directory(folder))
        return;

    bf::directory_iterator end_it;
    for( bf::directory_iterator f_iter( folder); f_iter!=end_it;f_iter++)
    {
        if( !bf::is_regular_file(*f_iter))
            continue;

        string extname = bf::extension( *f_iter );
        boost::algorithm::to_lower(extname);

        if( extname != ".jpg" && extname != ".bmp" && extname != ".png")
            continue;
        image_path.push_back( f_iter->path().string());
    }
}

int main( int argc, char** argv)
{
    string image_root_folder = "";   
    if( bf::is_directory( image_root_folder))
    {
        cerr<<image_root_folder<<" is not a folder "<<endl;
        return -1;
    }

    string model_deploy_file = "sensetime.prototxt";   
    string model_binary_file = "sensetime.model";
    string model_mean_file   = "";

    /* set cnn model */
    cnn_master cnnfeature;
    cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);

    cnnfeature.set_input_width(144);
    cnnfeature.set_input_height(144);
    cnnfeature.set_input_channel(1);
    
    /* add folder path .. */
    vector<string> folder_path;
    bf::directory_iterator end_it;
    for( bf::directory_iterator folder_iter(image_root_folder); folder_iter!=end_it;folder_iter++)
    {
        if( !bf::is_directory(*folder_iter))
        {
            cout<<"skip non-folder"<<endl;
            continue;
        }
        folder_path.push_back(folder_iter->path().string());
    }
    

    for( unsigned int i=0; i<folder_path.size();i++)
    {
        
    }

    return 0;
}
