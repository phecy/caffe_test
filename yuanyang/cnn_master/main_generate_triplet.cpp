#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include <fstream>

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

/*  choose K from N  */
void comb(int N, int K, vector<vector<int> > &all_combs, int len=0)
{
    all_combs.clear();
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's

    // print integers and permute bitmask
    do {
        vector<int> each_comb;
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i])
                each_comb.push_back(i);
        }
        all_combs.push_back( each_comb );
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

    if( len > 0 && len < all_combs.size())
    {
        std::random_shuffle( all_combs.begin(), all_combs.end());
        all_combs.resize(len);
    }
}

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

void commit_buffer( vector<string> &anchor_buffer,
                    vector<string> &pos_buffer,
                    vector<string> &neg_buffer,
                    ofstream &ofs)
{
    cout<<"commit buffer .."<<endl;
    assert( anchor_buffer.size() == pos_buffer.size() &&
            anchor_buffer.size() == neg_buffer.size() &&
            "anchor pos neg size must match");
    for( unsigned int i=0;i<anchor_buffer.size();i++)
        ofs<<anchor_buffer[i]<<" "<<0<<endl;

    for( unsigned int i=0;i<pos_buffer.size();i++)
        ofs<<pos_buffer[i]<<" "<<1<<endl;

    for( unsigned int i=0;i<neg_buffer.size();i++)
        ofs<<neg_buffer[i]<<" "<<2<<endl;

    anchor_buffer.clear();
    pos_buffer.clear();
    neg_buffer.clear();
}

int main( int argc, char** argv)
{
    /* main parameters */
    int images_per_people_max = 50;
    double margin = 0.7;
    string feature_name = "output";
    int image_load_option = CV_LOAD_IMAGE_GRAYSCALE;
    int input_img_width = 144;
    int input_img_height = 144;
    int input_img_channel = 1;
    int num_neg_sample_per_pair = 2;

    ofstream output_file("namelist.txt");

    vector<string> anchor_buffer;
    vector<string> pos_buffer;
    vector<string> neg_buffer;

    srand(time(NULL));

    string image_root_folder = string( argv[1]);   
    if( !bf::is_directory( image_root_folder))
    {
        cerr<<image_root_folder<<" is not a folder "<<endl;
        return -1;
    }

    string model_deploy_file = "face_deploy.prototxt";   
    string model_binary_file = "small_max_out.caffemodel";
    string model_mean_file   = "";

    /* set cnn model */
    cout<<"Loading DCNN model "<<endl;
    cnn_master cnnfeature;

    if(!cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file))
    {
        cerr<<"Load Model failed "<<endl;
        return -2;
    }

    cnnfeature.set_input_width(input_img_width);
    cnnfeature.set_input_height(input_img_height);
    cnnfeature.set_input_channel(input_img_channel);
    
    /* add folder path .. */
    cout<<"Adding folders "<<endl;
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
        cout<<"processing folder : "<<folder_path[i]<<endl;   
        vector<string> image_path;
        get_all_image_files( folder_path[i], image_path);
        
        vector<vector<int> > anchor_pos_pair;
        comb( image_path.size(), 2, anchor_pos_pair, images_per_people_max);
        
        /* choose a nagetive sample for each anchor_positive pair*/
        for( unsigned int index=0;index<anchor_pos_pair.size();index++)
        {
            Mat img1 = imread( image_path[anchor_pos_pair[index][0]], image_load_option);
            Mat img2 = imread( image_path[anchor_pos_pair[index][1]], image_load_option);

            vector<Mat> ap_imgs; 
            ap_imgs.push_back(img1); 
            ap_imgs.push_back(img2);

            Mat ap_mat;
            cnnfeature.extract_blob( feature_name, ap_imgs, ap_mat);
            double ap_dis = cv::norm( ap_mat.row(0), ap_mat.row(1));

            int got_matched = 0;
            int negative_people_index = i;

            /* if the distance is, pick random max_try people for comparation*/
            int max_try = 25;
            for( unsigned int iter=0; iter<max_try && got_matched < num_neg_sample_per_pair; iter++ )
            {
                /* choose another person first */
                do{
                    negative_people_index = rand()%folder_path.size();
                }
                while( negative_people_index == i );

                vector<string> neg_imgs;
                get_all_image_files( folder_path[negative_people_index], neg_imgs);

                /* compare 2 images at most  */
                for( unsigned int neg_index = 0; neg_index < neg_imgs.size() && neg_index < 2; neg_index++)
                {
                    Mat neg_tmp_img = imread( neg_imgs[neg_index], image_load_option);
                    vector<Mat> neg_tmp;
                    neg_tmp.push_back( neg_tmp_img);
                    Mat neg_fea;
                    cnnfeature.extract_blob( feature_name, neg_tmp, neg_fea);

                    double an_dis = cv::norm( ap_mat.row(0), neg_fea.row(0));
                    if( an_dis > ap_dis && an_dis < ap_dis + margin )
                    {
                        /* add to buffer  */
                        anchor_buffer.push_back(image_path[anchor_pos_pair[index][0]]);
                        pos_buffer.push_back(image_path[anchor_pos_pair[index][1]]);
                        neg_buffer.push_back( neg_imgs[neg_index]);

                        //cout<<"matched "<<endl;
                        //cout<<"reading negative image "<<neg_imgs[neg_index]<<endl;
                        //cout<<"a-p dis is "<<ap_dis<<endl;
                        //cout<<"a-n dis is "<<an_dis<<endl;
                        //imshow("anchor", img1);
                        //imshow("pos", img2);
                        //imshow("neg", neg_tmp_img);
                        //waitKey(0);

                        if( anchor_buffer.size() >= 60)
                            commit_buffer(anchor_buffer, pos_buffer, neg_buffer, output_file);

                        got_matched++;
                        if( got_matched >=num_neg_sample_per_pair )
                            break;
                    }
                }
            }

        }
    }

    return 0;
}
