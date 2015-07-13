/*
 * =====================================================================================
 *
 *       Filename:  yy_create_pair_data.cpp
 *
 *    Description:  create pairwise leveldb for caffe contrasitive loss training
 *
 *        Version:  1.0
 *        Created:  2015年07月08日 10时15分33秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YuanYang (), bengouawu@gmail.com
 *        Company: PCI
 *
 * =====================================================================================
 */

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>
#include <string>
#include <cstring>
#include <utility>
#include <vector>
#include <set>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


/* shuffle the data , useful for training */
DEFINE_bool(shuffle, true, "Randomly shuffle the order of images and their labels");
DEFINE_int32( width, 128, "input image's width ");
DEFINE_int32( height, 128, "input image's height ");

using namespace caffe;
using std::pair;
using boost::scoped_ptr;
using std::set;
using std::cout;
using std::endl;
using std::string;
using std::vector;

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

int main( int argc ,char** argv)
{
    
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Convert a set of images to the leveldb in pairwise format (image1, image2, sim)\n"
          "Usage:\n"
          "    yy_create_pair_data [FLAGS] ROOTFOLDER  LISTFILE DATASET\n");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4) 
    {
        LOG(WARNING)<<"input wrong ";
        gflags::ShowUsageWithFlagsRestrict(argv[0], "yy_create_pair_data");
        return 1;
    }
    

    /* 1 reading in the file  */
    vector<pair_info> train_file;
    read_in_file_list( string(argv[2]), train_file);

    /* shuffle ? */
    if( FLAGS_shuffle)
    {
        LOG(INFO)<<"Shuffling the data "<<endl;
        shuffle( train_file.begin(), train_file.end() );
    }
    LOG(INFO)<<"Total training sample "<<train_file.size();


    /*  create leveldb db */
    leveldb::Options options;
    options.block_size = 65536;
    options.write_buffer_size = 268435456*4;
    options.max_open_files = 1000;
    options.error_if_exists = true;    /* if database exists, just open it */
    options.create_if_missing = true;   /* create the dasebase if not exist */

    leveldb::DB *db_;
    string source(argv[3]);
    leveldb::Status status = leveldb::DB::Open(options, source, &db_);
    CHECK(status.ok()) << "Failed to open leveldb " << source<< std::endl << status.ToString();
    LOG(INFO) << "Opened leveldb " << source;
    scoped_ptr<leveldb::DB> db(db_);
    
    string root_folder( argv[1] );
    Datum datum;
    const int kMaxKeyLength = 15;
    char key_cstr[kMaxKeyLength];
    leveldb::WriteBatch *batch = new leveldb::WriteBatch();
    
    LOG(INFO)<<"Start processing data";
    LOG(INFO)<<"Input image's width "<<FLAGS_width;
    LOG(INFO)<<"Input image's height "<<FLAGS_height;

    uchar *image_buffer = new uchar[2*FLAGS_height*FLAGS_height];
    int image_size = FLAGS_height*FLAGS_width;

    int c_count = 0;
    for( unsigned long i=0; i<train_file.size();i++)
    {
        snprintf( key_cstr, kMaxKeyLength, "%013d", c_count);
        datum.set_channels(2);
        datum.set_height(FLAGS_height);
        datum.set_width(FLAGS_width);
        
        cv::Mat img1  = cv::imread( root_folder+train_file[i].image1, CV_LOAD_IMAGE_GRAYSCALE );
        cv::Mat img2  = cv::imread( root_folder+train_file[i].image2, CV_LOAD_IMAGE_GRAYSCALE );
        
        /* check the data size  */
        CHECK_EQ(img1.cols, FLAGS_width);
        CHECK_EQ(img2.cols, FLAGS_width);
        CHECK_EQ(img1.rows, FLAGS_height);
        CHECK_EQ(img2.rows, FLAGS_height);

        memcpy( image_buffer, img1.data, image_size);
        memcpy( image_buffer+image_size, img2.data, image_size);

        datum.set_data( image_buffer, 2*image_size);
        datum.set_label( train_file[i].label);

        /* debug */
        //cout<<"lable is "<<train_file[i].label<<endl;
        //cv::Mat debug_img = cv::Mat::zeros( FLAGS_height, FLAGS_width, CV_8UC1);
        //memcpy( debug_img.data, image_buffer, image_size);
        //cv::imshow("img1", debug_img);
        //cv::waitKey(0);
        //memcpy( debug_img.data, image_buffer+image_size, image_size);
        //cv::imshow("img2", debug_img);
        //cv::waitKey(0);

        string out_str;
        CHECK(datum.SerializeToString(&out_str));
        batch->Put( string(key_cstr),out_str);

        if (++c_count % 1000 == 0) 
        {
          // Commit db
            leveldb::Status status = db->Write(leveldb::WriteOptions(), batch);
            CHECK(status.ok()) << "Failed to write batch to leveldb "<< std::endl << status.ToString();
            delete batch;
            batch = new leveldb::WriteBatch();
            LOG(INFO) << "Processed " << c_count << " files.";
        }
    }
    /* store the last batch */
    if( c_count%1000 !=0)
    {
        leveldb::Status status = db->Write(leveldb::WriteOptions(), batch);
        CHECK(status.ok()) << "Failed to write batch to leveldb "<< std::endl << status.ToString();
        delete batch;
        LOG(INFO) << "Processed " << c_count << " files.";
    }
    return 0;
}
