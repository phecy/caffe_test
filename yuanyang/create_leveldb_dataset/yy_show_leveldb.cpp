/*
 * =====================================================================================
 *
 *       Filename:  yy_split_leveldb.cpp
 *
 *    Description:  split the data into train & test 
 *
 *        Version:  1.0
 *        Created:  2015年06月04日 18时08分38秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YuanYang (), bengouawu@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "opencv2/highgui/highgui.hpp"

using namespace caffe;
using std::pair;
using boost::scoped_ptr;
using std::map;
using std::vector;
using std::string;
using std::cout;
using std::endl;

DEFINE_bool( show_img ,false, "show the image or not");
DEFINE_bool( output, true, "output the file name and label ");

cv::Mat DatumToCvMat( const Datum& datum )
{
    int img_type;
    switch( datum.channels() )
    {
        case 1:
            img_type = CV_8UC1;
            break;
        case 3:
            img_type = CV_8UC3;
            break;
        default:
            CHECK(false) << "Invalid number of channels.";
            break;
    }
 
    cv::Mat mat( datum.height(), datum.width(), img_type );
    int datum_channels = datum.channels();
    int datum_height = datum.height();
    int datum_width = datum.width();

    const string img_data = datum.data();
    for (int h = 0; h < datum_height; ++h) 
    {
        uchar* ptr = mat.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < datum_width; ++w) 
        {
            for (int c = 0; c < datum_channels; ++c) 
            {
                int datum_index = (c * datum_height + h) * datum_width + w;

                /*  for byte data */
                ptr[img_index++] = static_cast<uchar>( img_data[datum_index] );

                /*  for float data */
                //float datum_float_val = datum.float_data(datum_index);
                //if( datum_float_val >= 255.0 )
                //{
                //    ptr[img_index++] = 255;
                //}
                //else if ( datum_float_val <= 0.0 )
                //{
                //    ptr[img_index++] = 0;
                //}
                //else
                //{
                //    ptr[img_index++] = static_cast<uchar>( lrint( datum_float_val) );
                //}
            }
        }
    }
    return mat;
} 

int main(int argc, char** argv) 
{

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    cout<<"argc is "<<argc<<endl;
    gflags::SetUsageMessage("Usage: ./yy_show_leveldb source_leveldb  [show]");
    gflags::ParseCommandLineFlags( &argc, &argv, true);

    cout<<"argc is "<<argc<<endl;

    if( argc !=2 )
    {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "./yy_show_leveldb ");
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    const bool show_or_not = FLAGS_show_img;
    const bool output_or_not = FLAGS_output;
    cout<<"show or not is "<<show_or_not<<endl;

    /*  open leveldb db */
    leveldb::Options options;
    options.block_size = 65536;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;
    options.error_if_exists = false;    /* if database exists, just open it */
    options.create_if_missing = false;  
    leveldb::DB *db_;

    string source(argv[1]);
    leveldb::Status status = leveldb::DB::Open(options, source, &db_);
    CHECK(status.ok()) << "Failed to open leveldb " << source<< std::endl << status.ToString();
    LOG(INFO) << "Opened leveldb " << source;
    scoped_ptr<leveldb::DB> db(db_);
    
    /* iterate the database, store the label and correspoding */
    Datum datum;
    map<int, vector<string> > total_list;
    int sample_label;
    string key_str;

    fstream writer("infos.txt");

    /* scanning the dataset */
    LOG(INFO)<<"Scan the datebase ... ";
    leveldb::Iterator *it=db->NewIterator(leveldb::ReadOptions());
    for(it->SeekToFirst(); it->Valid(); it->Next())
    {
        datum.ParseFromString( it->value().ToString());
        sample_label = datum.label();
        key_str = it->key().ToString();
        if(output_or_not)
            writer<<key_str<<" "<<sample_label<<endl;
        if(show_or_not)
        {
            cout<<"img size "<<DatumToCvMat(datum).cols<<" "<<DatumToCvMat(datum).rows<<" "<<DatumToCvMat(datum).channels()<<endl;
            cv::imshow("show", DatumToCvMat(datum));
            cv::waitKey(0);
        }
    }
    delete it;

    return 0;
}

