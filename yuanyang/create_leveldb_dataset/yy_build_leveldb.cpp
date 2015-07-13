/*
 * =====================================================================================
 *
 *       Filename:  yy_build_leveldb.cpp
 *
 *    Description:  create leveldb from a folder and the filelist
 *
 *        Version:  1.0
 *        Created:  2015年06月03日 23时47分04秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YuanYang (), bengouawu@gmail.com
 *        Company:  PCI
 *
 * =====================================================================================
 */
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>
#include <string>
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

/*  define some useful flag  */
DEFINE_bool(gray, false, "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, true, "Randomly shuffle the order of images and their labels");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, true, "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false, "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "", "Optional: What type should we encode the image as ('png','jpg',...).");

using namespace caffe;
using std::pair;
using boost::scoped_ptr;
using std::set;
using std::cout;
using std::endl;

int main(int argc, char** argv) 
{
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Convert a set of images to the leveldb\n"
          "format used as input for Caffe.\n"
          "Usage:\n"
          "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE \n"
          "The ImageNet dataset for the training demo is at\n"
          "    http://www.image-net.org/download-images\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4) 
    {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
      return 1;
    }

    const bool is_color = !FLAGS_gray;
    const bool check_size = FLAGS_check_size;
    const bool encoded = FLAGS_encoded;
    const string encode_type = FLAGS_encode_type;

    /*  reading in the whole file list, <file_path, label> */
    std::ifstream infile(argv[2]);
    std::vector<std::pair<std::string, int> > lines;
    std::string filename;
    int label;

    set<int> class_set;
    while (infile >> filename >> label)
    {
        lines.push_back(std::make_pair(filename, label));
        class_set.insert( label) ;
    }

    LOG(INFO)<<"Number of class is "<<class_set.size();

    /*  shullfe ? */
    if(FLAGS_shuffle)
    {
        LOG(INFO)<<"Shuffling Data";
        shuffle( lines.begin(), lines.end() );
    }
    LOG(INFO)<<"Total number of file :"<<lines.size();

    /*  encode ? */
    if (encode_type.size() && !encoded)
        LOG(INFO) << "encode_type specified, assuming encoded=true.";

    /*  resize ? */
    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width = std::max<int>(0, FLAGS_resize_width);

    /*  create leveldb db */
    leveldb::Options options;
    options.block_size = 65536;
    options.write_buffer_size = 268435456*4;
    options.max_open_files = 1000;
    options.error_if_exists = false;    /* if database exists, just open it */
    options.create_if_missing = true;   /* create the dasebase if not exist */
    leveldb::DB *db_;
    string source(argv[3]);
    leveldb::Status status = leveldb::DB::Open(options, source, &db_);
    CHECK(status.ok()) << "Failed to open leveldb " << source<< std::endl << status.ToString();
    LOG(INFO) << "Opened leveldb " << source;
    scoped_ptr<leveldb::DB> db(db_);
    
    /* extract all the keys */
    set<string> key_exists;
    leveldb::Iterator *it=db->NewIterator(leveldb::ReadOptions());
    for(it->SeekToFirst(); it->Valid(); it->Next())
        key_exists.insert(it->key().ToString());
    delete it; /*  always remember to delete the iterator */
    LOG(INFO)<<"Number of files in database -> "<<key_exists.size()<<endl;

    /* update&store the date to leveldb */
    std::string root_folder(argv[1]);
    Datum datum;
    int count = 0;
    const int kMaxKeyLength = 512;
    char key_cstr[kMaxKeyLength];
    int data_size = 0;
    bool data_size_initialized = false;
    leveldb::WriteBatch *batch = new leveldb::WriteBatch();

    LOG(INFO)<<"Start processing data ... ";
    for( unsigned int line_id=0;line_id<lines.size();line_id++)
    {
        int length = snprintf(key_cstr, kMaxKeyLength, "%s", lines[line_id].first.c_str());
        string key_mes( key_cstr, length);
        
        /* skip those already in the database */
        if( key_exists.count( key_mes) != 0 )
            continue;

        bool status;
        std::string enc = encode_type;
        if(encoded && !enc.size()) 
        {
            // Guess the encoding type from the file name
            string fn = lines[line_id].first;
            size_t p = fn.rfind('.');
            if ( p == fn.npos )
                LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
            enc = fn.substr(p);
            std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        status = ReadImageToDatum(root_folder + lines[line_id].first,lines[line_id].second, resize_height, resize_width, is_color, enc, &datum);
        if(status == false) continue;

        /* every image should have the identical size */
        if(check_size) 
        {
            if (!data_size_initialized) 
            {
                data_size = datum.channels() * datum.height() * datum.width();
                data_size_initialized = true;
            } 
            else
            {
                const std::string& data = datum.data();
                CHECK_EQ(data.size(), data_size) << "Incorrect data field size "<< data.size();
            }
        }
        // sequential

        // Put in db
        string out;
        CHECK(datum.SerializeToString(&out));
        batch->Put( string(key_cstr, length),out );

        if (++count % 1000 == 0) 
        {
          // Commit db
            leveldb::Status status = db->Write(leveldb::WriteOptions(), batch);
            CHECK(status.ok()) << "Failed to write batch to leveldb "<< std::endl << status.ToString();
            delete batch;
            batch = new leveldb::WriteBatch();
          
            LOG(INFO) << "Processed " << count << " files.";
        }
    }
    /* store the last batch */
    if( count%1000 !=0)
    {
        leveldb::Status status = db->Write(leveldb::WriteOptions(), batch);
        CHECK(status.ok()) << "Failed to write batch to leveldb "<< std::endl << status.ToString();
        delete batch;
        LOG(INFO) << "Processed " << count << " files.";
    }
    return 0;
}

