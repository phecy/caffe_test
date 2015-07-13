// This script converts the images  to a leveldb format used by caffe
// Usage:
//    convert_mnist_data path_to_folder

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
#include <map>
#include <sstream>

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "boost/filesystem.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

namespace bf=boost::filesystem;

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;
using std::string;

/* crop image according to the rect, region will padded with boudary pixels if needed */
Mat cropImage( const Mat &input_image, const Rect &inrect )
{
	Mat outputImage;

	int top_pad=0;
	int bot_pad=0;
	int left_pad=0;
	int right_pad=0;
	
	if( inrect.x < 0)
		left_pad = std::abs(inrect.x);
	if( inrect.y < 0)
		top_pad = std::abs(inrect.y);
	if( inrect.x + inrect.width > input_image.cols )
		right_pad = inrect.x + inrect.width - input_image.cols;
	if( inrect.y+inrect.height >  input_image.rows)		
		bot_pad = inrect.y+inrect.height - input_image.rows;

	Rect imgRect( 0, 0, input_image.cols, input_image.rows );
	Rect target_region_with_pixel = imgRect & inrect;

	copyMakeBorder( input_image(target_region_with_pixel), outputImage, top_pad,bot_pad,left_pad,right_pad,BORDER_REPLICATE);
	return outputImage;
} 


Rect resizeToFixedRatio( const Rect &inRect,				/* in : input boudingbox informtion */
					     double w_h_ratio,					/* in : target ratio */
						 int flag)							/* in : respect to width  = 0 
																	respect to height > 0 */
{
	int center_x = inRect.x + inRect.width/2;
	int center_y = inRect.y + inRect.height/2;

	if( flag > 0)	/* respect to height*/
	{
		int target_width = int(inRect.height*w_h_ratio);
		int target_x = center_x - target_width/2;
		return Rect( target_x, inRect.y , target_width, inRect.height );
	}
	else			/* respect to width */
	{
		int target_height = int(inRect.width/ w_h_ratio);
		int target_y = center_y - target_height/2;
		return Rect( inRect.x, target_y, inRect.width, target_height );
	}
}


bool save_map( const map<int,int> &codebook,
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

bool load_map( map<int,int> &codebook,
                const string &path_of_file)
{
    ifstream ss( path_of_file.c_str());
    if( !ss.is_open())
    {
        cout<<"can not open file "<<path_of_file<<endl;
        return false;
    }
    boost::archive::text_iarchive iarch(ss);
    
}


int main(int argc, char** argv) 
{
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
    gflags::SetUsageMessage("This script converts the images dataset to\n"
        "the leveldb format used by Caffe to load data.\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    /*  check */
    if (argc != 3) 
    {
        gflags::ShowUsageWithFlagsRestrict(argv[0],"./convert_dataset filelist.txt leveldb_path");
        return -2;
    }
    else 
        google::InitGoogleLogging(argv[0]);
    
    /*  create the leveldb, open it */
    leveldb::DB *db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    unsigned long counter = 0;
    
    /*  open the leveldb file  */
    string output_db_path( argv[2] );
    LOG(INFO)<<"Opening leveldb "<<output_db_path;
    leveldb::Status status = leveldb::DB::Open( options, output_db_path.c_str(), &db );
    CHECK(status.ok()) << "Failed to open leveldb " << output_db_path<< ". Is it already existing?";
    
    /* also save the map from class_name to label ,
     * and map from label to class_name*/
    map<int, int> class_to_label;
    map<int, int> label_to_class;
    
    /* check if argv[1] is a folder */
    if( !bf::is_regular_file( string(argv[1])) )
    {
        cout<<string(argv[1])<<" is not a file !"<<endl;
        return -3;
    }
    /* iterate the folder */
    unsigned int label = 0;
    stringstream ss;

	int r, img_label, bbox_x, bbox_y, bbox_width, bbox_height;
	char img_path[30];
	FILE *fp = fopen( argv[1], "r");
	if(fp == NULL)
	{
		cout<<"can not open file "<<argv[1]<<endl;
		return -3;
	}

	/*  leveldb writer buffer */
	leveldb::WriteBatch* batch = NULL;
	batch =  new leveldb::WriteBatch();
	while(1)
	{
		r = fscanf(fp, "%s %d %d %d %d %d\n", img_path, &img_label, &bbox_x, &bbox_y, &bbox_width, &bbox_height);
		if( r == EOF )
			break;
		cv::Mat input_img = cv::imread( string(img_path) );
		if(input_img.empty())
		{
			cout<<"img empty ! "<<endl;
			return -5;
		}
		
		/*  read the adjust the image to the fixed size 256x256 */
		Rect adjusted_rect;
		if( bbox_height > bbox_width )
			adjusted_rect = resizeToFixedRatio(Rect( bbox_x, bbox_y, bbox_width, bbox_height), 1, 1);
		else
			adjusted_rect = resizeToFixedRatio(Rect( bbox_x, bbox_y, bbox_width, bbox_height), 1, 0);

		Mat crop_img = cropImage( input_img, adjusted_rect );
		resize( crop_img, crop_img, Size(256,256), 0, 0, INTER_AREA);

		/*decide the label*/
		if( class_to_label.count(img_label) ==0)
		{
			class_to_label[ img_label] = label;
			label_to_class[label] = img_label;
			label++;
		}
		
		/* write the Datum to leveldb */
		Datum datum;
		string value;
		const int kMaxKeyLength = 10;   /*  enough for this dataset */
		char key_cstr[kMaxKeyLength];
		
        /*  convert Mat to Datum using caffe util */
        CVMatToDatum( crop_img, &datum );
        datum.set_label(class_to_label[img_label]);

        //datum.set_channels(crop_img.channels());
        //datum.set_height(crop_img.rows);
        //datum.set_width(crop_img.cols);
        //datum.set_data(crop_img.data, crop_img.cols*crop_img.cols*crop_img.channels()); /* wrong, caffe's data format is num channel height wid */
        //datum.set_label( class_to_label[img_label] ); /*  remember to set the label */
        
        snprintf(key_cstr, kMaxKeyLength, "%08d", counter++);
        datum.SerializeToString(&value);

        batch->Put( key_cstr, value);
		if( counter % 1000 == 0)
		{
			db->Write(leveldb::WriteOptions(), batch);
			delete batch;
			batch =  new leveldb::WriteBatch();
		}
		
		cout<<"processing image "<<img_path<<" with label "<<class_to_label[img_label]<<endl;
		cout<<"car type is "<<class_to_label[img_label]<<endl;
		//rectangle( input_img, cv::Rect( bbox_x, bbox_y, bbox_width, bbox_height), cv::Scalar(255,0,0) );
		//imshow( "input", input_img);
		//imshow("adjust", crop_img);
        
        //cout<<"write image to "<<"./show_test/"+string(img_path)<<std::endl;
        //imwrite("./show_test/"+string(img_path), crop_img);
		//waitKey(0);
	}
	if( counter%1000 != 0 )
	{
		db->Write(leveldb::WriteOptions(), batch);
		delete batch;
		delete db;
	}
	
    cout<<"size of class_to_label is "<<class_to_label.size()<<endl;
    cout<<"size of label_to_class is "<<label_to_class.size()<<endl;

    save_map( class_to_label, "class_to_label_test.data");
    save_map( label_to_class, "label_to_class_test.data");

    return 0;
}
