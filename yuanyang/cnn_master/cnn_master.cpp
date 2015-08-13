/*
 * =====================================================================================
 *
 *       Filename:  cnn_master.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YuanYang (), bengouawu@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#include <string>
#include <vector>
#include <iostream>
#include <queue>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/pointer_cast.hpp"

#include "google/protobuf/text_format.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/data_layers.hpp"

#include "cnn_master.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using boost::shared_ptr;
using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::priority_queue;

namespace bf = boost::filesystem;


/*  helper functions */
/* giving a array, return the topk biggest value and index */
bool get_topk_value_and_index( const vector<float> &input_array,
                                vector<int> &topk_index,
                                vector<float> &topk_value,
                                int k)
{
    if( input_array.empty() || k <=0)
        return false;
    topk_index.resize(k);
    topk_value.resize(k);

    /* make the priority_queue */ 
    priority_queue<std::pair<float,int> > q;
    for( unsigned int c=0;c<input_array.size();c++)
    {
        q.push( std::pair<float,int>( input_array[c],c ));
    }
    
    for( unsigned int i=0;i<k;i++)
    {
        topk_index[i] = q.top().second;
        topk_value[i] = q.top().first;
        q.pop();
    }

    return true;
}


cnn_master::cnn_master()
{
    /* do nothing */
    m_input_channels = 0;
    m_input_height = 0;
    m_input_width = 0;
    m_scale_factor = 1.0;
    m_subs_values.clear();
}


bool cnn_master::load_model( const string &deploy_file_path,    /* in : path of deploy file( *.prototxt) */
                             const string &mean_file_path,      /* in : path of image mean file( *.binaryproto) */
                             const string &model_file_path      /* in : path of model file (*.caffemodel) */
                            )
{
    /* check */
    if( !bf::exists( deploy_file_path) )
    {
        cout<<"error, file "<<deploy_file_path<<" does not exist , check again"<<endl;
        return false;
    }

    if( !bf::exists( model_file_path) )
    {
        cout<<"error, file "<<model_file_path<<" does not exist , check again"<<endl;
        return false;
    }
    
    /* by default use GPU 0 */
    Caffe::set_mode( Caffe::CPU);   /* if it dosen't have GPU, will use CPU instead */
    unsigned int device_id = 0;
    Caffe::SetDevice( device_id );

    /* --------------- loading---------------*/
    m_network = NULL;
    m_network = new Net<float>( deploy_file_path, caffe::TEST);
    if( !m_network )
    {
        cout<<"Can not load the model_file from "<<deploy_file_path<<endl;
        return false;
    }
    m_network->CopyTrainedLayersFrom(model_file_path);
    cout<<"Loading model file done "<<endl;

    /* set input image size */
    /* binary mean should be already loaded( it's alse written in the file deploy_file_path)
     * here we just want to get the input image's size( as mean image size ), since caffe lacks
     * the interface*/
    caffe::BlobProto blob_proto;
    
    if( bf::exists( mean_file_path) )
    {
        cout<<"Mean file exists, loading ..."<<endl;
        caffe::ReadProtoFromBinaryFileOrDie(mean_file_path.c_str(), &blob_proto);
        m_data_mean.FromProto(blob_proto);
    }
    
    /* now we extract the right input width, height, channel */
    if( !m_network->has_layer("data"))
    {
        cout<<"--> Error, network should be start with layer names data "<<endl;
        return false;
    }
    const shared_ptr<caffe::Layer<float> > data_layer = m_network->layer_by_name("data");
    if( 0!=strcmp("MemoryData",data_layer->type()) )
    {
        cout<<"--> Error, input layer should be of type MemoryData "<<endl;
        return false;
    }
    else
    {
        const shared_ptr<caffe::MemoryDataLayer<float> > me_data_layer = 
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(data_layer);

        m_input_channels = me_data_layer->channels();
        m_input_height = me_data_layer->height();
        m_input_width  = me_data_layer->width();

        // sometimes we will have a mean_file which is larger than the actual input size
        // (eg, we have a mean file 256x256, but we crop the 227x227 from the image during 
        // the training process. but during test phase, we need to crop the same size of mean_file
        // for substraction ~~
        if( !mean_file_path.empty())
        {
            if( m_input_channels != m_data_mean.channels())
            {
                cout<<"--> Error, input_channels != mean_file_channels, check your prototxt and meanfile "<<endl;
                return false;
            }
            if( m_input_width  > m_data_mean.width() || m_input_height > m_data_mean.height() )
            {
                cout<<"--> Errorm, make sure (input_width <= mean_file_width) && (input_height <= mean_file_height) "<<endl;
                return false;
            }
            if( m_input_width < m_data_mean.width() || m_input_height < m_data_mean.height() )
            {
                cout<<"Info: crop the mean_file to the same size as input defined in prorotxt"<<endl;
                m_input_width = m_data_mean.width();
                m_input_height = m_data_mean.height();
            }
        }
    }
    
    m_batch_size = 32;
    return true;
}


bool cnn_master::is_model_ready() const
{
    if( m_input_channels <=0 || m_input_height <=0 || m_input_width <=0 || !m_network)
        return false;
    return true;
}

int cnn_master::get_output_dimension( const string &blob_name) const   /*  in : the name of the blob */
{
    if( !is_model_ready())
    {
        cout<<"model is not ready "<<endl;
        return 0;
    }
    if( !m_network->has_blob(blob_name))
    {
        cout<<"blob "<<blob_name<<" does not exist "<<endl;
        return 0;
    }
    const shared_ptr< Blob<float> > output_blob = m_network->blob_by_name( blob_name);
    return output_blob->count()/output_blob->num();
}

cnn_master::~cnn_master()
{
    if(m_network)
        delete m_network;
}

bool cnn_master::extract_blob(  const string &blob_name,                     /* in : the name of the blob */
                                const std::vector<cv::Mat> &input_images,    /* in : input img */
                                cv::Mat &cnn_feature                         /* out: output cnn feature */
                                )
{

    /*  check check check */
    if(!is_model_ready())
    {
        cout<<"error, Model not ready yet"<<endl;
        return false;
    }

    if(input_images.empty())
    {
        cout<<"error, input_images is empty, return"<<endl;
        return false;
    }
    
    if( !m_network->has_blob(blob_name))
    {
        cout<<"error, Net does not have the blob "<<blob_name<<endl;
        return false;
    }

    /*  get the infos about every blob */
    const std::vector<string> blob_names = m_network->blob_names();
    const std::vector<string> layer_names = m_network->layer_names();

    /* lock and load */
    const shared_ptr<Blob<float> > input_blob = m_network->blob_by_name( blob_names[0] );

    /* make sure the image'size is the same with the input layer */
    for(unsigned int c=0;c<input_images.size();c++)
    {
        if( input_images[c].channels() != m_input_channels)
        {
            cout<<"error, inpur image should have "<<m_input_channels<<" channels, instead of "<<input_images[c].channels()<<endl;
            return false;
        }
        if( input_images[c].cols != m_input_width || input_images[c].rows != m_input_height)
        {
            cout<<"error, input image should have width "<<m_input_width<<" and height "<<m_input_height<<endl;
            return false;
        }
    }
    
    shared_ptr<caffe::MemoryDataLayer<float> > md_layer = boost::dynamic_pointer_cast <caffe::MemoryDataLayer<float> >(m_network->layers()[0]);
    if( !md_layer)
    {
        cout<<"error, The first layer is not momory data layer"<<endl;
        return false;
    }

    /*  prepare the output blob */
    const shared_ptr<Blob<float> > output_blob = m_network->blob_by_name( blob_name );

    /* prepare the output memory */
    int output_dimension = output_blob->count() / output_blob->num();
    cnn_feature = cv::Mat::zeros( input_images.size(), output_dimension, CV_32F);

    /* once a m_batch_size */
    std::vector<cv::Mat>::const_iterator start_iter = input_images.begin();
    std::vector<cv::Mat>::const_iterator end_iter   = input_images.begin() + 
                                                     ( m_batch_size > input_images.size()? input_images.size():m_batch_size);


	/* buffer for input blobs */

    while(1)
    {
        if( start_iter >= input_images.end())
            break;
        if( end_iter >= input_images.end()) /* adjust the input batch if it is too short */
            end_iter = input_images.end();

        std::vector<cv::Mat> mat_batch( start_iter, end_iter );

        /* extract the features and store */
        std::vector<int> no_use_labels( mat_batch.size(), 0 );
        md_layer->set_batch_size( mat_batch.size() );       /* sometimes it may less than m_batch_size */
        md_layer->AddMatVector( mat_batch, no_use_labels);

        /* fire the network */
        float no_use_loss=0;
        m_network->ForwardPrefilled(&no_use_loss);
        
        /* store the blob to Mat */
        const float *feature_blob_data = NULL;
        for(  int c=0;c<output_blob->num();c++)
        {
            feature_blob_data = output_blob->cpu_data() + output_blob->offset(c);
            memcpy( cnn_feature.ptr( start_iter - input_images.begin() + c ), feature_blob_data, sizeof(float)*output_dimension);
        }

        /* update the iterator */
        start_iter = end_iter;
        if(m_batch_size > input_images.end() - start_iter)
            end_iter = input_images.end();
        else
            end_iter = start_iter + m_batch_size;
    }
    return true;
}


bool cnn_master::get_class_label( const std::vector<cv::Mat> &input_images, /* in : input images */
                                  cv::Mat &output_label)                    /* out: output labels */
{
    /*  classification result stores in 'prob' blob */
    string target_blob = "prob";
    cv::Mat blob_feature;
    if(!extract_blob( target_blob, input_images, blob_feature ))
    {
        cout<<"error in extract_blob ..."<<endl;
        return false;
    }

    /* find the max activation*/
    output_label = cv::Mat::zeros( blob_feature.rows, 2, CV_32F);
    for( unsigned int c=0; c<blob_feature.rows; c++)
    {
        float max_conf = -1;
        int   max_index = 0;
        for( unsigned int i=0;i<blob_feature.cols;i++)
        {
            if( blob_feature.at<float>( c, i) > max_conf)
            {
                max_conf = blob_feature.at<float>( c, i);
                max_index = i;
            }
        }
        /* store results */
        output_label.at<float>(c, 0) = (float)max_index;
        output_label.at<float>(c, 1) = max_conf;
    }
    return true;
}

bool cnn_master::get_topk_label(    const std::vector<cv::Mat> &input_images,       /*  in: input image */
                                    std::vector<cv::Mat> &label_conf,               /* out:size: kx2, float <label, confidence> each line*/
                                    const int k)
{
    if( k < 1)
    {
        cout<<"make sure k >=1 "<<endl;
        return false;
    }

    /*  classification result stores in 'prob' blob */
    string target_blob = "prob";
    cv::Mat blob_feature;
    if(!extract_blob( target_blob, input_images, blob_feature ))
    {
        cout<<"error in extract_blob ..."<<endl;
        return false;
    }

    /* preallocate */
    label_conf.resize(input_images.size());
    for( unsigned int c=0;c<input_images.size();c++)
    {
        label_conf[c] = cv::Mat::zeros( k, 2, CV_32F); /*  <label , confidence> */
        /* extract the confidence line */
        const float *conf_line_ptr = (const float*)(blob_feature.ptr(c));
        vector<float> conf_line( conf_line_ptr, conf_line_ptr+blob_feature.cols );

        vector<float> topk_value;
        vector<int> topk_index;
        get_topk_value_and_index( conf_line, topk_index, topk_value, k);

        /* fill the output */
        for( unsigned int i=0;i<k;i++)
        {
            label_conf[c].at<float>(i,0)=topk_index[i];
            label_conf[c].at<float>(i,1)=topk_value[i];
        }
    }

    return true;
}


bool cnn_master::make_blob_from_mat( const std::vector<cv::Mat> &input_imgs,
									caffe::Blob<float> &output_blobs) const
{
	/* check the data */
	for( unsigned int i=0; i<input_imgs.size(); i++)
	{
		if( input_imgs[i].channels() != m_input_channels ||
				input_imgs[i].cols != m_input_width ||
				input_imgs[i].rows != m_input_height ||
				!input_imgs[i].isContinuous() ||
				( input_imgs[i].type()!=CV_8U && input_imgs[i].type() != CV_8UC3))
		{
			cout<<"Error--> Wrong input shape : "<<input_imgs[i].channels()<<" "<<input_imgs[i].rows<<" "<<input_imgs[i].cols<<
			"\t should have "<<m_input_channels<<" "<<m_input_height<<" "<<m_input_width<<
			"Or input image's memory is not continuous "<<
			"Or input image's format is not CV_8U or CV_8UC3 "<<endl;
			return false;
		}
	}

	/* reshape the blob, <num, channels, height, width> */
	output_blobs.Reshape( input_imgs.size(), m_input_channels, m_input_height, m_input_width);

	float for_subs_buffer = 0;
	/* convert rgb image form the opencv format <BGRBGRBGR> to caffe format<RRRGGGBBB> */
	for( unsigned int n=0; n<output_blobs.num(); n++)
		for( unsigned int c=0;c<output_blobs.channels();c++)
			for( unsigned int h=0;h<output_blobs.height();h++)
				for( unsigned int w=0;w<output_blobs.width();w++)
				{
					if( m_data_mean.count() != 0 ) /* use mean file */
						for_subs_buffer = m_data_mean.cpu_data()[ m_data_mean.offset(0,c,h,w)];
					else if( !m_subs_values.empty()) /* use mean value */
						for_subs_buffer  =  m_subs_values[c];
					else /* do nothing */
						for_subs_buffer = 0;

					output_blobs.mutable_cpu_data()[output_blobs.offset(n,c,h,w)]=
						m_scale_factor*((float)(unsigned char)input_imgs[n].data[h*m_input_width*m_input_channels+w*m_input_channels+c] 
							- for_subs_buffer);
				}

	return true;
}


bool cnn_master::crop_blob( caffe::Blob<float> &input_blob,
                    const unsigned int crop_width,
                    const unsigned int crop_height)
{
    if( input_blob.width() < crop_width || input_blob.height() < crop_height)
    {
        cout<<"--> Error , crop_size less than input size "<<endl;
        return false;
    }
    caffe::Blob<float> buffer_blob;
    buffer_blob.Reshape( 1, input_blob.channels(), crop_height, crop_width );

    const int offset_h = (input_blob.height() - crop_height)/2;
    const int offset_w = (input_blob.width() - crop_width)/2;

	for( unsigned int n=0; n<buffer_blob.num(); n++)
		for( unsigned int c=0;c<buffer_blob.channels();c++)
			for( unsigned int h=0;h<buffer_blob.height();h++)
				for( unsigned int w=0;w<buffer_blob.width();w++)
                {
					buffer_blob.mutable_cpu_data()[buffer_blob.offset(n,c,h,w)]= input_blob.cpu_data()[input_blob.offset(n,c,h+offset_h,w+offset_w)];
                }
    
    input_blob.CopyFrom( buffer_blob, true, true);
    return true;
}
