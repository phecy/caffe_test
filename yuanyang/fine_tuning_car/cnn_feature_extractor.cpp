/*
 * =====================================================================================
 *
 *       Filename:  cnn_feature_extractor.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2015年04月22日 16时52分02秒
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

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"

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

#include "cnn_feature_extractor.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
using std::vector;

namespace bf = boost::filesystem;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cnn_feature_extractor
 *  Description:  Load the model and definition, NOTE the path of the mean file
 *                .binaryproto is recorded in the network file
 * =====================================================================================
 */
cnn_feature_extractor::cnn_feature_extractor(
                                                const string &network_file_path,    /* in : path of the network( .prototxt) file */
                                                const string &mean_file_path,       /* in : path of the mean file(.binaryproto) file */
                                                const string &model_file_path       /* in : path of the model( .caffemodel) file */
                                             )
{
    /* check if the file exists */
    bf::path file_1( network_file_path);
    bf::path file_2( mean_file_path);
    bf::path file_3( model_file_path);
    if( !bf::exists(file_1) || !bf::exists(file_2) || !bf::exists(file_3))
        return;

    /* set log and choose GPU device */
    unsigned int device_id = 0;     /*  use device 0 by default */
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);   

    m_network = NULL;
    m_network = new Net<float>( network_file_path, caffe::TEST);
    if( !m_network )
    {
        std::cout<<"Can not load the network_file from "<<network_file_path<<std::endl;
        return ;
    }
    m_network->CopyTrainedLayersFrom(model_file_path);
    std::cout<<"copy Trained layers done "<<std::endl;
    /* ------------- set input image size ------------ */
    /* binary mean should be already loaded( it's alse written in the file network_file_path)
     * here we just want to get the input image's size( as mean image size ), since caffe lacks
     * the interface*/
    caffe::BlobProto blob_proto;
    caffe::Blob<float> data_mean;
    caffe::ReadProtoFromBinaryFileOrDie(mean_file_path.c_str(), &blob_proto);
    data_mean.FromProto(blob_proto);
    
    m_input_height = data_mean.height();
    m_input_width = data_mean.width();
    m_input_channels = data_mean.channels();

    /* batch size means network processes m_batch_size images one Forward */
    m_batch_size = 128;     
    std::cout<<"read mean file done "<<std::endl;
    /* -------------set output feature's dimension-------------- */
    /*  infos about the feature output layer */
    m_feature_layer = "fc7";
    if( !m_network->has_blob(m_feature_layer))
    {
        std::cout<<"ERROR:sorry , net does not have a blob :"<<m_feature_layer<<std::endl;
        return ;
    }
    const shared_ptr<Blob<float> > feature_blob = m_network->blob_by_name(m_feature_layer);
    int batch_size = feature_blob->num();
    int dim_feature = feature_blob->count() / batch_size;
    m_output_dimension = dim_feature;
    std::cout<<"set feature blob done "<<std::endl;
}


bool cnn_feature_extractor::is_model_ready() const
{
    if( m_input_channels <=0 || m_input_height <=0 || m_input_width <=0 || m_output_dimension <=0 || !m_network)
        return false;
    return true;
}



cnn_feature_extractor::~cnn_feature_extractor()
{
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  extract_cnn
 *  Description:  extract the cnn feature giving a image
 * =====================================================================================
 */
bool cnn_feature_extractor::extract_cnn( 
                                            const vector<cv::Mat> &input_images,    /*  in : input img */
                                            cv::Mat &cnn_feature                    /*  out: output cnn feature */
                                        )
{
    if(!is_model_ready())
    {
        std::cout<<"error, Model not ready yet"<<std::endl;
        return false;
    }
    /*  check the data first */
    if(input_images.empty())
    {
        std::cout<<"error, input_images is empty, return"<<std::endl;
        return false;
    }
    
    /*  get the infos about every blob */
    const std::vector<string> blob_names = m_network->blob_names();
    const std::vector<string> layer_names = m_network->layer_names();
    
    /* show the blob and layer */
    //for(int c=0;c<blob_names.size();c++)
    //{
    //    std::cout<<"blob "<<c<<" is "<<blob_names[c]<<std::endl;
    //}

    //for(int c=0;c<layer_names.size();c++)
    //{
    //    std::cout<<"layer "<<c<<" is "<<layer_names[c]<<std::endl;
    //}

    const shared_ptr<Blob<float> > input_blob = m_network->blob_by_name( blob_names[0] );

    /* make sure the image'size is the same with the input layer
     * otherwise resize it*/
    for(unsigned int c=0;c<input_images.size();c++)
    {
        if( input_images[c].channels() != m_input_channels)
        {
            std::cout<<"error, inpur image should have "<<m_input_channels<<" channels, instead of "<<input_images[c].channels()<<std::endl;
            return false;
        }
        if( input_images[c].cols != m_input_width || input_images[c].rows != m_input_height)
        {
            std::cout<<"error, input image should have width "<<m_input_width<<" and height "<<m_input_height<<std::endl;
            return false;
        }
    }
    

    /* lock and load */
    shared_ptr<caffe::MemoryDataLayer<float> > md_layer = boost::dynamic_pointer_cast <caffe::MemoryDataLayer<float> >(m_network->layers()[0]);
    if( !md_layer)
    {
        std::cout<<"error, The first layer is not momory data layer"<<std::endl;
        return false;
    }

    /* prepare the output memory */
    cnn_feature = cv::Mat::zeros( input_images.size(), m_output_dimension, CV_32F);

    /* once a m_batch_size */
    std::vector<cv::Mat>::const_iterator start_iter = input_images.begin();
    std::vector<cv::Mat>::const_iterator end_iter   = input_images.begin() + 
                                                     ( m_batch_size > input_images.size()? input_images.size():m_batch_size);
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
        
        /*  infos about the feature output layer */
        //const shared_ptr<Blob<float> > feature_blob = m_network->blob_by_name(m_feature_layer);
        //int batch_size = feature_blob->num();
        

        /* get prob */
        shared_ptr<Blob<float> > probs = m_network->blob_by_name("prob");
        std::cout<<"prob.count() is "<<probs->count()<<std::endl;
        std::cout<<"probs size is "<<probs->num()<<std::endl;

        for( int c=0;c<probs->num();c++)
        {
            float maxval= 0;
            int   maxinx= 0;
            for (int i = 0; i < probs->count(); i++)
            {
                float val= (probs->cpu_data()+ probs->offset(c))[i];
                if (val> maxval)
                {
                    maxval= val;
                    maxinx= i;
                }
                //std::cout << "[" << i << "]" << val<< "\n";
            }
            std::cout << "Max value = " << maxval<< ", Max index = " << maxinx<< "\n";
        }


        /* extract the save the cnn features */
        //const float *feature_blob_data = NULL;
        //for(unsigned int c=0;c<batch_size;c++)
        //{
        //    feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(c);
        //    /* copy data to the output feature Mat*/
        //    memcpy( cnn_feature.ptr( start_iter - input_images.begin() + c ), feature_blob_data, sizeof(float)*m_output_dimension);
        //}

        /* update the iterator */
        start_iter = end_iter;
        if(m_batch_size > input_images.end() - start_iter)
            end_iter = input_images.end();
        else
            end_iter = start_iter + m_batch_size;
    }
    return true;
}
