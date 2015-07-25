/*
 * =====================================================================================
 *
 *       Filename:  cnn_master.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *       Compiler:  gcc
 *
 *         Author:  YuanYang (), bengouawu@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef CNN_MASTER_HPP
#define CNN_MASTER_HPP

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"

using caffe::Caffe;
using caffe::Net;
using std::string;

class cnn_master
{
    
    public:
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  cnn_feature_extractor
     *  Description:  empty constructor
     * =====================================================================================
     */
    cnn_master();

    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  load_model
     *  Description:  load the model files, return false if fails
     * =====================================================================================
     */
    bool load_model( const string &deploy_file_path,    /* in : path of deploy file( *.prototxt) */
                     const string &mean_file_path,      /* in : path of image mean file( *.binaryproto) */
                     const string &model_file_path      /* in : path of model file (*.caffemodel) */
                     );

    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  extract_blob
     *  Description:  extract the blob for giving images and blob name
     *                to get the classification result, set blob_name to "prob"
     *                to get the cnn feature , set the blob_name to the last fc layer
     * =====================================================================================
     */
    bool extract_blob( const string &blob_name,                     /* in : the name of the blob */
                       const std::vector<cv::Mat> &input_images,    /* in : input img */
                       cv::Mat &cnn_feature                         /* out: output cnn feature */
                       );

    
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  get_class_label
     *  Description:  do the classification, warpper of extract_blob
     *                output_label size : number_of_image x 2, each row
     *                consists of < label, confidence>
     * =====================================================================================
     */
    bool get_class_label( const std::vector<cv::Mat> &input_images, /* in : input images */
                          cv::Mat &output_label);                   /* out: output labels */

    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  get_output_dimension
     *  Description:  return the dimension of the giving blob
     * =====================================================================================
     */
    int get_output_dimension( const string &blob_name) const;   /*  in : the name of the blob */


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  get_topk_label
     *  Description:  return the top k results
     * =====================================================================================
     */
    bool get_topk_label( const std::vector<cv::Mat> &input_image,       /* in : input image */
                         std::vector<cv::Mat> &lableconfs,              /* out: size: kx2, float <label, confidence> each line*/
                         const int k=1);                                /* in : top k's k*/


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  set_transform_para
     *  Description:  x_out = ( x_ori - substract_values_of_this_channel )*scale_factor
     *                can not set both substract_values and mean_file in load_model functions()
     * =====================================================================================
     */
    bool set_transform_para( const float &scale_factor,             /* in : scale factor */
                             std::vector<float> &substract_values); /* in : substract those values for each channel */


    /*  Get the infos about the network's input and output */
    int get_input_width() const
    {
        return m_input_width;
    };
    int get_input_height() const
    {
        return m_input_height;
    };
    int get_input_channels() const
    {
        return m_input_channels;
    }

    ~cnn_master();

    private:


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  is_model_ready
     *  Description:  check if the model is valid 
     * =====================================================================================
     */
    bool is_model_ready() const;


	/* 
	 * ===  FUNCTION  ======================================================================
	 *         Name:  make_blob_form_mat
	 *  Description:  convert the mats into blobs, transform them if specified
	 * =====================================================================================
	 */
	bool make_blob_from_mat( const std::vector<cv::Mat> &input_imgs,
							 caffe::Blob<float> &output_blobs) const;

    /*  no copy */
    cnn_master( const cnn_master &rhs);
    cnn_master& operator=(const cnn_master &rhs);

    /*  Network model */
    Net<float> *m_network;

    /* Networks's input size */
    unsigned int m_input_width;
    unsigned int m_input_height;
    unsigned int m_input_channels;

    /* Batch size of one Forward operation, this is limited by
     * the memory of the GPU, don't forget the network itself*/
    unsigned int m_batch_size; /* set it to 8,16,32,64, 128 or 256 */

    /* scale factor */
    float m_scale_factor;

    /* mean values */
    std::vector<float> m_subs_values;
	
	/* or mean image */
    caffe::Blob<float> m_data_mean;
};


#endif

