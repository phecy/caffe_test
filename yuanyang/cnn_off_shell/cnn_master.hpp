/*
 * =====================================================================================
 *
 *       Filename:  cnn_master.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2015年04月22日 16时52分11秒
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

    /*  no copy */
    cnn_master( const cnn_master &rhs);
    cnn_master& operator=(const cnn_master &rhs);

    /*  Network model */
    Net<float> *m_network;

    /* Networks's input size */
    int m_input_width;
    int m_input_height;
    int m_input_channels;

    /* Batch size of one Forward operation, this is limited by
     * the memory of the GPU, don't forget the network itself*/
    int m_batch_size; /* set it to 64, 128 or 256 */
};


#endif

