/*
 * =====================================================================================
 *
 *       Filename:  cnn_feature_extractor.hpp
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

#ifndef CNN_FEATURE_EXTRACTOR_HPP
#define CNN_FEATURE_EXTRACTOR_HPP

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>

#include "boost/algorithm/string.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/net.hpp"

using caffe::Caffe;
using caffe::Net;
using std::string;

class cnn_feature_extractor
{
    
    public:
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  cnn_feature_extractor
     *  Description:  Load the model and definition, NOTE the path of the mean file
     *                .binaryproto is recorded in the network file. should be in the 
     *                same folder with network file
     * =====================================================================================
     */
    cnn_feature_extractor(
                            const string &network_file_path,    /* in : path of the network( .prototxt) file */
                            const string &mean_file_path,       /* in : path of the mean file(.binaryproto) file */
                            const string &model_file_path       /* in : path of the model( .caffemodel) file */
                         );

    
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  extract_cnn
     *  Description:  extract the cnn feature giving a image
     * =====================================================================================
     */
    bool extract_cnn( 
                        const std::vector<cv::Mat> &input_images ,       /*  in : input img */
                        cv::Mat &cnn_feature                         /*  out: output cnn feature */
                    );

    ~cnn_feature_extractor();

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
    int get_output_dimension() const
    {
        return m_output_dimension;
    }


    private:


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  is_model_ready
     *  Description:  check if the model is valid 
     * =====================================================================================
     */
    bool is_model_ready() const;

    /*  no copy */
    cnn_feature_extractor( const cnn_feature_extractor &rhs);
    cnn_feature_extractor& operator=(const cnn_feature_extractor &rhs);

    /*  Network model */
    Net<float> *m_network;

    /* Networks's input size */
    int m_input_width;
    int m_input_height;
    int m_input_channels;

    /* Network's output feature dimension */
    int m_output_dimension;

    /* which layer's output do we want */
    string m_feature_layer;

    /* Batch size of one Forward operation, this is limited by
     * the memory of the GPU, don't forget the network itself*/
    int m_batch_size;
};


#endif

