#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <sstream>

#include "boost/filesystem.hpp"
#include "boost/serialization/serialization.hpp"
#include "boost/serialization/map.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cnn_master.hpp"

using namespace std;
using namespace cv;
namespace bf=boost::filesystem;

/* 我使用的参数是 resize_to_fixed(input_img, Size(20,44), Size(22,46))*/
Mat resize_to_fixed( const Mat& input_image,
					const Size &inner_size,
					const Size &out_size)
{
	if( input_image.empty())
	{
		cout<<"########### error , img empty ################"<<endl;
		return Mat();
	}

	/*  resize image to 40x40 */
	Mat resize_img;
	cv::resize( input_image, resize_img, inner_size, 0, 0,INTER_AREA);

	Mat big_img = 255*Mat::ones( out_size, resize_img.type());

	copyMakeBorder( resize_img, big_img, 
		(big_img.rows-resize_img.rows)/2, 
		(big_img.rows-resize_img.rows)/2,
		(big_img.cols-resize_img.cols)/2,
		(big_img.cols-resize_img.cols)/2,
		BORDER_CONSTANT,
		Scalar(127,127,127));

	resize(big_img, big_img, out_size, 0, 0);

	return big_img;
}

template<class T> bool save_map( T &codebook,
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

template< class T>
bool load_map( T &codebook,
			  const string &path_of_file)
{
	ifstream ss( path_of_file.c_str());
	if( !ss.is_open())
	{
		cout<<"can not open file "<<path_of_file<<endl;
		return false;
	}
	boost::archive::text_iarchive iarch(ss);
	iarch>>codebook;
	return true;
}

int main( int argc, char **argv )
{	
	/*  映射文件表 */
	//map<int,string> label_to_class;
	//map<string,int> class_to_label;
	//if(!load_map( label_to_class, "label_to_class_map.data") ||
	//	!load_map( class_to_label, "class_to_label_map.data"))
	//{
	//	cout<<"can not load map data"<<endl;
	//	return -3;
	//}

	/*  模型路径 */
	//string model_deploy_file = "../deploy_license_2.prototxt";   
	//string model_mean_file   = "../license_char_mean_2.binaryproto";
	//string model_binary_file = "../license_char_model_2.caffemodel";
    
    
	string model_deploy_file = string(argv[1]);   
	string model_binary_file = string(argv[2]);
	string model_mean_file   = "";

	cnn_master cnnfeature;
	cnnfeature.load_model( model_deploy_file, model_mean_file, model_binary_file);

    cnnfeature.set_input_width(256);
    cnnfeature.set_input_height(256);
    cnnfeature.set_input_channel(3);

	cout<<"input should have width : "<<cnnfeature.get_input_width()<<endl;
	cout<<"input should have height : "<<cnnfeature.get_input_height()<<endl;
	cout<<"input should have channels : "<<cnnfeature.get_input_channels()<<endl;


	Mat input_image = imread( argv[3] );
    Mat input_img2 = imread( argv[4] );
    cv::resize( input_image, input_image, Size(256,256), 0, 0);
    cv::resize( input_img2, input_img2, Size(256,256), 0, 0);

    //cv::normalize( input_image, input_image, 0, 255, CV_MINMAX);
	vector<Mat> imagelists;
	imagelists.push_back( input_image);
	imagelists.push_back( input_img2);
    
	//imagelists.push_back( input_image);
	//imagelists.push_back( input_image);
	//imagelists.push_back( input_image);
	//imagelists.push_back( input_image);
	//imagelists.push_back( input_image);
	//imagelists.push_back( input_image);
	//imagelists.push_back( input_image);
    
    /* show feature*/
    Mat show_feature;
    cnnfeature.extract_blob("output", imagelists, show_feature);
    cout<<"feature dim "<<cnnfeature.get_output_dimension("output")<<endl;
    cout<<"fc6 :\n"<<show_feature<<endl;
    

	/* 1 每个图片返回一个类标 */
	//Mat predict_label;
	//cnnfeature.get_class_label( imagelists, predict_label);
	//cout<<"predict to label "<<predict_label.at<float>(0,0)<<" with confidence "<<predict_label.at<float>(0,1)<<endl;
	////cout<<"convert the label to string -> "<<label_to_class[int(predict_label.at<float>(0,0))]<<endl;

	///* 2 每个图片返回top_N的类标和对应的置信度 */
	//vector<Mat> labelconfs;
	//cnnfeature.get_topk_label( imagelists, labelconfs, 3 );
	//for( unsigned int c=0;c<labelconfs.size();c++)
	//{
	//	cout<<"label&conf is "<<labelconfs[c]<<endl;
	//}

	//imshow("test", input_image);
	//waitKey(0);
	
	return 0;
}
