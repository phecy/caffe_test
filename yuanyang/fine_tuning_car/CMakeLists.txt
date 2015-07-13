cmake_minimum_required(VERSION 2.8.8)
find_package(OpenCV REQUIRED)

######## work around in this machine #########
set( Boost_INCLUDE_DIR /usr/local/include/ )
set( Boost_LIBRARY_DIR /usr/local/lib)
set(CMAKE_CXX_COMPILER /usr/local/bin/c++)
set(CMAKE_C_COMPILER /usr/local/bin/gcc)
#############################################

find_package(Boost COMPONENTS system filesystem serialization REQUIRED)
find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY


add_executable(fune_tuning_car   prepare_data.cpp)
add_executable( test_on_fly test_on_fly.cpp) 

add_library( cnn_feature_extractor cnn_feature_extractor.hpp cnn_feature_extractor.cpp)
target_link_libraries( cnn_feature_extractor ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

#link for executable
target_link_libraries( fune_tuning_car  ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries( test_on_fly cnn_feature_extractor ${Caffe_LIBRARIES}
    ${OpenCV_LIBS} ${Boost_LIBRARIES} )

