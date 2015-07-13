cmake_minimum_required(VERSION 2.8.8)
find_package(OpenCV REQUIRED)

######## work around in this machine #########
#set( Boost_INCLUDE_DIR /usr/local/include/ )
#set( Boost_LIBRARY_DIR /usr/local/lib)
#set(CMAKE_CXX_COMPILER /usr/local/bin/c++)
#set(CMAKE_C_COMPILER /usr/local/bin/gcc)
#############################################

find_package(Boost COMPONENTS system filesystem serialization REQUIRED)
find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_executable(test_master  main.cpp)
add_executable(test_license  main_test_license.cpp)
add_executable(test_car  main_test_car.cpp)
add_executable(test_just_one main_one.cpp)
add_executable(extract_feature main_extract_feature_demo.cpp)
add_executable(extract_feature_root_folder main_extract_feature_root_folder.cpp)

add_library( cnn_master cnn_master.hpp cnn_master.cpp)
add_library( misc misc.hpp misc.cpp)

target_link_libraries( cnn_master ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

#link for executable
target_link_libraries( test_master cnn_master ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( test_license cnn_master ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( test_car cnn_master ${Caffe_LIBRARIES} ${OpenCV_LIBS}
    ${Boost_LIBRARIES} misc)
target_link_libraries( test_just_one cnn_master ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( extract_feature cnn_master ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( extract_feature_root_folder cnn_master ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )

