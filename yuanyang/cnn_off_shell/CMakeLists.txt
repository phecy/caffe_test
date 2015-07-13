cmake_minimum_required(VERSION 2.8.8)
 
find_package(OpenCV REQUIRED)


# work around in this machine ........
set( Boost_INCLUDE_DIR /usr/local/include/ )
set( Boost_LIBRARY_DIR /usr/local/lib)
find_package(Boost COMPONENTS system filesystem REQUIRED)
set(CMAKE_CXX_COMPILER /usr/local/bin/c++)
set(CMAKE_C_COMPILER /usr/local/bin/gcc)


find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_library( cnn_master cnn_master.hpp cnn_master.cpp)

add_library( libsvm svm.h svm.cpp)
add_library( opencv_libsvm opencv_warpper_libsvm.h opencv_warpper_libsvm.cpp)

add_executable(extract_feature main.cpp)
add_executable( test_on_fly test_demo.cpp )

# link for library
target_link_libraries( opencv_libsvm libsvm)
target_link_libraries( cnn_master ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

#link for executable
target_link_libraries(extract_feature cnn_master ${Caffe_LIBRARIES}  ${OpenCV_LIBS} )
target_link_libraries( test_on_fly  cnn_master opencv_libsvm ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

