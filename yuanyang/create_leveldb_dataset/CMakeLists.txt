cmake_minimum_required(VERSION 2.8.8)
find_package(OpenCV REQUIRED)

######## work around in this machine #########
#set( Boost_INCLUDE_DIR /usr/local/include/ )
#set( Boost_LIBRARY_DIR /usr/local/lib)
#set(CMAKE_CXX_COMPILER /usr/local/bin/c++)
#set(CMAKE_C_COMPILER /usr/local/bin/gcc)
##############################################

find_package(Boost COMPONENTS system filesystem serialization REQUIRED)
find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_executable(yy_build_leveldb  yy_build_leveldb.cpp)
add_executable(yy_split_leveldb yy_split_leveldb.cpp)
add_executable(yy_show_leveldb yy_show_leveldb.cpp)
add_executable(yy_create_pair_dataset yy_create_pair_data.cpp)


#link for executable
target_link_libraries( yy_build_leveldb ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( yy_split_leveldb ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( yy_show_leveldb ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( yy_create_pair_dataset ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )

