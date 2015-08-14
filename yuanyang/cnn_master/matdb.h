/*
 * =====================================================================================
 *
 *       Filename:  matdb.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2015年08月14日 10时20分59秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YuanYang (), bengouawu@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

extern "C"
{
    #include "unqlite.h"
}

using namespace std;
using namespace cv;

/*!
 * @brief A wrapper around opencv' Mat and unqlite
 *        for quick storing and aquiring data
 */
class matdb{
    public:
        //! @brief empty constructor
        matdb();

        //! @brief shutdown the database and clean
        ~matdb();
        
        /*! 
         * @brief open the database db_name ,create a new one if needed
         * @param  db_name IN database'name
         * @param  read_only IN read only or read/write
         * @return is opened ?
         */
        bool open_db( const string &db_name, const bool read_only = false);

        /*!
         * @brief close the database
         */
        bool close_db();

        /*!
         * @brief store a Mat into database
         * @param key_value IN key value
         * @param mat_value IN Mat to store, empty Mat allowed
         * @return true if success
         *
         *  the original value will be overwrite , if the key_value has been used
         */
        bool store_mat( const string &key_value, const Mat &mat_value);

        /*!
         * @brief fetch a Mat using key value
         * @param key_value IN key value
         * @param mat_value OUT fetched Mat object, empty if failed
         * @return true if success
         */
        bool fetch_mat( const string &key_value, Mat &mat_value) const;

        /*!
         * @brief delete a mat record
         * @param key_value IN key
         * @return true if success
         */
        bool delete_mat( const string &key_value);

        /*!
         * @brief test if the key exist
         * @param key_value IN key_value
         * @return true if exists
         */
        bool has_key( const string &key_value) const;

        /*!
         * @brief check if the database is opened
         */
        inline bool is_opened() const
        {
            if( m_db) 
                return true;
            else
                return false;
        }

        unsigned long long get_total_num() const
        {
            return m_number_of_record;
        }

    private:
        /// @brief actual database handle
        unqlite *m_db;

        /// @brief cursor for database
        unqlite_kv_cursor *m_db_cur;

        /// @brief FileStorage buffer for fetch and store 
        mutable FileStorage m_fs;

        /// @brief since unqlite lacks the ability to return the total number of record ...
        unsigned long long m_number_of_record;

        /// @brief buffer for image reading
        mutable vector<char> m_buffer;
};

