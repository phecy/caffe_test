/*
 * =====================================================================================
 *
 *       Filename:  matdb.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2015年08月14日 10时21分05秒
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
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "matdb.h"

extern "C"
{
    #include "unqlite.h"
}

using namespace std;
using namespace cv;

matdb::matdb()
{
    m_db = NULL;
    m_db_cur = NULL;
    m_number_of_record = 0;
}

matdb::~matdb()
{
    if(m_db)
    {
        close_db();
    }
}

bool matdb::close_db()
{
    /// release the cursor 
    if( m_db_cur )
    {
        if( UNQLITE_OK != unqlite_kv_cursor_release( m_db, m_db_cur))
        {
            cerr<<"Close database failed when release the cursor"<<endl;
            return false;
        }
        m_db_cur = NULL;
    }

    /// release the QFontDatabase
    if( m_db)
    {
        int rc = unqlite_close(m_db);
        if( rc != UNQLITE_OK )
        {
            cerr<<"data_base close failed , may loss data .. "<<endl;
            return false;
        }
        m_db = NULL;
    }
    cout<<"database closed "<<endl;
    return true;
}

bool matdb::open_db( const string &db_name,const bool read_only)
{
    int rc=0;
    if( read_only )
    {
        rc = unqlite_open( &m_db, db_name.c_str(), UNQLITE_OPEN_READONLY);
        cout<<"READONLY model "<<endl;
    }
    else
    {
        rc = unqlite_open( &m_db, db_name.c_str(), UNQLITE_OPEN_CREATE);
        cout<<"READ&WRITE model"<<endl;
    }

    /// open the database
    if( rc != UNQLITE_OK )
    {
        cerr<<"Can not open database "<<db_name<<endl;
        m_db = NULL;
        return false;
    }

    /// initialize the cursor
    if( UNQLITE_OK != unqlite_kv_cursor_init(m_db, &m_db_cur))
    {
        cerr<<"Error, Can not initialize the cursor"<<endl;
        return false;
    }

    /// update the m_number_of_record if opens a existing database
    rc = unqlite_kv_cursor_first_entry(m_db_cur);
    if( rc != UNQLITE_OK)
    {
        cerr<<"Can not locate the first cursor"<<endl;
        return false;
    }
    while( unqlite_kv_cursor_valid_entry(m_db_cur) )
    {
        m_number_of_record++;
        unqlite_kv_cursor_next_entry(m_db_cur);
    }
    return true;
}

bool matdb::store_mat( const string &key_value, const Mat &mat_value)
{
    if( !is_opened() )
    {
        cerr<<"Database not opened "<<endl;
        return false;
    }

    if( key_value.empty())
    {
        cerr<<"Can not use empty string as key value "<<endl;
        return false;
    }
    if( mat_value.empty())
    {
        cerr<<"Can not use empty Mat as value"<<endl;
        return false;
    }
    
    bool has_before = has_key( key_value);

    /// convert a Mat into string format using FileStorage 
    m_fs.open(".xml", FileStorage::WRITE+FileStorage::MEMORY);
    m_fs<<"mat"<<mat_value;
    string buf = m_fs.releaseAndGetString();
    
    /// store it into database
    int rc = unqlite_kv_store( m_db, key_value.c_str(), key_value.length(), buf.c_str(), buf.length() );
    if( rc != UNQLITE_OK )
    {
        cerr<<"Store Mat failed "<<endl;
        return false;
    }
    
    /// update the total number of record
    if( !has_before )
        m_number_of_record++;

    return true;
}


bool matdb::fetch_mat( const string &key_value, Mat &mat_value) const
{
    if( key_value.empty())
    {
        cerr<<"Error, key_value should not be empty"<<endl;
        return false;
    }
    
    if( !is_opened() )
    {
        cerr<<"Error, database is not opened"<<endl;
        return false;
    }
    
    /// fetch the value, first get the length of the message, second fecth it 
    int rc=0;
    /*! first get the size */
    unqlite_int64 message_len = 0;
    rc = unqlite_kv_fetch( m_db, key_value.c_str(), -1, NULL, &message_len );
    if( rc != UNQLITE_OK)
    {
        cerr<<"Can not get the length of the Mat "<<endl;
        return false;
    }
    
    if( m_buffer.size() < message_len)
        m_buffer.resize(message_len, '0');

    rc = unqlite_kv_fetch(m_db, key_value.c_str(), -1, &m_buffer[0], &message_len);
    if( rc != UNQLITE_OK)
    {
        cerr<<"Can not get the message "<<endl;
        return false;
    }

    /*! convert vector char to string */
    string string_buf( m_buffer.begin(), m_buffer.begin()+message_len);

    /*! convert buf into Mat*/
    m_fs.open( string_buf, FileStorage::READ+FileStorage::MEMORY);
    m_fs["mat"]>>mat_value;

    return true;
}

bool matdb::has_key( const string &key_value) const
{
    if( !is_opened())
    {
        cerr<<"Error , Database is not opened "<<endl;
        return false;
    }

    int rc = 0;
    unqlite_int64 message_len=0;
    rc = unqlite_kv_fetch( m_db, key_value.c_str(), -1, NULL, &message_len);
    if( rc == UNQLITE_OK)
        return true;
    else
        return false;
}


bool matdb::delete_mat( const string &key_value)
{
    if( !is_opened())
    {
        cerr<<"Error , Database is not opened "<<endl;
        return false;
    }

    int rc=0;
    rc = unqlite_kv_delete( m_db, key_value.c_str(), key_value.length() );
    if( rc == UNQLITE_OK)
    {
        if( m_number_of_record > 0 )
            m_number_of_record--;
        return true;
    }
    else
        return false;
}
