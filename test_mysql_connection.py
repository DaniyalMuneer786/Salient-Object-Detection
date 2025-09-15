#!/usr/bin/env python3
"""
Simple MySQL connection test script
Run this to test if your MySQL connection is working
"""

import mysql.connector
from mysql.connector import Error

def test_mysql_connection():
    """Test MySQL connection with the same credentials as the Flask app"""
    
    # Database configuration (same as in app.py)
    config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '',  # Update this if you have a password
        'database': 'SOD',
        'port': 3306,
        'charset': 'utf8mb4',
        'autocommit': True,
        'connect_timeout': 10
    }
    
    print("Testing MySQL connection...")
    print(f"Host: {config['host']}")
    print(f"User: {config['user']}")
    print(f"Database: {config['database']}")
    print(f"Password: {'[SET]' if config['password'] else '[EMPTY]'}")
    print("-" * 50)
    
    try:
        # Try to connect
        connection = mysql.connector.connect(**config)
        
        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"✅ Successfully connected to MySQL Server version {db_info}")
            
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            database_name = cursor.fetchone()
            print(f"✅ Connected to database: {database_name[0]}")
            
            # Test a simple query
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            print(f"✅ Test query successful: {result}")
            
            # Check if user_management table exists
            cursor.execute("SHOW TABLES LIKE 'user_management'")
            table_exists = cursor.fetchone()
            if table_exists:
                print("✅ user_management table exists")
                
                # Count users
                cursor.execute("SELECT COUNT(*) FROM user_management")
                user_count = cursor.fetchone()[0]
                print(f"✅ Found {user_count} users in user_management table")
            else:
                print("❌ user_management table does not exist")
                
            cursor.close()
            connection.close()
            print("✅ Connection closed successfully")
            
        return True
        
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        
        # Provide specific error messages for common issues
        if "Access denied" in str(e):
            print("\n🔧 Possible solutions:")
            print("1. Check if MySQL is running")
            print("2. Verify username and password")
            print("3. Make sure the user has access to the database")
        elif "Can't connect to MySQL server" in str(e):
            print("\n🔧 Possible solutions:")
            print("1. Check if MySQL service is running")
            print("2. Verify the host and port")
            print("3. Check firewall settings")
        elif "Unknown database" in str(e):
            print("\n🔧 Possible solutions:")
            print("1. Create the 'SOD' database")
            print("2. Check database name spelling")
        
        return False

def create_database_if_not_exists():
    """Create the SOD database if it doesn't exist"""
    
    # Connect without specifying database
    config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '',  # Update this if you have a password
        'port': 3306,
        'charset': 'utf8mb4',
        'autocommit': True
    }
    
    try:
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS SOD")
        print("✅ Database 'SOD' created or already exists")
        
        cursor.close()
        connection.close()
        return True
        
    except Error as e:
        print(f"❌ Error creating database: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MySQL Connection Test Script")
    print("=" * 60)
    
    # First test connection without database
    if create_database_if_not_exists():
        # Then test connection with database
        test_mysql_connection()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
