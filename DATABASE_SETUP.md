# Database Setup Guide

## Quick Fix for "Database connection error"

If you're getting a "Database connection error" when trying to login, follow these steps:

### 1. Check if MySQL is Running

**Windows:**
```cmd
# Check if MySQL service is running
sc query mysql

# Start MySQL service if it's not running
net start mysql
```

**Alternative (if using XAMPP):**
- Open XAMPP Control Panel
- Start MySQL service

### 2. Test Database Connection

Run the test script to diagnose the issue:
```bash
python test_mysql_connection.py
```

### 3. Common Issues and Solutions

#### Issue: "Access denied for user 'root'@'localhost'"
**Solution:**
1. Open MySQL command line or phpMyAdmin
2. Reset root password:
```sql
ALTER USER 'root'@'localhost' IDENTIFIED BY '';
FLUSH PRIVILEGES;
```

#### Issue: "Unknown database 'SOD'"
**Solution:**
1. Create the database:
```sql
CREATE DATABASE SOD;
```

#### Issue: "Can't connect to MySQL server"
**Solution:**
1. Check if MySQL is running on port 3306
2. Verify firewall settings
3. Check if MySQL is bound to 127.0.0.1

### 4. Update Configuration

If you have a MySQL password, update `app.py`:
```python
app.config['MYSQL_PASSWORD'] = "your_actual_password"
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Test the App

1. Start your Flask app:
```bash
python app.py
```

2. Visit `/test_db` endpoint to test database connection:
```
http://localhost:5000/test_db
```

3. Try logging in again

## Database Schema

The app will automatically create these tables if they don't exist:

### user_management
- User_ID (Primary Key)
- Name
- Email (Unique)
- Password (Hashed)
- Created_At

### admin
- Admin_ID (Primary Key)
- Name
- Email (Unique)
- Password
- Salary
- Created_At

## Troubleshooting

### Check MySQL Status
```bash
# Windows
sc query mysql

# Linux/Mac
sudo systemctl status mysql
```

### Check MySQL Logs
```bash
# Windows (XAMPP)
C:\xampp\mysql\data\mysql_error.log

# Linux
sudo tail -f /var/log/mysql/error.log
```

### Test MySQL Connection Manually
```bash
mysql -u root -p -h 127.0.0.1
```

### Reset MySQL Root Password
1. Stop MySQL service
2. Start MySQL in safe mode
3. Reset password
4. Restart MySQL service

## Still Having Issues?

1. Check the Flask app logs for detailed error messages
2. Verify MySQL version compatibility
3. Ensure all required Python packages are installed
4. Check if the database user has proper permissions

## Quick Test Commands

```bash
# Test if MySQL is accessible
telnet 127.0.0.1 3306

# Check MySQL process
tasklist | findstr mysql

# Test database connection
python test_mysql_connection.py
```
