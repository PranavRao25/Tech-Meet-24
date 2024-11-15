#!/bin/bash

# Check if Python and pip are installed
if ! command -v python &> /dev/null
then
    echo "Python is not installed. Please install Python to proceed."
    exit
fi

if ! command -v pip &> /dev/null
then
    echo "pip is not installed. Please install pip to proceed."
    exit
fi

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Set up the database
echo "Setting up the database..."
python DataBase/setup.py
# echo "Database setup completed"

# # running the interface
# streamlit run ./User_Interface/Interface.py