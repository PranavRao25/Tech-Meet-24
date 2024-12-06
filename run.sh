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

# Set up the database in the background
echo "Setting up the database..."
python DataBase/setup.py &

# Wait for 400 seconds before starting the interface
echo "Waiting for 400 seconds to ensure the database is set up..."
sleep 500

# Running the interface
cd User_Interface
streamlit run Interface.py
