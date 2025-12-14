#!/bin/bash

# chmod +x setup.sh

# Step 1: Create a Python virtual environment in the current directory
echo "Creating a Python virtual environment in the current directory..."
python3 -m venv venv

# Step 2: Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Step 3: Install Apache Airflow using pip
echo "Installing Apache Airflow..."
pip install apache-airflow

# Step 4: Initialize the Airflow database
echo "Initializing the Airflow database..."
airflow db init

echo "Setup complete!"
