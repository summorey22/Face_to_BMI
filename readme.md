# Face to BMI using CNN
This project uses Convoluted Neural Networks to determine Body Mass Index from facial images of humans. It is developed using python.

Python version should be below Python 3.8

Recommended Version: [Python 3.7.8rc1](https://www.python.org/downloads/release/python-378rc1/)


For acquiring dataset, please send an email to sumit.morey20@vit.edu providing following details:

1. Name
2. Affiliation
3. Email Address
4. Reason for database download

## Create and Activating a Virtual Environment
Open a folder in your editor and enter the following commands in your terminal:

`python -m venv --system-site-packages .\venv`

If your are using a Windows machine, run the following command:

`Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force`

Finally, enter the following code in your terminal:

`.\venv\Scripts\activate`

## Installing Required Packages
Run the following code in your terminal:

`pip install -r requirements.txt`

## Generate .pb and .tflite Model
`python main.py`
