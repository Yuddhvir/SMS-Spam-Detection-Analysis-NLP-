# SMS-Spam-Detection-Analysis-NLP-

This project is about building a spam detection system for SMS messages using deep learning techniques in TensorFlow2. Three different architectures, namely Dense Network, LSTM, and Bi-LSTM, have been used to build the spam detection model. The accuracy of the models is compared and evaluated to select the best one.

Dataset Link : https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
The SMS Spam Collection dataset from the UCI Machine Learning Repository is used for this project. The dataset contains 5,574 SMS messages, out of which 4,827 messages are labeled as ham (non-spam) and 747 messages as spam. The dataset is split into 4,000 messages for training and 1,574 messages for testing.

Steps Involved :
The following steps are involved in the project:

Load and Explore the Data: The dataset is loaded into a Pandas DataFrame and explored to get insights into the distribution of ham and spam messages.

Prepare Train-Test Data: The messages are tokenized, and their corresponding labels are one-hot encoded. The dataset is split into training and testing sets in the ratio of 80:20.

Train the Spam Detection Model: The three models - Dense Network, LSTM, and Bi-LSTM - are trained using the training dataset. The models are evaluated using the validation dataset.

Compare and Select the Final Model: The accuracy of the models is compared, and the best-performing model is selected.

Use the Final Trained Classifier to Classify New Messages: The final model is used to classify new messages as ham or spam.

Usage :
To use the SMS spam detection model, follow these steps:

Install the required packages: cd sms-spam-detection pip install -r requirements.txt
Download the dataset from the UCI Machine Learning Repository and place it in the data directory: mkdir data wget https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection -O data/spam.csv
Run the .pynb script to train the models and select the best one
Run the streamlit app as : streamlit run app.py
Accuracy of the Models :
The accuracy of the models is as follows:

Dense Network - 98.5%
SVM - 97.6%
Bi-LSTM - 98.8%
LSTM - 98.6%
