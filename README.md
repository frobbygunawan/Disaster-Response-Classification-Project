# Disaster-Response-Classification-Project

***

### Overview

***

The project is based on the dataset provided by Figure 8 and Udacity on Twitter messages during disaster. The idea is to train a machine learning algorithm that can classify tweets into various categories related to the needs of victims. This would help notify the relevant organization to better provide support to those who needs specific aids the most. For example, if the message is classified to be in medical category, organization such as [Doctors Without Borders](https://www.doctorswithoutborders.org/) team could be notified and an emergency medical relief team could be dispatched immediately. 

The project involves building NLP (Natural Language Processing) and ML (Machine Learning) pipeline utilizing nltk and sklearn's random forest classifier library. GridSearchCV was utilized to get the best performance. The classification results of a new twitter messages are displayed in a flask web app for ease of visualizations. 

***

### Usage

***

1. In the project's root directory, run the following command sequentially :

    - ETL pipeline to load, clean, and store data in the database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - ML pipeline that train the model and store the fitted model to a pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory and run `python run.py` to open the web app

***

### Files in the repository

***

The following files are contained in the repository :  
app  
| - template  
| |- master.html # main page of web app  
| |- go.html # classification result page of web app  
|- run.py # Flask file that runs app  
data  
|- disaster_categories.csv # data to process  
|- disaster_messages.csv # data to process  
|- process_data.py  
|- DisasterResponse.db # database to save clean data to  
models  
|- train_classifier.py  
|- classifier.pkl # saved model  
README.md  
