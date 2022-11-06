# Disaster-Response-Classification-Project

***

### Overview

***

The project is based on the dataset provided by Figure 8 and Udacity on Twitter messages during disaster. The idea is to train a machine learning algorithm that can classify tweets into various categories related to the needs of victims. This would help notify the relevant organization to better provide support to those who needs specific aids the most. For example, if the message is classified to be in medical category, organization such as [Doctors Without Borders](https://www.doctorswithoutborders.org/) team could be notified and an emergency medical relief team could be dispatched immediately. 

The project involves building NLP (Natural Language Processing) and ML (Machine Learning) pipeline utilizing nltk and sklearn's random forest classifier library. GridSearchCV was utilized to get the best performance. The classification results of a new twitter messages are displayed in a flask web app for ease of visualizations. 