# disaster_respnse_pipeline

### Table of Contents

1. [Description](#description)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing](#licensing)
6. [Acknowledgements](#acknowledgements)




## Description<a name="description"></a>

When disaster comes, people would send infomation from several places like SMS, Tweet,Facebook and other public or private places. One of the important things is to classify their needs into specific categories from those messages so different agencies in charge can handle those needs quickly and  precisely.
This project is to create a machine learning pipeline to categorize these messages and set up a user-friendly webserver so that workers of agencies can easily classifying and distributing specific needs within those messages.

## Installation <a name="installation"></a>

Clone this GIT repository:

```
git clone https://github.com/eulyzi/disaster_respnse_pipeline.git
```

## File Descriptions <a name="files"></a>

There are 3 parts of the project:
 - `app` contains a flask webserver which shows visualizations of the data and a classification prediction by the trained model.
 - `data` shows a data set containing real messages that were sent during disaster events. 
 - `models` diplays the NLP pipeline used to categorize messages into specfic need.
  
 **The file structure of the project:**

```
    - app
    | - template
    | |- master.html  # main page of web app
    | |- go.html  # classification result page of web app
    |- run.py  # Flask file that runs app

    - data
    |- disaster_categories.csv  # data to process 
    |- disaster_messages.csv  # data to process
    |- process_data.py
    |- InsertDatabaseName.db   # database to save clean data to

    - models
    |- train_classifier.py
    |- classifier.pkl  # saved model 
```

### Instructions <a name="instructions"></a>:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to `http://0.0.0.0:3001/` or `http://127.0.0.1:3001/` for Windows.

## Licensing <a name="licensing"></a>

MIT.

## Acknowledgements <a name="acknowledgements"></a>

This project is part of  [Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025) 's `Data Science Nanodegree Program`.  
The dataset is provided by [Figure Eight](https://appen.com/).
