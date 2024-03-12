[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/matheus-perches/)

#### [Return to my Portfolio](https://github.com/matheusperches/matheusperches.github.io) 

## [Music Recommendation ML](https://github.com/matheusperches/MusicRecommendation)

#### An experiment using Python with the Jupyter Notebook platform to make music recommendations with Machine Learning.

#### Background
- This is part of my series of study on Python with Machine Learning, using a small sinthetic dataset of music preferences by age and gender. 
- This project includes a persistent prediction model for music prefences based on gender and age. It uses both as input and outputs the music preference.

### Prediction Code

```py
# Importing the Pandas and SKLearn libraries.
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reading the data from a CSV file.
music_data = pd.read_csv('music.csv')
# This is the input data 
X = music_data.drop(columns=['genre'])
# This is the output data 
y = music_data['genre']
# Defining a percentage of the data for testing the accuracy of the model
# first two input sets and last two are output sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Selecting the ML model from SKLearn
model = DecisionTreeClassifier()
# Importing the input and the output data into the model using fit()
model.fit(X_train.values, y_train)
# Making a prediction of the music preferences
predictions = model.predict(X_test.values)
# Calculating the accuracy of the model with the test data output and the predictions made with the X test input data.
score = accuracy_score(y_test, predictions)
score
```

Output: ``1.0``

- Due to the low amount of data in this dataset, the model accuracy will vary between each training session. 
- For experimentation and study purposes, this should suffice.


### Persisting the model 

#### In order to persist this model, we have to dump it into a file that can be read later. By utilizing the joblib library, this can be easily achieved.

```py
# Importing the Pandas and SKLearn libraries.
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import joblib

# Reading the data from a CSV file.
music_data = pd.read_csv('music.csv')

# This is the input data 
X = music_data.drop(columns=['genre'])

# This is the output data 
y = music_data['genre']

# Selecting the ML model from SKLearn
model = DecisionTreeClassifier()

# Importing the input and the output data into the model using fit()
model.fit(X.values, y)

joblib.dump(model, 'music-recommender.joblib')
```


### Visualization tree

Below you can see a visualization tree of how the model makes its prediction. You can create an age and gender (1 = male ; 0 = female) and follow along.

![Visualization tree in dot language](https://raw.githubusercontent.com/matheusperches/MusicRecommendation/main/media/dotfile%20visualization.png)

This dot file was exported using the file "model_decisiontree.ipynb" and read using Visual Studio Code using the following libraries:
- Graphviz (dot) language support for Visual Studio Code
- Graphviz Interactive Preview

#### The following snippet describes the graph:

```python
# Importing the Pandas and SKLearn libraries.
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Reading the CSV
music_data = pd.read_csv('music.csv')

# Creating input and output sets
X = music_data.drop(columns = ['genre'])
y = music_data['genre']

# Creating the model
model = DecisionTreeClassifier()

# Training
model.fit(X, y)

tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names = ['age', 'gender'], 
                    class_names = sorted(y.unique()),
                    label = 'all',
                    rounded = True,
                    filled = True)
```





