
# Deploying Machine Learning Models on the Cloud

## The Dataset

The dataset is the infamous iris dataset.

| Sepal Length | Sepal Width  | Petal Length  | Petal Width | Species |
|---- | --- | --- | --- | --------------- |
| 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa     |
| 7.0 | 3.2 | 4.7 | 1.4 | Iris-versicolor |
| 6.3 | 3.3 | 6.0 | 2.5 | Iris-virginica  |

![Iris Dataset](https://deeplearning.cms.waikato.ac.nz/img/iris.png)

It consists of three species of flowers. Each flower has its own sepals and petals with a length and width.

![Iris sepal and petal](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQM3aH4Q3AplfE1MR3ROAp9Ok35fafmNT59ddXkdEvNdMkT8X6E)

## Creating the Model

We're going to be creating a fairly simple classifier.

```{python}
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  

def get_classifier():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign column names to the dataset
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(url, names=names)  

    # Set features (x) and labels (y)
    X = dataset.iloc[:, :-1].values  
    y = dataset.iloc[:, 4].values  

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

    # Fit the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Apply the scaler to the data (0 to 1)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the classifier
    classifier = KNeighborsClassifier(n_neighbors=10)  
    classifier.fit(X_train, y_train)  

    # We'll need to use the scaler and the classifier later
    return scaler, classifier
```

## Creating the Server

We'll be using flask for our python server! Fun fact: Don't use flask for a sites that need to handle a lot of traffic. It was actually created as an [April's fool joke.](http://lucumr.pocoo.org/2010/4/3/april-1st-post-mortem/). But for our purposes, it'll work fine!

Here's the code to create a bare minimum server:

```{python}
from flask import Flask, request

# initialize the flask app
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
```

## Combining the Model and Sever

Now, let's pull in our classifier so that we can use with with requests.

```{python}
from classifier import get_classifier

# build the classifier
scaler, classifier = get_classifier()
```

Now, let's add a route for our predictions

```{python}
import json

@app.route('/predict')
def make_prediction():
    if not request.get_json():
        abort(400)
    request_data = request.get_json()

    s_l = float(request_data.get("sepal_length"))
    s_w = float(request_data.get("sepal_width"))
    p_l = float(request_data.get("petal_length"))
    p_w = float(request_data.get("petal_width"))

    x = [[s_l, s_w, p_l, p_w]]
    x_transformed = scaler.transform(x)
    result = classifier.predict(x_transformed)
    return str(result[0])
```

See the final code for the server in `server.py`.

## Deploying

After paying for a server, you could set up the server, install python and all the necessary requirements. But that's a pain for everyone.

To make it easier, we're going to build a Docker container that contains everything the server needs to run.

Here is the Dockerfile that describes how we want the server to run.

```{docker}
FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["server.py"]
```

While in the directory of your server and Dockerfile, type

```{bash}
docker build -t my-model .
                 ^       ^
    container name      location of Dockerfile
```

### Request examples

#### Versicolor

```{json}
{
    "sepal_length": 6,
    "sepal_width": 2.8,
    "petal_length": 4,
    "petal_width": 1.3
}
```

#### Verginica

```{json}
{
    "sepal_length": 7,
    "sepal_width": 3,
    "petal_length": 6,
    "petal_width": 2
}
```

#### Setosa

```{json}
{
    "sepal_length": 5,
    "sepal_width": 4,
    "petal_length": 1,
    "petal_width": 0.3
}
```