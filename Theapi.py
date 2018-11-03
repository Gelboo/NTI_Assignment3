from model import My_model
from flask import Flask,request
app = Flask(__name__)
m = My_model()

@app.route("/")
def preprocessing():
    m.load_data()
    m.preprocessing()
    m.split_data()
    return "<h1> the data Is Loaded and preprocess Ready</h1>\
            <h2> Go to the train </h2>\
            <form method='GET' action='http://127.0.0.1:5000/train_page'>\
            <input type='submit' value='Go to train'>\
            </form>\
    "
@app.route("/train_page")
def train_page():
    return "<form method = 'GET' action ='http://127.0.0.1:5000/train'>\
    <h2>Please choose the model you want to work with:</h2>\
    </br>\
    <select name='name'>\
      <option value='LinearRegression'>LinearRegression</option>\
      <option value='Knn'>Knn</option>\
      <option value='SVR'>SVR</option>\
      <option value='DecisionTree'>DecisionTree</option>\
    </select>\
  </br></br>\
  <input type='submit' value='train'>\
  </form>"
@app.route("/train",methods=['GET','POST'])
def train():
    model_name = request.args.get('name')
    m.train(model_name)
    return "<h1>Train Completed :) </h1>\
    <h2> show the accuracy of the model </h2>\
    <form action='http://127.0.0.1:5000/accuracy'>\
    <input type='submit' value='Go to accuracy'>\
    </form>"

@app.route("/accuracy")
def accuracy():
    mse,r2 = m.compute_accuracy()
    return "<table border=1px solid black>\
            <tr>\
            <th>MSE</th>\
            <th>R2</th>\
            </tr>\
            <tr>\
            <td>"+str(mse)+"</td>\
            <td>"+str(r2)+"</td>\
            </tr>\
            </table>\
            </br>\
            <h1> Go to Prediction! </h1>\
            <form action='http://127.0.0.1:5000/predict_page'>\
            <input type='submit' value='Go to Predict'>\
            </form>\
            "
@app.route("/predict_page",methods=['GET','POST'])
def predict_page():
    return "<form method='GET' action='http://127.0.0.1:5000/predict'>\
            <h1> please Choose you input values </h1>\
            <p>rank</p>\
            </tr><select name='rank'>\
            <option>Prof</option>\
            <option>AssocProf</option>\
            <option>AsstProf</option>\
            </select>\
            </br>\
            <p>discipline</p>\
            <select name='discipline'>\
            <option>A</option>\
            <option>B</option>\
            </select>\
            </br>\
            <p>phd</p>\
            <input type='text' name=phd>\
            </br>\
            <p>service</p>\
            <input type='text' name=service>\
            </br>\
            <p>sex</p>\
            <select name='sex'>\
            <option>Male</option>\
            <option>Female</option>\
            </select>\
            </br></br>\
            <input type='submit' value='predict'>\
            </form>\
    "
@app.route("/predict",methods=['GET','POST'])
def predict():
    rank = request.args.get('rank')
    discipline = request.args.get('discipline')
    phd = request.args.get('phd')
    service = request.args.get('service')
    sex = request.args.get('sex')
    res = m.predict(rank=rank,discipline=discipline,phd=phd,service=service,sex=sex)[0]
    return "<h1> The Person with the information you enter </h1>\
            <h2> take salary = "+str(res)+" $</h2>"
