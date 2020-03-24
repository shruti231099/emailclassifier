
from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
word_list=pickle.load(open('my_string.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message=request.form.get('message')
    sample=[]
    for i in word_list:
        sample.append(message.split(" ").count(i[0]))
    sample=np.array(sample)
    result=model.predict(sample.reshape(1,3000))

    if result == 0:
        return render_template('index.html',label=1)
    else:
        return render_template('index.html',label=-1)


if __name__=='__main__':
    app.run(debug=True)