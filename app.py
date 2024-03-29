from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import nltk
import numpy as np
import tensorflow
import tflearn
import pickle
from nltk.stem.lancaster import LancasterStemmer
from flask_sqlalchemy import SQLAlchemy
import os

nltk.data.path.append('./nltk_data/')
stemmer = LancasterStemmer()


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"]='postgres://lrmblnrgliqize:f68a5084a90b05e68c1aba33d945417a1c7891317bcffd11ea7f4071f93481d1@ec2-34-198-31-223.compute-1.amazonaws.com:5432/d8khg822mrebg9'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)

class Mce(db.Model):
    __tablename__= 'mce_question'
    id =db.Column(db.Integer,primary_key=True)
    question=db.Column(db.String(500))
    answer = db.Column(db.String(1000))
    def __init__(self,question,answer):
        self.question=question
        self.answer=answer

    def __repr__(self):
        return '<Question %r>' % self.question






def bag_of_word(s, word):
    
    stemmer = LancasterStemmer()
    bag = [0 for _ in range(len(word))]
    s_word = nltk.word_tokenize(s)
    s_word = [stemmer.stem(s.lower()) for s in s_word]

    for se in s_word:
        for i, w in enumerate(word):
            if w in se:
                bag[i] = 1

    bag = np.array(bag)
    bag = bag.reshape( len(word))

    return bag

@app.route("/")
def hello():
    return "<h1><center>MECHATRONICS CHATBOT</center></h1>"



@app.route("/sms", methods=['POST'])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    # Fetch the message
    with open("data.pickle","rb") as f:
        word,label,data,train,output =pickle.load(f)
    resp = MessagingResponse()
    msg = request.form.get('Body')
    tensorflow.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(train[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.load("model.tflearn")

    result = model.predict([bag_of_word(msg, word)])[0]
    result_index = np.argmax(result)
    tag = label[result_index]
    if result[result_index] > 0.6:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                response = tg["response"]
                resp.message(response[0])
                for_store=Mce(question=msg,answer=response[0])
                db.session.add(for_store)
                db.session.commit()
    else:
        resp.message("i dont understand please ask another question")
        for_store = Mce(question=msg, answer="i dont understand please ask another question")
        db.session.add(for_store)
        db.session.commit()



    # Create reply



    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
