'''
Created on 18 Jul 2019

@author: filipe
'''
from flask import Flask 
from flask import render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")



@app.route("/beat_results.html")
def results():
    return None

if __name__ == "__main__":
    pass 