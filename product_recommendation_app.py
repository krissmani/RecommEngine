# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:19:23 2020

@author: MY394222
"""



from flask import Flask,request
from flask_httpauth import HTTPBasicAuth
from flask_cors import CORS
from CC_Reco_2020_dec10 import get_results
import json

app = Flask(__name__)
auth = HTTPBasicAuth()
cors = CORS(app)
CORS(app)

USER_DATA = {
    "admin": "Wipro@12345"
}

@auth.verify_password
def verify(username, password):
    if not (username and password):
        return False
    return USER_DATA.get(username) == password

@app.route('/api/v1/recommendations', methods=['POST'])
@auth.login_required
def geterror():
 
    if request.method == 'POST':
        try:
            post_request = request.get_json(force=True)
            return get_results(post_request)
        except Exception as e:
            return {"recommendations":[],"status":"failed","error":e}
            



if __name__ == '__main__':
    app.run()
