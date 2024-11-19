from flask import Flask , jsonify, request,send_file
# import io
# import run_script
# import numpy as np
# import pandas as pd
# import json
import os
# import shutil
# from modelTrain.config import *
# import teams_rt_improve as teams_rt
# from modelTrain.data_process import data_process
import threading


# import multiprocessing

# 当前文件的绝对路径
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(DIR_PATH,'frontEnd'), static_url_path='/')


@app.route('/')
def index_dataV():
    return app.send_static_file('index_DataV.html')

@app.route("/step1")
def step1():
    pass

@app.route("/step2")
def step2():
    pass

@app.route("/step3")
def step3():
    pass

@app.route("/step4")
def step4():
    pass

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)



