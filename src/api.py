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


# @app.route('/api/fzy',methods=['GET'])
# def get_fzy():
#     return jsonify({'time': time.time()})

# @app.route('/api/inject_fault', methods=['GET'])
# def inject_fault():
#     fault_style = request.args.get('fault_style')
#     seconds_start = request.args.get('seconds_start')
#     seconds_end = request.args.get('seconds_end')
#     # print(fault_style)
#     # print(seconds_start)
#     # print(seconds_end)
#     # print(str(fault_style))
#     with open(os.path.join(DIR_PATH,'communications','fault_injection_type.txt'), 'w') as f:
#         f.write(str(fault_style))

#     with open(os.path.join(DIR_PATH,"communications","inject_time.txt"), 'w') as f:
#         f.write('{} {}'.format(seconds_start, seconds_end))

#     process_state = "故障注入并且开始仿真"
#     run_script.run_simulation()

#     return jsonify({'process_state': '{}'.format(process_state)})

# @app.route('/api/detect_fault',methods=['GET'])
# def detect_fault():
#     process_state = '故障检测'
#     select_project = request.args.get("select_project")
#     print("ate故障检测")

#     run_script.run_detect(select_project)


#     return jsonify({'process_state': '{}'.format(process_state)})

# @app.route('/api/process_data',methods=['GET'])
# def process_data():
#     select_project = request.args.get("select_project")
#     classifier_file = request.args.get("classifier_file")
#     process_state = '数据处理'

#     # teams RT 算法 start
#     # 上阈值
#     UP_threshold = 0.8
#     # 下阈值
#     DOWN_threshold = 0.2

    
#     # 1. 读取“D矩阵”并处理为传统的D矩阵
#     df = pd.read_excel(os.path.join(DIR_PATH,"communications\D-matrix.xlsx"),index_col="fault_type",engine='openpyxl')
#     # 1.1 数值为0或1不改变
#     df[df == 1] = 1
#     df[df == 0] = 0
#     # 1.2 数值趋近 0或1的 就近处理
#     df[df > UP_threshold] = 1
#     df[df < DOWN_threshold] = 0
#     # 1.3 数值在中间的，第一个D矩阵 取0  第二个D矩阵取1
#     df_first = df.copy()
#     df_first[(df_first >= DOWN_threshold) & (df_first <= UP_threshold)] = 0
#     tempory_path_first = os.path.join(DIR_PATH,"communications\D-matrix_tempory_fisrt.xlsx")
#     df_first.to_excel(tempory_path_first)

#     df_second = df.copy()
#     df_second[(df_second >= DOWN_threshold) & (df_second <= UP_threshold)] = 1
#     tempory_path_second = os.path.join(DIR_PATH,"communications\D-matrix_tempory_second.xlsx")
#     df_second.to_excel(tempory_path_second)

#     # 2. 读取D矩阵
#     diag = teams_rt.Diagnostic_imporved()
#     DMatrix_info_first = diag.get_Dmatrix(tempory_path_first)
#     DMatrix_info_second = diag.get_Dmatrix(tempory_path_second)
#     # 3. 进行测试
#     with open(os.path.join(DIR_PATH,"communications\start_time.json"), 'r') as f:
#         testResult = json.load(f)["detect_result"]
    
#     # testResult = [1, 0, 0, 0, 1]+21*[0]
#     diag.improved_teams_diag(DMatrix_info_first,DMatrix_info_second, testResult)
#     print(diag.diagResultName)
#     # teams RT 算法 end
#     run_script.run_slice_data(select_project)
#     run_script.run_test_classifier(select_project,classifier_file)    
#     return jsonify({'process_state': '{}'.format(process_state),"teamsRTdiagResult":json.dumps(diag.diagResultName)})

# @app.route('/api/get_result',methods=['GET'])
# def get_result():
#     process_state = '获取结果'

#     results_df = pd.read_csv(os.path.join(DIR_PATH,'communications','diagnose_result.csv'))
#     results = results_df.values
#     result_table = [
#             {'probablity':"0",
#             'fault_style':"无故障"},
#             {'probablity':"0",
#             'fault_style':"自动油门自动断开"},
#             {'probablity':"0",
#             'fault_style':"自动油门杆偏差"},
#             {'probablity':"0",
#             'fault_style':"自动油门链路失效"},
#             {'probablity':"0",
#             'fault_style':"发动机空中停车"},
#             {'probablity':"0",
#             'fault_style':"发动机推力下降"},
#             {'probablity':"0",
#             'fault_style':"发动机漏油"},
#             {'probablity':"0",
#             'fault_style':"方向舵卡死"},
#             {'probablity':"0",
#             'fault_style':"方向舵硬件失效（上饱和）"},
#             {'probablity':"0",
#             'fault_style':"方向舵硬件失效（下饱和）"},
#             {'probablity':"0",
#             'fault_style':"方向舵漂浮"},
#             {'probablity':"0",
#             'fault_style':"方向舵控制效能损失"},
#             {'probablity':"0",
#             'fault_style':"副翼卡死"},
#             {'probablity':"0",
#             'fault_style':"副翼硬件失效（上饱和）"},
#             {'probablity':"0",
#             'fault_style':"副翼硬件失效（下饱和）"},
#             {'probablity':"0",
#             'fault_style':"副翼漂浮"},
#             {'probablity':"0",
#             'fault_style':"副翼控制效能损失"},
#             {'probablity':"0",
#             'fault_style':"升降舵卡死"},
#             {'probablity':"0",
#             'fault_style':"升降舵硬件失效（上饱和）"},
#             {'probablity':"0",
#             'fault_style':"升降舵硬件失效（下饱和）"},
#             {'probablity':"0",
#             'fault_style':"升降舵漂浮"},
#             {'probablity':"0",
#             'fault_style':"升降舵控制效能损失"},
#             {'probablity':"0",
#             'fault_style':"前缘襟翼卡死"},
#             {'probablity':"0",
#             'fault_style':"前缘襟翼硬件失效（上饱和）"},
#             {'probablity':"0",
#             'fault_style':"前缘襟翼硬件失效（下饱和）"},
#             {'probablity':"0",
#             'fault_style':"前缘襟翼漂浮"},
#             {'probablity':"0",
#             'fault_style':"前缘襟翼控制效能损失"},
#             {'probablity':"0",
#             'fault_style':"后缘襟翼卡死"},
#             {'probablity':"0",
#             'fault_style':"后缘襟翼硬件失效（上饱和）"},
#             {'probablity':"0",
#             'fault_style':"后缘襟翼硬件失效（下饱和）"},
#             {'probablity':"0",
#             'fault_style':"后缘襟翼漂浮"},
#             {'probablity':"0",
#             'fault_style':"后缘襟翼控制效能损失"},
#             {'probablity':"0",
#             'fault_style':"L控制器无信号"},
#             {'probablity':"0",
#             'fault_style':"M控制器无信号"},
#             {'probablity':"0",
#             'fault_style':"N控制器无信号"},
#             {'probablity':"0",
#             'fault_style':"T控制器无信号"},
#             {'probablity':"0",
#             'fault_style':"Y控制器无信号"},
#             {'probablity':"0",
#             'fault_style':"L控制器恒偏差"},
#             {'probablity':"0",
#             'fault_style':"M控制器恒偏差"},
#             {'probablity':"0",
#             'fault_style':"N控制器恒偏差"},
#             {'probablity':"0",
#             'fault_style':"T控制器恒偏差"},
#             {'probablity':"0",
#             'fault_style':"Y控制器恒偏差"},
#             {'probablity':"0",
#             'fault_style':"L控制器随机输出"},
#             {'probablity':"0",
#             'fault_style':"M控制器随机输出"},
#             {'probablity':"0",
#             'fault_style':"N控制器随机输出"},
#             {'probablity':"0",
#             'fault_style':"T控制器随机输出"},
#             {'probablity':"0",
#             'fault_style':"Y控制器随机输出"},
#             {'probablity':"0",
#             'fault_style':"L控制器死机"},
#             {'probablity':"0",
#             'fault_style':"M控制器死机"},
#             {'probablity':"0",
#             'fault_style':"N控制器死机"},
#             {'probablity':"0",
#             'fault_style':"T控制器死机"},
#             {'probablity':"0",
#             'fault_style':"Y控制器死机"},
#             {'probablity':"0",
#             'fault_style':"座舱驾驶杆操纵空程（升降舵）"},
#             {'probablity':"0",
#             'fault_style':"座舱驾驶杆操纵空程（副翼）"},
#             {'probablity':"0",
#             'fault_style':"座舱脚蹬偏斜"},
#             {'probablity':"0",
#             'fault_style':"座舱脚蹬振动"},
#             {'probablity':"0",
#             'fault_style':"开关按钮失效（前缘襟翼）"},
#             {'probablity':"0",
#             'fault_style':"开关按钮失效（后缘襟翼）"},
#             {'probablity':"0",
#             'fault_style':"座舱脚蹬卡滞"}
#             ]
#     for i, x in np.ndenumerate(results):
#         result_table[i[0]]['probablity'] = str( 100*round(1 / (1 + np.exp(-x*2)),2))+"%"
    
#     fault_probabilitys = {'data':[],'rowNum':7,'carousel': 'page','waitTime':50000000}

#     for item in result_table:
#         ele = {}
#         ele['name'] = item['fault_style']
#         ele['value'] = item['probablity']
#         fault_probabilitys['data'].append(ele)

#     with open(os.path.join(DIR_PATH,"communications","diagnosed_fault_kind.txt"), 'r') as f:
#         fault_type = f.read()
#         fault_type = eval(fault_type)
    

#     process_state = "故障诊断结果为："+result_table[fault_type]['fault_style']
#     # print(result_table)
#     # print(results)
    
#     return jsonify({'process_state': '{}'.format(process_state),
#                     'result_table': json.dumps(result_table),
#                     'result_fault': fault_type,
#                     'fault_probabilitys': json.dumps(fault_probabilitys)})

# @app.route('/image', methods=['GET'])
# def get_image():
#     value = request.args.get('value')
#     fault_style = request.args.get('hhh')
#     # print(value)
#     # print(fault_style,"hhh")
#     file_path = 'communications/variables_graph/{}.png'.format(value)
#     # print(file_path)
#     with open(os.path.join(DIR_PATH,file_path), 'rb') as f:
#         image_data = f.read()
    
#     return send_file(io.BytesIO(image_data), mimetype='image/png')

# #### 模型设置相关api ######
# @app.route('/api/create_project',methods=['GET'])
# def create_project():
#     projectCreate_form = json.loads(request.args.get('projectCreate_form'))
    

#     # 1.创建项目文件夹
#     # 项目名字例子 project-撒地方+samplingRate-400+sliceDurationS-10+sliceAheadS-2.5
#     # 其意义是 项目名字+采样频率（Hz）+截取时长（s）+提前裁切时长（s） 项目名称禁止包含-和+符号以及：:符号
#     project_name = "project-"+projectCreate_form['project_name']+'+samplingRate-'+str(projectCreate_form['sampling_rate'])+'+sliceDurationS-'+str(projectCreate_form['slice_duration_s'])+'+sliceAheadS-'+str(projectCreate_form['slice_ahead_s'])
#     project_folder_path = os.path.join(DIR_PATH, 'database', project_name)
#     if(not os.path.exists(project_folder_path)):
#         os.makedirs(project_folder_path)
#     else:
#         return jsonify({'process_state': '创建项目失败，项目已存在'})
#     # 2.创建数据集文件夹
#     data_for_diagnose_folder_path = os.path.join(project_folder_path, 'data_for_diagnose')
#     if(not os.path.exists(data_for_diagnose_folder_path)):
#         os.makedirs(data_for_diagnose_folder_path)
    
#     saved_model_folder_path = os.path.join(project_folder_path, 'saved_model')
#     if(not os.path.exists(saved_model_folder_path)):
#         os.makedirs(saved_model_folder_path)

#     # 3.对于原始数据进行裁切并保存到data_for_diagnose文件夹中
#     print(projectCreate_form)
#     # 3.1 初始化参数
#     data_process_config = Data_process_config()
#     data_process_config.sampling_rate = projectCreate_form['sampling_rate']
#     data_process_config.slice_duration_s = projectCreate_form['slice_duration_s']
#     data_process_config.slice_ahead_s = projectCreate_form['slice_ahead_s']
#     data_process_config.original_data_path = os.path.normpath(projectCreate_form['file_path'])
#     data_process_config.data_for_diagnose_path = data_for_diagnose_folder_path
#     # 3.2 裁切并保存数据
#     try:
#         data_process(data_process_config)
#     except Exception as e:
#         print(e)
#         return jsonify({'process_state': '创建项目失败，数据处理失败'})
    
#     return jsonify({'process_state': '创建项目成功'})


# @app.route('/api/delete_project',methods=['GET'])
# def delete_project():
#     selectedOptions_project = request.args.get('selectedOptions_project')
    
#     project_list = os.listdir(os.path.join(DIR_PATH, 'database'))
   

#     for project in project_list:
#         project_name = project.split('+')[0].split('-')[1]
   
#         if(project_name == selectedOptions_project):
#             project_file = os.path.join(DIR_PATH, 'database', project)
#             if(os.path.exists(project_file)):
   
#                 shutil.rmtree(project_file)
#             else:
#                 return jsonify({'process_state': '删除项目失败，项目不存在'})


#     return jsonify({'process_state': '删除项目成功'})


# @app.route('/api/choose_project',methods=['GET'])
# def choose_project():
#     selectedOptions_project = request.args.get('selectedOptions_project')
#     projectChoose = {"sampling_rate":0,"slice_duration_s":0,"slice_ahead_s":0}
#     options_models = []
#     project_list = os.listdir(os.path.join(DIR_PATH, 'database'))
#     for project in project_list:
#         project_name = project.split('+')[0].split('-')[1]
#         if(project_name == selectedOptions_project):
#             projectChoose['sampling_rate'] = int(project.split('+')[1].split('-')[1])
#             projectChoose['slice_duration_s'] = float(project.split('+')[2].split('-')[1])
#             projectChoose['slice_ahead_s'] = float(project.split('+')[3].split('-')[1])

#             model_list = os.listdir(os.path.join(DIR_PATH, 'database', project,'saved_model'))
#             for model in model_list:
#                 options_models.append({'value':model,'label':model})



#             return jsonify({'process_state': '选择项目成功','projectChoose':json.dumps(projectChoose),'options_models':json.dumps(options_models)})

#     return jsonify({'process_state': '选择项目失败，项目不存在'})


# @app.route('/api/updata_project_list',methods=['GET'])
# def updata_project_list():
#     project_list = os.listdir(os.path.join(DIR_PATH, 'database'))
#     options_project = []
#     for project in project_list:
#         project = project.split('+')[0].split('-')[1]
#         options_project.append({'value':project,'label':project})
#     return jsonify({'options_project': json.dumps(options_project)})


# @app.route('/api/extract_features',methods=['GET'])
# def extract_features():

#     # 1. 设置参数
#     # 1.1 worker_nums
#     n_workers = int(request.args.get('worker_num'))
#     # 1.2 data_for_diagnose_path
#     selectedOptions_project = request.args.get('selectedOptions_project')
#     project_list = os.listdir(os.path.join(DIR_PATH, 'database'))
#     for project in project_list:
#         project_name = project.split('+')[0].split('-')[1]
#         if(project_name == selectedOptions_project):
#             data_for_diagnose_path = os.path.join(DIR_PATH, 'database', project,'data_for_diagnose')
#             break
    
#     # 2. 提取特征
#     ## 开
#     try:
#         run_script.run_extract_features(n_workers,data_for_diagnose_path)
#     except Exception as e:
#         print(e)
#         return jsonify({'process_state': '提取特征失败'})
    
#     print("ate特征提取")
#     return jsonify({'process_state': '提取特征成功'})

# @app.route('/api/select_features',methods=['GET'])
# def select_features():
#     print("选择特征")
#     fdr_level = 10
#     selectedOptions_project = request.args.get('selectedOptions_project')
#     project_list = os.listdir(os.path.join(DIR_PATH, 'database'))
#     for project in project_list:
#         project_name = project.split('+')[0].split('-')[1]
#         if(project_name == selectedOptions_project):
#             data_for_diagnose_path = os.path.join(DIR_PATH, 'database', project,'data_for_diagnose')
#             select_file = os.path.join(DIR_PATH, 'database', project,'select.csv')
#             features_file = os.path.join(DIR_PATH, 'database', project,'all.csv')
#             break
#     try:
#         run_script.run_combine_features(data_for_diagnose_path,features_file)
#         run_script.run_select_features(features_file,fdr_level,select_file)
#     except  Exception as e:
#         print(e)
#         return jsonify({'process_state': '选择特征失败'})
    
#     return jsonify({'process_state': '选择特征成功'})

# @app.route('/api/model_train',methods=['GET'])
# def model_train():
#     test_proportion = 0.3
#     print("模型训练")
#     selectedOptions_project = request.args.get('selectedOptions_project')
#     project_list = os.listdir(os.path.join(DIR_PATH, 'database'))
#     for project in project_list:
#         project_name = project.split('+')[0].split('-')[1]
#         if(project_name == selectedOptions_project):
#             select_file = os.path.join(DIR_PATH, 'database', project,'select.csv')
#             result_file = os.path.join(DIR_PATH, 'database', project,'results.xlsx')
#             saved_model_path = os.path.join(DIR_PATH, 'database', project,'saved_model')
#             break
    
#     try:
#         run_script.run_train_and_evaluate(select_file,test_proportion,result_file,saved_model_path)
#     except Exception as e:
#         print(e)
#         return jsonify({'process_state': '选择特征成功'})
        
#     return jsonify({'process_state': '模型训练成功'})
# def cengci(filename):#输入：矩阵存储文件名，输出：矩阵权重分配和CR值
#     matrix=[]
#     with open(filename,'r') as file:
#         lines=file.readlines()
#         dimension=len(lines)#矩阵维度
#         for line in lines:
#             row=line.strip().split()
#             processed_row = []
#             for element in row:
#                 if '/' in element:  # 判断是否为分数形式，如果是保留两位小数
#                     numerator, denominator = element.split('/')
#                     processed_element = round(float(numerator) / float(denominator), 2)
#                 else:
#                     processed_element = float(element)
#                 processed_row.append(processed_element)
#             matrix.append(processed_row)#从文件中读取矩阵
#     ri=[0,0,0.58,0.9,1.12,1.24,1.32,1.41,1.45,1.49]#一致性指标
#     eigenvalues,eigenvectors=np.linalg.eig(matrix)
#     D=np.diag(eigenvalues)
#     V=eigenvectors
#     lamda=np.max(D)
#     indices=np.where(D==lamda)
#     row_index,col_index=indices[0][0],indices[1][0]
#     column=V[:,col_index]
#     w0=column/np.sum(column)#计算权重
#     cr0=(lamda-dimension)/(dimension-1)/ri[dimension]#计算CR值
#     return w0,cr0
# def SQ(data):
#     rows=data.shape[0]
#     cols=data.shape[1]#输入矩阵的大小，rows为行数（对象个数），cols为列数（指标个数）
#     R=data
#     Rmin=np.min(R,axis=0)#矩阵中最小行
#     Rmax=np.max(R,axis=0)#矩阵中最大行
#     A=Rmax-Rmin#分母 矩阵中最大行减最小行
#     y=R-np.tile(Rmin,(rows,1))#分子 R矩阵每一行减去最小行
#     for j in range(cols):#该循环用于正向指标标准化处理 分子/分母
#         y[:,j]=y[:,j]/A[j]
#     S=np.sum(y,axis=0)#列之和（用于列归一化）
#     Y=np.zeros((rows,cols))
#     for i in range(cols):
#         Y[:,i]=y[:,i]/S[i]#该循环用于列的归一化
#     k=1/np.log(rows)
#     lnYij1=np.zeros((rows,cols))
#     for i in range(rows):
#         for j in range(cols):
#             if Y[i,j]==0:
#                 lnYij1[i,j]=0
#             else:
#                 lnYij1[i,j]=np.log(Y[i,j])#循环遍历取对数
#     ej1=-k*np.sum(Y*lnYij1,axis=0)#计算正向指标标准化熵值ej1
#     weights1=(1-ej1)/(cols-np.sum(ej1))#正向指标权重weights1
#     return weights1

# @app.route('/api/risk_evaluate',methods=['GET'])
# def risk_evaluate():
#     matfirstlayer=os.path.join(DIR_PATH,"communications",'cengci1.txt')
#     w0,cr0=cengci(matfirstlayer)#计算第一层权重和CR值
#     matx=os.path.join(DIR_PATH,"communications",'cengci2_1.txt')
#     w2_1,cr2_1=cengci(matx)#计算第二层中X部分的权重和CR值
#     matim=os.path.join(DIR_PATH,"communications",'cengci2_2.txt')
#     w2_2,cr2_2=cengci(matim)#计算第二层中IM部分的权重和CR值
#     matic=os.path.join(DIR_PATH,"communications",'cengci2_3.txt')
#     w2_3,cr2_3=cengci(matic)#计算第二层中IC部分的权重和CR值
#     matar=os.path.join(DIR_PATH,"communications",'cengci2_4.txt')
#     w2_4,cr2_4=cengci(matar)#计算第二层中AR部分的权重和CR值
#     matland=os.path.join(DIR_PATH,"communications",'cengci2_5.txt')
#     w2_5,cr2_5=cengci(matland)#计算第二层中着舰部分的权重和CR值
#     cengci_result=[]
#     cr_result=[cr0]
#     for i in range(len(w0)):
#         name='w2_'+str(i+1)
#         value=eval(name)
#         result=w0[i]*value
#         cengci_result=np.append(cengci_result,result)#计算层次分析法权重结果
#     for i in range(len(w0)):
#         name='cr2_'+str(i+1)
#         value=eval(name)
#         cr_result=np.append(cr_result,value)#计算层次分析法CR值结果
#     trainfile=os.path.join(DIR_PATH,"communications",'riskAssessment_train_data.csv')#训练数据
#     data_train=pd.read_csv(trainfile)
#     datatrain=data_train.values
#     row=datatrain.shape[0]
#     column=datatrain.shape[1]
#     datatrainabs=np.abs(datatrain)#绝对值
#     ax_x=0#每个指标的正态隶属度归一化函数中有两个参数，定为a和b；ax_x中前一个x代表x方向偏差，后一个x代表X位置；X位置x方向偏差的正态隶属度函数中a参数
#     bx_x=1.5#X位置x方向偏差的正态隶属度函数中b参数
#     ay_x=0#X位置y方向偏差的正态隶属度函数中a参数
#     by_x=1#X位置y方向偏差的正态隶属度函数中b参数
#     az_x=1#X位置z方向偏差的正态隶属度函数中a参数
#     bz_x=1#X位置z方向偏差的正态隶属度函数中b参数
#     ax_im=0#ax_im指的是IM位置x方向偏差的正态隶属度函数中a参数
#     bx_im=1.5#IM位置x方向偏差的正态隶属度函数中b参数
#     ay_im=0#IM位置y方向偏差的正态隶属度函数中a参数
#     by_im=1#IM位置y方向偏差的正态隶属度函数中b参数
#     az_im=0#IM位置z方向偏差的正态隶属度函数中a参数
#     bz_im=1#IM位置z方向偏差的正态隶属度函数中b参数
#     ax_ic=0#IC位置x方向偏差的正态隶属度函数中a参数
#     bx_ic=1.5#IC位置x方向偏差的正态隶属度函数中b参数
#     ay_ic=0#IC位置y方向偏差的正态隶属度函数中a参数
#     by_ic=1#IC位置y方向偏差的正态隶属度函数中b参数
#     az_ic=0#IC位置z方向偏差的正态隶属度函数中a参数
#     bz_ic=1#IC位置z方向偏差的正态隶属度函数中b参数
#     ax_ar=0#AR位置x方向偏差的正态隶属度函数中a参数
#     bx_ar=1.5#AR位置x方向偏差的正态隶属度函数中b参数
#     ay_ar=0#AR位置y方向偏差的正态隶属度函数中a参数
#     by_ar=1#AR位置y方向偏差的正态隶属度函数中b参数
#     az_ar=0#AR位置z方向偏差的正态隶属度函数中a参数
#     bz_ar=1#AR位置z方向偏差的正态隶属度函数中b参数
#     avside_ar=0#AR位置侧向速度偏差的正态隶属度函数中a参数
#     bvside_ar=1#AR位置侧向速度偏差的正态隶属度函数中b参数
#     avdown_ar=0#AR位置下沉速度偏差的正态隶属度函数中a参数
#     bvdown_ar=1#AR位置下沉速度偏差的正态隶属度函数中b参数
#     aphi_ar=0#AR位置滚转角偏差的正态隶属度函数中a参数
#     bphi_ar=2#AR位置滚转角偏差的正态隶属度函数中b参数
#     atheta_ar=0#AR位置俯仰角偏差的正态隶属度函数中a参数
#     btheta_ar=1.5#AR位置俯仰角偏差的正态隶属度函数中b参数
#     avdown_fn=0#着舰位置下沉速度偏差的正态隶属度函数中a参数
#     bvdown_fn=1#着舰位置下沉速度偏差的正态隶属度函数中b参数
#     azend_fn=0#着舰位置舰尾净高偏差的正态隶属度函数中a参数
#     bzend_fn=1#着舰位置舰尾净高偏差的正态隶属度函数中b参数
#     atheta_fn=0#着舰位置俯仰角偏差的正态隶属度函数中a参数
#     btheta_fn=1#着舰位置俯仰角偏差的正态隶属度函数中b参数
#     aphi_fn=0#着舰位置滚转角偏差的正态隶属度函数中a参数
#     bphi_fn=1#着舰位置滚转角偏差的正态隶属度函数中b参数
#     apsi_fn=0#着舰位置偏航角偏差的正态隶属度函数中a参数
#     bpsi_fn=1#着舰位置偏航角偏差的正态隶属度函数中b参数
#     a=np.array([ay_x,az_x,ay_im,az_im,ay_ic,az_ic,ay_ar,az_ar,avside_ar,avdown_ar,aphi_ar,atheta_ar,avdown_fn,azend_fn,atheta_fn,aphi_fn,apsi_fn])
#     b=np.array([by_x,bz_x,by_im,bz_im,by_ic,bz_ic,by_ar,bz_ar,bvside_ar,bvdown_ar,bphi_ar,btheta_ar,bvdown_fn,bzend_fn,btheta_fn,bphi_fn,bpsi_fn])
#     #矩阵运算
#     miux = np.zeros((row, 1))
#     for j in range(row):
#         if datatrainabs[j, 17] < 0 and datatrainabs[j, 17] >= -6.1:
#             miux[j, 0] = (datatrainabs[j, 17] + 6.1) / 6.1
#         elif datatrainabs[j, 17] >= 0 and datatrainabs[j, 17] <= 6.1:
#             miux[j, 0] = (datatrainabs[j, 17] - 6.1) / -6.1
#         else:
#             miux[j, 0] = 0#落点纵向偏差的三角隶属度归一化
#     miuy = np.zeros((row, 1))
#     for j in range(row):
#         if datatrainabs[j, 18] < 0 and datatrainabs[j, 18] >= -1.52:
#             miuy[j, 0] = (datatrainabs[j, 18] + 1.52) / 1.52
#         elif datatrainabs[j, 18] >= 0 and datatrainabs[j, 18] <= 1.52:
#             miuy[j, 0] = (datatrainabs[j, 18] - 1.52) / -1.52
#         else:
#             miuy[j, 0] = 0#落点横向偏差的三角隶属度归一化
    
#     d1=datatrainabs[:,:17]#取训练数据前17列（指定变量顺序），第18列是落点纵向偏差，第19列是落点横向偏差
#     d2=1-np.exp(-((d1-a)/b)**2)#除落点纵向偏差和落点横向偏差外的参数用正态隶属度函数归一化
#     d3 = np.hstack((d2[:, :17], 1 - miux, 1 - miuy))
#     datatrain1 = d3
#     shangquan_result=SQ(datatrain1)#熵权法计算权重
#     sum3=0
#     quanzhong=[]
#     for i in range(len(cengci_result)):
#         sum3=sum3+cengci_result[i]*shangquan_result[i]#参数层次权重*熵权权重再求和
#     for i in range(len(cengci_result)):
#         quanzhong.append(cengci_result[i]*shangquan_result[i]/sum3)#合成归一法求最终权重，参数层次权重*熵权权重/总和
#     testfile=os.path.join(DIR_PATH,"communications",'riskAssessment_test_data.csv')#测试数据
#     data_test=pd.read_csv(testfile)
#     datatest=data_test.values
#     rowtest=datatest.shape[0]
#     coltest=datatest.shape[1]
#     datatestabs=np.abs(datatest)
#     miuxt = np.zeros((rowtest, 1))
#     for j in range(rowtest):
#         if datatestabs[j, 17] < 0 and datatestabs[j, 17] >= -6.1:
#             miuxt[j, 0] = (datatestabs[j, 17] + 6.1) / 6.1
#         elif datatestabs[j, 17] >= 0 and datatestabs[j, 17] <= 6.1:
#             miuxt[j, 0] = (datatestabs[j, 17] - 6.1) / -6.1
#         else:
#             miuxt[j, 0] = 0#三角隶属度归一化
#     miuyt = np.zeros((rowtest, 1))
#     for j in range(rowtest):
#         if datatestabs[j, 18] < 0 and datatestabs[j, 18] >= -1.52:
#             miuyt[j, 0] = (datatestabs[j, 18] + 1.52) / 1.52
#         elif datatestabs[j, 18] >= 0 and datatestabs[j, 18] <= 1.52:
#             miuyt[j, 0] = (datatestabs[j, 18] - 1.52) / -1.52
#         else:
#             miuyt[j, 0] = 0#三角隶属度归一化
#     d1t=datatestabs[:,:17]
#     d2t=1-np.exp(-((d1t-a)/b)**2)#正态隶属度归一化
#     d3t = np.hstack((d2t[:, :17], 1 - miuxt, 1 - miuyt))
#     datatest1 = d3t
#     result_zuhefuquan=np.zeros((rowtest,1))
#     for i in range(rowtest):
#         for j in range(coltest):
#             result_zuhefuquan[i]=datatest1[i,j]*quanzhong[j]+result_zuhefuquan[i]
#     np.savetxt(os.path.join(DIR_PATH,"communications",'riskAssessment_result.csv'),result_zuhefuquan,delimiter=',')
#     print(result_zuhefuquan)#测试数据的评估结果
#     result_zuhefuquan=result_zuhefuquan.tolist()
#     re_rounded=round(result_zuhefuquan[0][0],3)
#     quanzhong_str=[]
#     for i in range(len(quanzhong)):
#         quanzhong_str.append(str(quanzhong[i]))
#     return jsonify({'risk_score':json.dumps(re_rounded),
#                         'quanzhong_final':json.dumps(quanzhong_str)})
# @app.route('/api/copy_file_cengci1',methods=['POST'])
# def copy_file_cengci1():
#     target_directory=os.path.join(DIR_PATH,"communications")
#     file=request.files['file']
#     file_name='cengci1.txt'
#     target_path=os.path.join(target_directory,file_name)
#     file.save(target_path)
#     return '文件复制成功'
# @app.route('/api/copy_file_cengci2_x',methods=['POST'])
# def copy_file_cengci2_x():
#     target_directory=os.path.join(DIR_PATH,"communications")
#     file=request.files['file']
#     file_name='cengci2_1.txt'
#     target_path=os.path.join(target_directory,file_name)
#     file.save(target_path)
#     return '文件复制成功'
# @app.route('/api/copy_file_cengci2_im',methods=['POST'])
# def copy_file_cengci2_im():
#     target_directory=os.path.join(DIR_PATH,"communications")
#     file=request.files['file']
#     file_name='cengci2_2.txt'
#     target_path=os.path.join(target_directory,file_name)
#     file.save(target_path)
#     return '文件复制成功'
# @app.route('/api/copy_file_cengci2_ic',methods=['POST'])
# def copy_file_cengci2_ic():
#     target_directory=os.path.join(DIR_PATH,"communications")
#     file=request.files['file']
#     file_name='cengci2_3.txt'
#     target_path=os.path.join(target_directory,file_name)
#     file.save(target_path)
#     return '文件复制成功'
# @app.route('/api/copy_file_cengci2_ar',methods=['POST'])
# def copy_file_cengci2_ar():
#     target_directory=os.path.join(DIR_PATH,"communications")
#     file=request.files['file']
#     file_name='cengci2_4.txt'
#     target_path=os.path.join(target_directory,file_name)
#     file.save(target_path)
#     return '文件复制成功'
# @app.route('/api/copy_file_cengci2_land',methods=['POST'])
# def copy_file_cengci2_land():
#     target_directory=os.path.join(DIR_PATH,"communications")
#     file=request.files['file']
#     file_name='cengci2_5.txt'
#     target_path=os.path.join(target_directory,file_name)
#     file.save(target_path)
#     return '文件复制成功'
# @app.route('/api/copy_file_traindata',methods=['POST'])
# def copy_file_traindata():
#     target_directory=os.path.join(DIR_PATH,"communications")
#     file=request.files['file']
#     file_name='riskAssessment_train_data.csv'
#     target_path=os.path.join(target_directory,file_name)
#     file.save(target_path)
#     return '文件复制成功'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
    # multiprocessing.Process(target=run).start()


