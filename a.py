from flask import Flask,render_template,request,Response,session, redirect, url_for,flash
import pandas  
from pandas import read_csv,cut,get_dummies,to_numeric
import os
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import cvzone
from sklearn import tree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import datasets
import numpy as np
import seaborn as sns
from werkzeug.utils import secure_filename
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from mlxtend.plotting import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import io
import nltk
from nltk.chat.util import Chat, reflections
import base64
import cv2
import mediapipe as mp# it also gives us drawing utils
import pyautogui
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
app = Flask(__name__)
app.secret_key = 'replace_this_with_a_long_random_string'

UPLOAD_FOLDER='static/upload/'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER   

users={
    "pateldarsh710@gmail.com":"hello",
}

@app.route('/dashboard')
def home():
    
    return render_template('dashboard.html')



@app.route('/',methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email] == password:
            session['email'] = email
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password')
    
    
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    
    return redirect(url_for('login'))

  
 

@app.route('/dt')
def dt():
    
    return render_template('dt.html')


@app.route('/addition')
def addition():
    
    return render_template('addition.html')

@app.route('/addition', methods=['POST'])
def result():
    first=request.form['one']
    second=request.form['two']  
    sum=first+second
    print(sum)   
    return render_template('addition.html',msg=sum)

@app.route('/np1')
def np1():
    
    return render_template('np1.html')

@app.route('/numpy',methods=['POST'])
def createarray():
    
    x=request.form['x']
    y=request.form['y']
    
    set1_array = [int(num) for num in x.split(',')]
    set2_array = [int(num) for num in y.split(',')]
    
    row = np.array([set1_array, set2_array])
    
    return render_template('np1.html',row=row)

@app.route('/sort',methods=['POST'])
def sort():
    set=request.form['z']
    set_array=[int(num) for num in set.split(',')]
    sortarray=np.array(set_array)
    sorted=np.sort(sortarray)
    
    
    return render_template('np1.html',sorted=sorted,sortarray=sortarray)

@app.route('/slice',methods=['POST'])
def slice():
    z=request.form['z']
    e=int(request.form['a'])
    f=int(request.form['b'])
    g=int(request.form['c'])
    array=[int(num) for num in z.split(',')]
    array1=np.array(array)
    slice=array[e:f:g]
      
    return render_template('np1.html',slice=slice)


@app.route('/pan')
def pd():
    
    return render_template('pan.html')

@app.route('/pan',methods=['POST'])
def pdfill():
    

    ff=request.files['file']
    df=read_csv(ff)
    head=df.head()
    tail=df.tail()
    
    z=df.isna().sum()
    a=df.describe()
    b=df.dtypes
    
    # y=request.form['y']
    # t=request.form['t']
    # df1=df.rename(columns={'t': 'y'})
    # newhead=df1.head()
    
    
    return render_template('pdop.html',x=head.to_html(),y=tail.to_html(),z=z,a=a.to_html(),b=b.to_list())

@app.route('/pan',methods=['POST'])
def pdfill1():
    
    
    
    
    ff=request.files['file']
    df=read_csv(ff)
    x=request.form['col']
    df1=df.drop('x',axis=1)
    newhead=df1.head()  
   
    return render_template('pdop.html',x=newhead.to_html())

@app.route('/seaborn')
def seaborn():
    

    return render_template("seaborn.html") 

@app.route('/normaldistribution',methods=['POST'])
def normaldistri():
    from numpy import random
    n=int(request.form['n'])
    y=int(request.form['y'])
    if y==1:
     sns.distplot(random.normal(size=n))
    else:
      sns.distplot(random.normal(size=n),hist=False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')    


    
    return render_template("seaborn.html",img4=img) 

@app.route('/binomialdistribution',methods=['POST'])    
def binomial():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    p=int(request.form['p'])
    q=int(request.form['q'])
    r=float(request.form['r'])
    sns.distplot(random.binomial(n=q, p=r, size=p), hist=True, kde=False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')    


    
    return render_template("seaborn.html",img5=img)

@app.route('/comparsion',methods=['POST'])    
def Comp():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    z=int(request.form['z'])
    q=int(request.form['q'])
    c=int(request.form['c'])
    r=float(request.form['r'])
    
    sns.distplot(random.normal(size=z), hist=False, label='normal')
    sns.distplot(random.binomial(n=c, p=r, size=z), hist=False, label='binomial')
    sns.distplot(random.poisson(lam=q, size=z), hist=False, label='poisson')
    sns.distplot(random.logistic(size=z), hist=False, label='logistic')


    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')    


    
    return render_template("seaborn.html",img8=img)

@app.route('/regression')
def reg():
    
    return render_template('regg.html')

@app.route('/regression',methods=['POST'])
def linearregression():
    
    diabetes=datasets.load_diabetes()
    head=diabetes.feature_names
    
    X=diabetes.data[:,2]
    Y=diabetes.target
    td=int(request.form['data'])/100
    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=td)
    model=linear_model.LinearRegression()
    model.fit(xtrain.reshape(-1, 1),ytrain)
    prediction=model.predict(xtest.reshape(-1, 1))
    coe=model.coef_#they are the coefficients for the features variable
    intercept=model.intercept_
    mse=mean_squared_error(ytest,prediction)
    r2=r2_score(ytest,prediction)
    
    fig, ax=plt.subplots()
    ax.scatter(xtrain,ytrain, color = 'red') # plotting the training set
    ax.plot(xtest, prediction, color = 'blue') # plotting the linear regression line
    ax.set_title('Truth or Bluff (Linear Regression)') # adding a tittle to our plot
    ax.set_xlabel('Position Level') # adds a label to the x-axis
    ax.set_ylabel('Salary') # adds a label to the y-axis
    buffer = io.BytesIO()
    fig, ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')
    
    return render_template('regg.html',head=head,mse=mse,r2=r2,img=img)



@app.route('/overall')
def overall():
    
    return render_template('general.html')

@app.route('/overall',methods=['POST'])
def general():
    
    ds=request.files['file']
    df=read_csv(ds)
#displaying the data
    head=df.head()

#exploratory data analysis
    desc=df.describe()# give information about data like min max mean etc

    null=df.isna().sum()#checking nulls in data set
    info=df.info()#We can use the info() function to list the data types within our data set

#renaming the column
    df.rename(columns={'utc_timestamp':'utc'},inplace=True)
    df.rename(columns={'IT_load_new':'load'},inplace=True)
    df.rename(columns={'IT_solar_generation':'solargenerated'},inplace=True)
    
    head1=df.head()

   
    td=int(request.form['data'])/100
    x=df['load'].mean()#finding mean of the column named load

    df['load'].fillna(x,inplace=True)#filling the null values using mean of that column

    lb=LabelEncoder()

    df['utc']=lb.fit_transform(df['utc'])

    X=df.drop('solargenerated',axis=1)
    Y=df['solargenerated']

    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=td)


    kn=KNeighborsClassifier()
    kn.fit(xtrain,ytrain)
    predictionknn=kn.predict(xtest)

    knacc=metrics.accuracy_score(predictionknn,ytest)
    knproba=kn.predict_proba(xtest)[:,1]

    pipe=make_pipeline(StandardScaler(),LogisticRegression())#StandardScaler removes the mean and scales each feature/variable to unit variance.
    pipe.fit(xtrain,ytrain)
# lr=LogisticRegression()
    predictionlr=pipe.predict(xtest)

    lracc=metrics.accuracy_score(predictionlr,ytest)
    lrproba=pipe.predict_proba(xtest)[:,1]

    dt=DecisionTreeClassifier()
    dt.fit(xtrain,ytrain)
    predictiondt=dt.predict(xtest)

    dtacc=metrics.accuracy_score(predictiondt,ytest)
    dtproba=dt.predict_proba(xtest)[:,1]

    rf=RandomForestClassifier()
    rf.fit(xtrain,ytrain)
    predictionrf=dt.predict(xtest)

    rfacc=metrics.accuracy_score(predictionrf,ytest)
    rfproba=rf.predict_proba(xtest)[:,1]#positve


    fpr1, tpr1, thresholds1 = roc_curve(ytest,knproba,pos_label=1)
    fpr2, tpr2, thresholds2 = roc_curve(ytest,lrproba, pos_label=1)
    fpr3, tpr3, thresholds3 = roc_curve(ytest,dtproba, pos_label=1)
    fpr4, tpr4, thresholds4 = roc_curve(ytest,rfproba, pos_label=1)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)
    fig,ax = plt.subplots()
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr1, tpr1, label ='knn =%0.2f'% roc_auc1)
    ax.plot(fpr2, tpr2, label='lr = %0.2f' % roc_auc2)
    ax.plot(fpr3, tpr3, label='dt = %0.2f' % roc_auc3)
    ax.plot(fpr4, tpr4, label='rf = %0.2f' % roc_auc4)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')
    
    return render_template('general.html',img=img,head=head.to_html(),head1=head1.to_html(),desc=desc.to_html(),null=null)



UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/f')
def f():
   
    return render_template('knnform.html')

@app.route('/f', methods=['POST'])
def knn():#KNN ALGORITHM                                           
    
    ds=request.files['file']
    filename=secure_filename(ds.filename)
    ds.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    td=int(request.form['data'])/100
    k=int(request.form['vv'])
    target=int(request.form['target'])
    # print(td)
    df=read_csv(path)
   
    
    # obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    # for col in obj_cols:
    #     df[col] = to_numeric(df[col], errors='coerce')
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    head=df.head()
    le= LabelEncoder()
    
    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))
    
    for i in object_cols_float:
        df[i] = le.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]
    
    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')
    
    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    
    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=td)
    
    if len(X_train) == 0:
        return "Error: Training set is empty."
    
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train,Y_train)
    Y_predict = KNN.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = KNN.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_knn = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_KNN = %0.2f' % auc_knn)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')
    
   
    
        
    
    
      
    return render_template('result.html',msg=acc,msg2=fscore,img_data=img_cm,img1=img_roc,head=head.to_html())   
@app.route('/rf')
def rff():
    
    return render_template('rfform.html')   
@app.route('/dt', methods=['POST'])
def random_forest_classification():
    
    ds=request.files['file']
    filename=secure_filename(ds.filename)
    ds.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    td=int(request.form['data'])/100
    # k=int(request.form['vv'])
    target=int(request.form['target'])
    # print(td)
    df=read_csv(path)
   
    
    # obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    # for col in obj_cols:
    #     df[col] = to_numeric(df[col], errors='coerce')
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    head=df.head()
    le= LabelEncoder()
    
    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))
    
    for i in object_cols_float:
        df[i] = le.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]
    
    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')
    
    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    
    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=td)
    
    # if len(X_train) == 0:
    #     return "Error: Training set is empty."
    
    RF= RandomForestClassifier(n_estimators=5)
    RF.fit(X_train,Y_train)
    Y_predict = RF.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="pink", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = RF.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_rf = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_RF = %0.2f' % auc_rf)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')
    
    

    return render_template('result.html', msg=acc,msg2=fscore,img_data=img_cm,img1=img_roc)

@app.route('/tree')
def tree():
    
    return render_template('tree.html')

@app.route('/tree', methods=['POST'])
def dtree():#RANDOM FOREST
    ds=request.files['file']
    filename=secure_filename(ds.filename)
    ds.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    td=int(request.form['data'])/100
    k=int(request.form['vv'])
    target=int(request.form['target'])
    # print(td)
    df=read_csv(path)
   
    
    # obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    # for col in obj_cols:
    #     df[col] = to_numeric(df[col], errors='coerce')
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    head=df.head()
    le= LabelEncoder()
    
    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))
    
    for i in object_cols_float:
        df[i] = le.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]
    
    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')
    
    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    
    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=td)
    
    if len(X_train) == 0:
        return "Error: Training set is empty."
    
    DT= DecisionTreeClassifier(max_depth=k)
    DT.fit(X_train,Y_train)
    Y_predict = DT.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="orange", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = DT.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_dt = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_DT = %0.2f' % auc_dt)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')
 
    
    return render_template('result.html',msg=acc,msg2=fscore,img_data=img_cm,img1=img_roc)

@app.route('/dt', methods=['POST'])
def svm():#RANDOM FOREST
    ds=request.files['file']
    filename=secure_filename(ds.filename)
    ds.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    td=int(request.form['data'])/100
 
    target=int(request.form['target'])
    # print(td)
    df=read_csv(path)
   
    
    # obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    # for col in obj_cols:
    #     df[col] = to_numeric(df[col], errors='coerce')
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    head=df.head()
    le= LabelEncoder()
    
    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))
    
    for i in object_cols_float:
        df[i] = le.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]
    
    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')
    
    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    
    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=td)
    
    if len(X_train) == 0:
        return "Error: Training set is empty."
    
    svm=SVC(probability=True)
    svm.fit(X_train,Y_train)
    Y_predict = svm.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="blue", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = svm.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_svm = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_svm = %0.2f' % auc_svm)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')
    
    # ds=request.files['file']
    # filename=secure_filename(ds.filename)
    # ds.save(os.path.join(UPLOAD_FOLDER,filename))
    # path=os.path.join(UPLOAD_FOLDER,filename)
    # td=int(request.form['data'])/100
    # # x=int(request.form['vv'])
    # # print(td)
    # df=read_csv(path)

    # l= LabelEncoder()
    # df['variety']=l.fit_transform(df['variety'])


    # X=df.values[:,0:4]
    # Y=df.values[:,4]

    # xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=td)

    # svm=SVC(probability=True)

    
    # svm.fit(xtrain,ytrain)
    # prediction=svm.predict(xtest)
    # print(prediction)
    # x=metrics.accuracy_score(ytest,prediction)
    # print(x)
    # y=f1_score(ytest,prediction,average='weighted')
    
    # cm=metrics.confusion_matrix(ytest,prediction)
    # fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4,4), cmap=plt.cm.Greens)
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # buffer = io.BytesIO()
    # fig, ax.figure.savefig(buffer, format="png")
    # buffer.seek(0)
    # image_memory = base64.b64encode(buffer.getvalue())
    # imgcm=image_memory.decode('utf-8')
    
    # probs=svm.predict_proba(xtest)

    # probs = probs[:, 1]
    
    # fpr, tpr, thresholds = roc_curve(ytest,probs,pos_label=1)
    # print(tpr)
    # roc_auc = auc(fpr, tpr)
    # print(roc_auc)
    # f,ax=plt.subplot(figsize=(5,5))
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # buffer = io.BytesIO()

    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # ax.figure.savefig(buffer,format='png')
    # buffer.seek(0)
    # image_memory=base64.b64encode(buffer.getvalue())
    # imgroc=image_memory.decode('utf-8')
    
    return render_template('result.html',msg=acc,msg2=fscore,img_data=img_cm,img1=img_roc)

@app.route('/form')
def logistic():
    
    return render_template('logiform.html')
@app.route('/lg', methods=['POST'])
def logisticalgo():
    ds=request.files['file']
    filename=secure_filename(ds.filename)
    ds.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    td=int(request.form['data'])/100
    # k=int(request.form['vv'])
    target=int(request.form['target'])
    # print(td)
    df=read_csv(path)
   
    
    # obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    # for col in obj_cols:
    #     df[col] = to_numeric(df[col], errors='coerce')
    df.fillna(method='ffill')
    df.fillna(method='bfill')
    df.dropna(inplace=True)
    head=df.head()
    le= LabelEncoder()
    
    p = (df.dtypes == 'object')
    q = (df.dtypes == 'bool')
    object_cols_float = list(p[p].index)
    object_cols_float.extend(list(q[q].index))
    
    for i in object_cols_float:
        df[i] = le.fit_transform(df[i])

    s = (df.dtypes == 'int64')
    t = (df.dtypes == 'float64')
    object_cols = list(s[s].index)
    object_cols.extend(list(t[t].index))
    object_cols.extend(object_cols_float)
    x = object_cols[target]
    
    to_drop = []
    cor = df.corr()
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    corr = image_memory.decode('utf-8')
    
    cor_target = abs(cor[x])
    relevant_features = cor_target[cor_target < 0.3]
    irr = relevant_features.to_frame()
    irr.reset_index(drop=False, inplace=True)

    for i in irr['index']:
        to_drop.append(i)

    df = df.drop(to_drop, axis = 1)

    
    Y = df[x]
    X = df.drop(x,axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=td)
    
    if len(X_train) == 0:
        return "Error: Training set is empty."
    
    LR= LogisticRegression()
    LR.fit(X_train,Y_train)
    Y_predict = LR.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="blue", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = LR.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_lr = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_LR = %0.2f' % auc_lr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')
    
    # ds=request.files['file']
    # filename=secure_filename(ds.filename)
    # ds.save(os.path.join(UPLOAD_FOLDER,filename))
    # path=os.path.join(UPLOAD_FOLDER,filename)
    # td=int(request.form['data'])/100
    # # x=int(request.form['vv'])
    # # print(td)
    # df=pd.read_csv(path)

    # l= LabelEncoder()
    # df['variety']=l.fit_transform(df['variety'])


    # X=df.values[:,0:4]
    # Y=df.values[:,4]

    # xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=td)

    # lr=LogisticRegression()

    
    # svm.fit(xtrain,ytrain)
    # prediction=lr.predict(xtest)
    # print(prediction)
    # x=metrics.accuracy_score(ytest,prediction)
    # print(x)
    # y=f1_score(ytest,prediction,average='weighted')
    
    # cm=metrics.confusion_matrix(ytest,prediction)
    # fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4,4), cmap=plt.cm.Greens)
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # buffer = io.BytesIO()
    # fig, ax.figure.savefig(buffer, format="png")
    # buffer.seek(0)
    # image_memory = base64.b64encode(buffer.getvalue())
    # imgcm=image_memory.decode('utf-8')
    # probs=lr.predict_proba(xtest)

    # probs = probs[:, 1]
    
    # fpr, tpr, thresholds = roc_curve(ytest,prediction,pos_label=1)
    # print(tpr)
    # roc_auc = auc(fpr, tpr)
    # print(roc_auc)
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')

    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # buffer2 = io.BytesIO()
    # ax.figure.savefig(buffer2,format='png')
    # buffer.seek(0)
    # image_memory=base64.b64encode(buffer2.getvalue())
    # imgroc=image_memory.decode('utf-8')
    
     
    
    return render_template('result.html',msg=acc,msg2=fscore,img_data=img_cm,img1=img_roc)


@app.route('/matplotlib')
def matplotlib():
   
    
    return render_template('mat.html')

@app.route('/create-plot', methods=['POST'])
def createplot():
   
    pldata1=request.form['pldata1']
    pldata2=request.form['pldata2']
    ls=request.form['ls']
    lw=request.form['lw']
    color=request.form['color']
    marker=request.form['marker']
    pldata1_array = [int(num) for num in pldata1.split(',')]
    pldata2_array = [int(num) for num in pldata2.split(',')]
    print(pldata1_array)
    ax= plt.plot(pldata1_array,pldata2_array,c=color,ls=ls,marker=marker,lw=lw)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    imgpl = image_memory.decode('utf-8')
           
    return render_template('matplotoutput.html',image=imgpl)
  
@app.route('/create-scatterplot', methods=['POST'])
def createsplot():
   
    xdata=request.form['xdata']
    ydata=request.form['ydata']
    colorr=request.form['colorr']
    cmap=request.form['cmap']
    alpha=float(request.form['alpha'])
    xdataarray = [int(num) for num in xdata.split(',')]
    ydataarray = [int(num) for num in ydata.split(',')]
    colorr = [int(num) for num in colorr.split(',')]
    colrr=np.array(colorr)
    size=request.form['size']
    size = [int(num) for num in size.split(',')]
    ax= plt.scatter(xdataarray,ydataarray,c=colrr,s=size,cmap=cmap,alpha=alpha)
    plt.colorbar()
    plt.xticks(rotation=90)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    imgspl = image_memory.decode('utf-8')
           
    return render_template('matplotoutput.html',image=imgspl)
   
@app.route('/create-pie', methods=['POST'])
def createpie():
   
    xdata=request.form['datapi']
    colorr=request.form['colorpi']
    labels=request.form['labels']
   
    datapi = [int(num) for num in xdata.split(',')]
    colorr = [num for num in colorr.split(',')]
    labels = [num for num in labels.split(',')]
    explode=request.form['explode']
    explode = [float(num) for num in explode.split(',')]
    ax= plt.pie(datapi,colors=colorr,labels=labels,explode=explode,autopct='%1.1f%%',counterclock='false',shadow='true')
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    imgspl = image_memory.decode('utf-8')
           
    return render_template('matplotoutput.html',image=imgspl)

@app.route('/create3d',methods=['POST'])
def create3d():
    x=int(request.form['x'])
    y=int(request.form['y'])
    z=int(request.form['z'])
    x_data2=np.arange(x,y,z)
    y_data2=np.arange(x,y,z)
    ax=plt.axes(projection="3d")
    x,y=np.meshgrid(x_data2,y_data2)
    z=np.sin(x)*np.cos(y)
    ax.plot_surface(x,y,z,cmap="plasma")
    buffer67 = io.BytesIO()
    plt.savefig(buffer67, format="png")
    buffer67.seek(0)
    image_memory = base64.b64encode(buffer67.getvalue())
    im13 = image_memory.decode('utf-8')
 
    return render_template('matplotoutput.html',image=im13)

@app.route('/values')
def values():
   
    
    return render_template('values.html')

@app.route('/values',methods=['POST'])
def mat():
    x=request.form['vv']
    set1_array = [int(num) for num in x.split(',')]
    
    y=request.form['xx']
    set2_array = [int(num) for num in y.split(',')]
    
    fig, ax = plt.subplots()
    ax.set_title('My Chart Title')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    
    ax.bar(set1_array, set2_array)
    
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    bar=image_memory.decode('utf-8')
    
    fig, ax1 = plt.subplots()
    ax1.plot(set1_array, set2_array)
    buffer = io.BytesIO()
    ax1.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    line=image_memory.decode('utf-8')
    
    fig, ax2 = plt.subplots()
    ax2.scatter(set1_array, set2_array)
    buffer = io.BytesIO()
    ax2.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    scatter=image_memory.decode('utf-8')
    
    # fig, ax3 = plt.subplots()
    # ax3.(set1_array, set2_array)
    # buffer = io.BytesIO()
    # ax3.figure.savefig(buffer, format="png")
    # buffer.seek(0)
    # image_memory = base64.b64encode(buffer.getvalue())
    # areaplot=image_memory.decode('utf-8')
    
    return render_template('r.html',bar=bar,line=line,scatter=scatter)

@app.route('/values',methods=['POST'])
def linegraph():
    x1=request.form['v']
    set1_array1 = [int(num) for num in x1.split(',')]
    
    y1=request.form['x']
    set2_array1 = [int(num) for num in y1.split(',')]
    
    fig, ax = plt.subplots()
    ax.set_title('My Chart Title')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    
    ax.plot(set1_array1, set2_array1)
    
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    line=image_memory.decode('utf-8')
    
    
    
    
    
#     
   
   
    
    return render_template('r.html',line=line)



@app.route('/vision')
def vision():
    
    return render_template('vision.html')


@app.route('/vision',methods=['POST'])
def cv():
    hands_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screenwidth, screenheight = pyautogui.size()
    index_y = 0
    
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hands_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks
        
        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand)
                landmarks = hand.landmark 
                for id, landmarks in enumerate(landmarks):
                    x = int(landmarks.x * frame_width)
                    y = int(landmarks.y * frame_height)
                    
                    if id == 8:
                        cv2.circle(img=frame, center=(x,y), radius=10, color=(0,255,255))
                        index_x = screenwidth/frame_width*x
                        index_y = screenheight/frame_height*y
                        pyautogui.moveTo(index_x, index_y)
                        
                    if id == 4:
                        cv2.circle(img=frame, center=(x,y), radius=10, color=(0,255,255))
                        thumb_x = screenwidth/frame_width*x
                        thumb_y = screenheight/frame_height*y
                        if abs(index_y - thumb_y) < 20:
                            pyautogui.click()
                            pyautogui.sleep(1)
                            
        cv2.imshow('virtual mouse', frame)
        if cv2.waitKey(10) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return render_template('vision.html')
# def cv():
#     cap=cv2.VideoCapture(0)
#     hand_detector=mp.solutions.hands.Hands()# It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame.
#     drawing_utils=mp.solutions.drawing_utils
#     screenwidth,screenheight=pyautogui.size()
#     index_y=0

#     # video camera
#     while True:
#         _, frame= cap.read()#for the frame of the video read
#         frame=cv2.flip(frame,1)# to flip the frame or screen when the camera starts beacuse in default we get inverse image
#         frame_height,frame_width,_=frame.shape
        
        
        
#         rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#converting the color in rgb form using this method
        
#         output=hand_detector.process(rgb_frame)
#         hands= output.multi_hand_landmarks#land mark is nothing but they are given points on your hand(we total have 20 landmaeks in our hands)
#         # print(hands)
        
#         if hands:
#             for hand in hands:
#                 drawing_utils.draw_landmarks(frame,hand)# this will give use the point/landmark on our hand
#                 landmarks=hand.landmark 
#                 for id, landmarks in enumerate(landmarks):#enumerate is used for numeric representation
#                  x=int(landmarks.x*frame_width)#x axis
#                  y=int(landmarks.y*frame_height)#y axis
#                 #    print(x, y)
                
#                 if  id==8:#(this is finger tip)
#                     cv2.circle(img=frame, center=(x,y), radius=10,color=(0,255,255))  # it will mark your index finger    
#                     # now using this finger as a mouse
                    
#                     index_x=screenwidth/frame_width*x#computerscreen width/frame width
#                     index_y=screenheight/frame_height*y
#                     # thus this will increase your finger tips reach in the windows screen
#                     pyautogui.moveTo(index_x, index_y)  # this will move your mouse on nyour own but for small area only so we will incrase tthe screen size
#                     # now when we touch our thumb with first finger it will perform cloick
#                 if  id==4:#(this is thumb tip)
#                     cv2.circle(img=frame, center=(x,y), radius=10,color=(0,255,255))     
                    
                    
#                     thumb_x=screenwidth/frame_width*x
#                     thumb_y=screenheight/frame_height*y
                    
#                     print('outside',abs(index_y-thumb_y))
                    
#                     if abs(index_y-thumb_y)< 20:#if abs is the absolute difference
#                         print('click')# so if the condition staisfies it clicks
#                         pyautogui.click()
#                         pyautogui.sleep(1)#sleep with oone sceoond
#         cv2.imshow('virtual mouse',frame)# showing that is dispaklying the image
#         cv2.waitKey(1)
    
#     return render_template('vision.html')

@app.route('/kclus')
def kclus():
   
    
    return render_template('kclus.html')

@app.route('/kclus', methods=['POST'])
def kmean():
    
    ds=request.files['file']
    
    # td=int(request.form['data'])/100
    # x=int(request.form['vv'])
    # print(td)
    df=read_csv(ds)
    l= LabelEncoder()
    df['variety']=l.fit_transform(df['variety'])
    
    t=df['variety']


    
    
    
    wcss = []

    for i in range(1,11):
     km = KMeans(n_clusters=i)
     km.fit_predict(df)
     wcss.append(km.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(range(1,11),wcss)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    lime=image_memory.decode('utf-8')

    X=df.drop('variety',axis=1)
    km=KMeans(n_clusters=3)
    ymeans=km.fit_predict(X)
    
    x=ymeans
    print(x)
    


    
    X['cluster_label'] = km.labels_
    
    fig, ax2 = plt.subplots()
    ax2.scatter(X["petal.length"], X["petal.width"], c=X['cluster_label'], cmap='rainbow')
    ax2.set_xlabel('Petal Length (cm)')
    ax2.set_ylabel('Petal Width (cm)')
    buffer = io.BytesIO()
    ax2.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    scatter=image_memory.decode('utf-8')

    
    
    cp=sns.countplot(x=ymeans)
    cp.set_title('number of clusters') 
    buffer = io.BytesIO()
    cp.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    count=image_memory.decode('utf-8')
    
    
    
     
        
    
    
      
    return render_template('kclus.html',lime=lime,scatter=scatter,count=count,t=t,x=x)  

@app.route('/heiarchical')
def hei():
   
    
    return render_template('heiarchical.html')

@app.route('/heiarchical', methods=['POST'])
def heiarchical():
    ds=request.files['file']
    filename=secure_filename(ds.filename)
    ds.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    
    
    df=read_csv(path)
    
    print(df.describe())

    print(df.info())

    print(df.head())

#Encoding Variables and Feature Engineering
#Let's start by dividing the Age into groups that vary in 10, so that we have 20-30, 30-40, 40-50, and so on. Since our youngest customer is 15, we can start at 15 and end at 70, which is the age of the oldest customer in the data. Starting at 15, and ending at 70, we would have 15-20, 20-30, 30-40, 40-50, 50-60, and 60-70 intervals.

    intervals = [15, 20, 30, 40, 50, 60, 70]
    col = df['Age']
    df['Age Groups'] = pandas.cut(x=col, bins=intervals)
    df1 = pandas.get_dummies(df)
    #At the moment, we have two categorical variables, Age and Genre, which we need to transform into numbers to be able to use in our model. There are many different ways of making that transformation - we will use the Pandas get_dummies() method that creates a new column for each interval and genre and then fill its values with 0s and 1s- this kind of operation is called one-hot encoding.
    a=sns.scatterplot(x=df['Annual Income (k$)'],y=df['Spending Score (1-100)'])
    a.set_title('number of clusters') 
    buffer = io.BytesIO()
    a.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    scatter1=image_memory.decode('utf-8')
# now trying to make clusters using pca
    df1 = df1.drop(['Age'], axis=1)
    pca = PCA(n_components=10)
    pca.fit_transform(df1)
    pca.explained_variance_ratio_.cumsum()

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df1)

    pc1_values = pcs[:,0]
    pc2_values = pcs[:,1]
    z=sns.scatterplot(x=pc1_values, y=pc2_values)
    z.set_title('number of clusters') 
    buffer = io.BytesIO()
    z.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    pcascatter1=image_memory.decode('utf-8')
    # plt.show()

    
    
    fig, ax = plt.subplots()
    ax.set_title("customer dendrogram")
    # ax.figure(figsize=(10, 7))
# Selecting Annual Income and Spending Scores by index
    selected_data = df1.iloc[:, 1:3]
    clusters = shc.linkage(selected_data,
            method='ward',
            metric="euclidean")
    shc.dendrogram(Z=clusters)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    dendrogram=image_memory.decode('utf-8')
    
#Implementing an Agglomerative Hierarchical Clustering
    clustering_model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    clustering_model.fit(selected_data)
    print(clustering_model.labels_)

    data_labels = clustering_model.labels_
    b=sns.scatterplot(x='Annual Income (k$)',
                y='Spending Score (1-100)',
                data=selected_data,
                hue=data_labels,
                palette="rainbow")
    b.set_title('Labeled Customer Data')
    # b.set_title('number of clusters') 
    buffer2 = io.BytesIO()
    b.figure.savefig(buffer2, format="png")
    buffer2.seek(0)
    image_memory = base64.b64encode(buffer2.getvalue())
    opscatter1=image_memory.decode('utf-8')

    

#This is our final clusterized data. You can see the color-coded data points in the form of five clusters.
#
# The data points in the bottom right (label: 0, purple data points) belong to the customers with high salaries but low spending. These are the customers that spend their money carefully.
#
# Similarly, the customers at the top right (label: 2, green data points), are the customers with high salaries and high spending. These are the type of customers that companies target.
#
# The customers in the middle (label: 1, blue data points) are the ones with average income and average spending. The highest numbers of customers belong to this category. Companies can also target these customers given the fact that they are in huge numbers.
#
# The customers in the bottom left (label: 4, red) are the customers that have low salaries and low spending, they might be attracted by offering promotions.
#
# And finally, the customers in the upper left (label: 3, orange data points) are the ones with high income and low spending, which are ideally targeted by marketing.
    clustering_model_pca = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    clustering_model_pca.fit(pcs)

    data_labels_pca = clustering_model_pca.labels_

    c=sns.scatterplot(x=pc1_values,
                y=pc2_values,
                hue=data_labels_pca,
                palette="rainbow")
    c.set_title('Labeled Customer Data Reduced with PCA')
    buffer = io.BytesIO()
    c.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    oppscatter2=image_memory.decode('utf-8')
    
    
    return render_template('heiarchical.html',scatter1=scatter1,pcascatter1=pcascatter1,opscatter1=opscatter1,oppscatter2=oppscatter2,dendrogram=dendrogram)  

@app.route('/table')
def table():
    
    
    return render_template('table.html')

@app.route('/table',methods=['POST'])
def fill():
    
    r=request.files['file']
    
    df=read_csv(r)
    head=df.head()
    
    return render_template('table1.html',x=head.to_html())
    
@app.route('/table1',methods=['POST'])
def graph():
    
    
    ds=request.files['file']
    filename=secure_filename(ds.filename)
    ds.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    df=read_csv(path)
    
    
    x=request.form['vv']
    set1_array = [str(text) for text in x.split(',')]
    #['sepal.width','sepal.length','petal.length','petal.width']
    a=sns.pairplot(df,hue='variety',palette="muted",size=5,vars=set1_array,kind='scatter')
    buffer = io.BytesIO()
    a.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    imgsns=image_memory.decode('utf-8')
    
    c=sns.FacetGrid(df,col='variety')
    c.map(plt.scatter,'sepal.length','sepal.width')
    c.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory1 = base64.b64encode(buffer.getvalue())
    imgsns1=image_memory1.decode('utf-8')
    
    ax = sns.swarmplot(x='variety', y='sepal.length', data=df)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    imgsns2=image_memory.decode('utf-8')
    
    
    
    
    
    return render_template('op.html',imgsns=imgsns,imgsns1=imgsns1,imgsns2=imgsns2)

@app.route('/goggle')
def goggle():
    
    
    return render_template('goggle.html')

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/goggle',methods=['POST'])
def sunglasses():
       
        
        num=1

        while True:
            # k=cv2.waitKey(100)
            # if k==ord('s'):
            #     num=num+1
            #print(num)    
            if(num<=29):
                    overlay = cv2.imread('Glasses/glass4.png'.format(num), cv2.IMREAD_UNCHANGED)
                
            _, frame = cap.read()
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale)
            for (x, y, w, h) in faces:
                #cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
                overlay_resize = cv2.resize(overlay,(w,int(h*0.8)))
                frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
            cv2.imshow('SnapLens', frame)
            if cv2.waitKey(10) == ord('q') or num>29:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return render_template('goggle.html')



@app.route('/goggle2')
def goggle4():
    
    
    return render_template('goggle.html')
@app.route('/goggle2',methods=['POST'])
def sunglasses2():
        
        cap = cv2.VideoCapture(0)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num=1

        while True:
            k=cv2.waitKey(100)
            if k==ord('s'):
                num=num+1
            #print(num)    
            if(num<=29):
                    overlay = cv2.imread('Glasses/glass5.png'.format(num), cv2.IMREAD_UNCHANGED)
                
            _, frame = cap.read()
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale)
            for (x, y, w, h) in faces:
                #cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
                overlay_resize = cv2.resize(overlay,(w,int(h*0.8)))
                frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
            cv2.imshow('SnapLens', frame)
            if cv2.waitKey(10) == ord('q') or num>29:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return render_template('goggle.html')


@app.route('/goggle3')
def goggle3():
    
    
    return render_template('goggle.html')   
@app.route('/goggle3',methods=['POST'])
def sunglasses3():
        
        cap = cv2.VideoCapture(0)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num=1

        while True:
            k=cv2.waitKey(100)
            if k==ord('s'):
                num=num+1
            #print(num)    
            if(num<=29):
                    overlay = cv2.imread('Glasses/glass8.png'.format(num), cv2.IMREAD_UNCHANGED)
                
            _, frame = cap.read()
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale)
            for (x, y, w, h) in faces:
                #cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
                overlay_resize = cv2.resize(overlay,(w,int(h*0.8)))
                frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
            cv2.imshow('SnapLens', frame)
            if cv2.waitKey(10) == ord('q') or num>29:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return render_template('goggle.html')


  
@app.route('/goggle4')
def goggle5():
    
    
    return render_template('goggle.html')    
@app.route('/goggle5',methods=['POST'])
def sunglasses4():
        
        cap = cv2.VideoCapture(0)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num=1

        while True:
            k=cv2.waitKey(100)
            if k==ord('s'):
                num=num+1
            #print(num)    
            if(num<=29):
                    overlay = cv2.imread('Glasses/glass9.png'.format(num), cv2.IMREAD_UNCHANGED)
                
            _, frame = cap.read()
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale)
            for (x, y, w, h) in faces:
                #cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
                overlay_resize = cv2.resize(overlay,(w,int(h*0.8)))
                frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
            cv2.imshow('SnapLens', frame)
            if cv2.waitKey(10) == ord('q') or num>29:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return render_template('goggle.html')    

@app.route('/goggle10',methods=['POST'])
def sunglasses10():
        
        cap = cv2.VideoCapture(0)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num=1

        while True:
            k=cv2.waitKey(100)
            if k==ord('s'):
                num=num+1
            #print(num)    
            if(num<=29):
                    overlay = cv2.imread('Glasses/glass10.png'.format(num), cv2.IMREAD_UNCHANGED)
                
            _, frame = cap.read()
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale)
            for (x, y, w, h) in faces:
                #cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
                overlay_resize = cv2.resize(overlay,(w,int(h*0.8)))
                frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
            cv2.imshow('SnapLens', frame)
            if cv2.waitKey(10) == ord('q') or num>29:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return render_template('goggle.html')    
    
@app.route('/goggle24',methods=['POST'])
def sunglasses20():
        
        cap = cv2.VideoCapture(0)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num=1

        while True:
            k=cv2.waitKey(100)
            if k==ord('s'):
                num=num+1
            #print(num)    
            if(num<=29):
                    overlay = cv2.imread('Glasses/glass20.png'.format(num), cv2.IMREAD_UNCHANGED)
                
            _, frame = cap.read()
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale)
            for (x, y, w, h) in faces:
                #cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 255, 0), 2)
                overlay_resize = cv2.resize(overlay,(w,int(h*0.8)))
                frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
            cv2.imshow('SnapLens', frame)
            if cv2.waitKey(10) == ord('q') or num>29:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return render_template('goggle.html')  


import nltk
from nltk.chat.util import Chat, reflections


pairs = [
  [
    r"what is data science",
    ["Data science is the study of data to extract meaningful insights for business. It combines principles from mathematics, statistics, and computer engineering to analyze large amounts of data."],
  ],
  [
    r"why is data science important",
    ["Data science is important because it helps organizations generate meaning from data."],
  ],
  [
    r"what are machine learning algorithms",
    ["Machine learning algorithms enable computers to learn from data by building models and making predictions."],
  ],
  [
    r"what is deep learning",
    ["Deep learning refers to artificial neural networks that simulate the function of the human brain to recognize patterns in data."],
  ],
  [
    r"what is KNN algorithm",
    ["KNN or K-Nearest Neighbours is a type of supervised machine learning algorithm used to classify new data points based on distance measure (Euclidean, Manhattan, etc.) between features in the training set."],
  ],
  [
    r"what is SVM algorithm",
    ["SVM or Support Vector Machines is a type of supervised machine learning algorithm that constructs a hyperplane in high-dimensional space to separate classification classes with maximum margin."],
  ],
  [
    r"what is decision tree algorithm",
    ["Decision tree algorithm is a type of supervised machine learning algorithm that builds a flowchart-like model of decisions and their possible consequences (outcomes) using a set of rules based on features in the training set."],
  ],
  [
    r"what is logistic regression algorithm",
    ["Logistic regression algorithm is a type of supervised machine learning algorithm used to predict the probability of categorical dependent variables based on independent variables."],
  ],
  [
    r"what is random forest algorithm",
    ["Random forest algorithm is a type of supervised machine learning algorithm that builds multiple decision trees and aggregates their predictions to improve the accuracy and avoid overfitting."],
  ],
  [
    r"what is K means clustering",
    ["K-means clustering is a type of unsupervised machine learning algorithm used to group data points into K clusters based on their similarity to centroid points."],
  ],
  [
    r"what is hierarchical clustering",
    ["Hierarchical clustering is a type of unsupervised machine learning algorithm used to group data points into nested clusters based on their similarity to each other."],
  ],
  [
    r"what is the trade-off between bias and variance",
    ["Bias is error due to erroneous or overly simplistic assumptions in the learning algorithm youre using. This can lead to the model underfitting your data, making it hard for it to have high predictive accuracy and for you to generalize your knowledge from the training set to the test set.Variance is error due to too much complexity in the learning algorithm youre using. This leads to the algorithm being highly sensitive to high degrees of variation in your training data, which can lead your model to overfit the data. Youll be carrying too much noise from your training data for your model to be very useful for your test data.The bias-variance decomposition essentially decomposes the learning error from any algorithm by adding the bias, the variance and a bit of irreducible error due to noise in the underlying dataset. Essentially, if you make the model more complex and add more variables, youll lose bias but gain some variance — in order to get the optimally reduced amount of error, youll have to tradeoff bias and variance. You dont want either high bias or high variance in your model."],
  ],
  [
    r"What is the difference between supervised and unsupervised machine learning",
    ["Supervised learning requires training labeled data. For example, in order to do classification (a supervised learning task), you’ll need to first label the data you’ll use to train the model to classify data into your labeled groups. Unsupervised learning, in contrast, does not require labeling data explicitly."],
  ],
   [
    r"What is a Confusion Matrix",
    ["The Confusion Matrix is the summary of prediction results of a particular problem. It is a table that is used to describe the performance of the model. The Confusion Matrix is an n*n matrix that evaluates the performance of the classification model."],
  ],
    [
    r"What are dimensionality reduction and its benefits",
    ["The Dimensionality reduction refers to the process of converting a data set with vast dimensions into data with fewer dimensions (fields) to convey similar information concisely. This reduction helps in compressing data and reducing storage space. It also reduces computation time as fewer dimensions lead to less computing. It removes redundant features; for example, there's no point in storing a value in two different units (meters and inches). "],
  ],
    [
    r"How do you handle missing or corrupted data in a dataset",
    ["You could find missing/corrupted data in a dataset and either drop those rows or columns, or decide to replace them with another value.In Pandas, there are two very useful methods: isnull() and dropna() that will help you find columns of data with missing or corrupted data and drop those values. If you want to fill the invalid values with a placeholder value (for example, 0), you could use the fillna() method."],
  ],
    [
    r"When should you use classification over regression",
    [" Classification produces discrete values and dataset to strict categories, while regression gives you continuous results that allow you to better distinguish differences between individual points. You would use classification over regression if you wanted your results to reflect the belongingness of data points in your dataset to certain explicit categories (ex: If you wanted to know whether a name was male or female rather than just how correlated they were with male and female names.)"],
  ],
     [
    r"What is the F1 score",
    ["The F1 score is a measure of a models performance. It is a weighted average of the precision and recall of a model, with results tending to 1 being the best, and those tending to 0 being the worst. You would use it in classification tests where true negatives dont matter much."],
  ],
    [
    r"Which is more important to you: model accuracy or model performance",
    ["Such machine learning interview questions tests your grasp of the nuances of machine learning model performance! Machine learning interview questions often look towards the details. There are models with higher accuracy that can perform worse in predictive power—how does that make sense?Well, it has everything to do with how model accuracy is only a subset of model performance, and at that, a sometimes misleading one. For example, if you wanted to detect fraud in a massive dataset with a sample of millions, a more accurate model would most likely predict no fraud at all if only a vast minority of cases were fraud. However, this would be useless for a predictive model—a model designed to find fraud that asserted there was no fraud at all! Questions like this help you demonstrate that you understand model accuracy isn’t the be-all and end-all of model performance."],
  ],
   [
    r"difference between deep and machine learning",
    ["Deep learning is a subset of machine learning that is concerned with neural networks: how to use backpropagation and certain principles from neuroscience to more accurately model large sets of unlabelled or semi-structured data. In that sense, deep learning represents an unsupervised learning algorithm that learns representations of data through the use of neural nets."],
  ],
     [
    r"Explain the difference between L1 and L2 regularization",
    ["L2 regularization tends to spread error among all the terms, while L1 is more binary/sparse, with many variables either being assigned a 1 or 0 in weighting. L1 corresponds to setting a Laplacean prior on the terms, while L2 corresponds to a Gaussian prior."],
  ],
    [
    r"Explain how a ROC curve works",
    ["The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. Its often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives)."],
  ],
    [
    r"How is KNN different from k-means clustering",
    ["K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm. While the mechanisms may seem similar at first, what this really means is that in order for K-Nearest Neighbors to work, you need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.The critical difference here is that KNN needs labeled points and is thus supervised learning, while k-means doesn’t—and is thus unsupervised learning."],
  ],
    [
    r"How to read an image in OpenCV",
    ["The imread() function is used in OpenCV to read an image. The imread function loads an image from the specified location and returns it. If the function is unable to read the image, which can happen due to a variety of reasons such as incorrect file path, unsupported format, or improper permission to work with the file, then the function returns an empty matrix."],
  ],
   [
    r"How can you resize an image in OpenCV",
    ["Resizing refers to the scaling of an image. When the resizing of an image is done, the pixel information changes. Thus, resampling of the pixels is required while reducing the size of an image whereas reconstruction of the image is required while increasing the size of an image. Reconstruction means interpolation of new pixels.The scaling size for the image can be specified manually, or scaling factor can be used. Scaling of an image in OpenCV can also be achieved using different interpolation methods."],
  ],
   [
    r"What are the different flags in OpenCV imread function",
    ["The different flags which can be set while reading an image using the imread function in OpenCV are: 1. cv2.IMREAD_COLOR: This flag specifies to load a color image. This mode neglects the transparency of the image. It is the default flag. Integer value 1 is passed for this flag.2. cv2.IMREAD_GRAYSCALE: It specifies the loading of an image in grayscale mode. The integer value 0 is passed for this flag.3. cv2.IMREAD_UNCHANGED: It specifies to include the alpha channel while loading the image. Integer value -1 is passed for this flag."],
  ],
   [
    r"What is the VideoCapture operation in OpenCV",
    ["OpenCV provides the option of reading a video interface by either capturing the live feed from the system camera or by reading a saved video file. The mode in which the video is going to be accessed is given as an argument in the VideoCapture function in OpenCV.The VideoCapture function accepts the index of the device and video file name as the arguments. The function converts the read video into grayscale mode and saves it frame by frame."],
  ],
   [
    r"What is Cascade Classifier in OpenCV",
    ["Cascade classifier is a machine learning approach where the positive and negative images are used to train a cascade function. The cascade classifier, as the name suggests, is used to classify or detect objects in images. The algorithm requires a large amount of training data images with the object to be detected and images without the object to be detected."],
  ],
   [
    r"Show me few images",
    ["<img src='/static/img/datacsience.jpg'>"],
   ],

     
            
  
  # Add more pairs/questions/answers as per your needs
  
]

chatbot = Chat(pairs, reflections)    

@app.route('/chatbot')
def chatbot111():
    
    
    return render_template('chatbot.html') 

@app.route('/chatbot', methods=["POST"])
def chatbotfun():
    user_input = request.form["user_input"]
    response = chatbot.respond(user_input)
    
   
    
    return render_template('chatbotoutput.html',response=response)    




questions = [
    "What is the capital of France?",
    "What is the highest mountain in the world?",
    "Which of the following is not a programming language?",
    "Which of the following is a supervised learning algorithm?",
    "What is the formula for calculating the area of a circle?"
]

options = [
    ['A. London', 'B. Paris', 'C. Madrid', 'D. Berlin'],
    ['A. Mount Everest', 'B. K2', 'C. Denali', 'D. Aconcagua'],
    ['A. Python', 'B. Java', 'C. C++', 'D. HTML'],
    ['A. Linear Regression', 'B. K-Means Clustering', 'C. Random Forest', 'D. SVM'],
    ['A. pi * r^2', 'B. 2 * pi * r', 'C. pi * d', 'D. 4 * pi * r^2']
]

answers = ['B', 'A', 'D', 'A', 'A']

# Define a variable to keep track of the current question number
current_question = 0

# Define a variable to keep track of the user's score
score = 0

@app.route('/quiz')
def quiz():
    global current_question
    
    current_question = 0
    
    
    if current_question < len(questions):
     return render_template('quiz.html',question=questions[current_question], options=options[current_question]) 

@app.route('/quizimp',methods=["POST"])
def quizimp():
    global current_question
    global score
    
    user_answer = request.form['answer']
    
    # Check if the user's answer is correct
    if user_answer == answers[current_question]:
        score += 1
    current_question += 1
    
    if current_question == len(questions):  
        # End of quiz, show score
        return render_template('score.html', score=score)
    else:
    # Render the quiz template with the current question and options
        return render_template('quiz.html', question=questions[current_question], options=options[current_question])
    
import cv2
import img2pdf
from werkzeug.utils import secure_filename
import os
from docx import Document   


def processimage(filename, operation):
    print(f'The operation is {operation} and file is {filename}')
    img = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))

    for case in operation:
        if case == "cgray":
            imgprocessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            new_filename=f"static/upload/{filename}"
            cv2.imwrite(new_filename, imgprocessed)
            return new_filename
        elif case == "cpng":
            new_filename=f"static/upload/{filename.split('.')[0]}.png" 
            cv2.imwrite(new_filename, img)
            return new_filename
        elif case == "cjpg":
            new_filename=f"static/upload/{filename.split('.')[0]}.jpg"
            cv2.imwrite(new_filename, img)
            return new_filename
        elif case == "cwebp":
            new_filename=f"static/upload/{filename.split('.')[0]}.webp"
            cv2.imwrite(new_filename, img)  
            return new_filename  
        elif case == "cpdf":
            new_filename = f"static/upload/{filename.split('.')[0]}.pdf"
            with open(new_filename, "wb") as f:
                f.write(img2pdf.convert(os.path.join(UPLOAD_FOLDER, filename)))
            return new_filename   
        elif case == "cdocx":
            new_filename = f"static/upload/{filename.split('.')[0]}.docx"
            doc = Document()
            doc.add_picture(os.path.join(UPLOAD_FOLDER, filename))
            doc.save(new_filename)
            return new_filename
@app.route('/imageedit')
def img():
    return render_template('imageedit.html')

@app.route('/edit', methods=['POST'])
def imgedit():
    ds = request.files['file']
    filename = secure_filename(ds.filename)
    ds.save(os.path.join(UPLOAD_FOLDER, filename))
    operation = request.form.getlist('selection')
    new=processimage(filename, operation)
    flash(f"YOUR image has been processed and is available <a href='/{new}' target='_blank'>here</a>")
    

    return render_template('imageedit.html')
# def processimage(filename, operation):
#     print(f'The operation is {operation} and file is {filename}')
#     img = cv2.imread(f"upload/{filename}")

#     for case in operation:
#         if case =="cgray":
#             imgprocessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), imgprocessed)
#         elif case =="cpng":
#             cv2.imwrite(f"static/upload/{filename.split('.')[0]}.jpg", imgprocessed)    
#         elif case =="cjpg":
#             cv2.imwrite(f"static/upload/{filename.split('.')[0]}.png", imgprocessed)       
            
# @app.route('/imageedit')
# def img():
#     return render_template('imageedit.html')     

# @app.route('/edit', methods=['POST'])
# def imgedit():
#     ds = request.files['file']
#     filename = secure_filename(ds.filename)
#     ds.save(os.path.join(UPLOAD_FOLDER, filename))
#     operation = request.form['selection']
#     processimage(filename, operation)
    
#     return render_template('imageedit.html')
# what is api?

if __name__=='__main__':#use for running the given function
    app.run(debug=True)#we can even change the port number with (.....,port=8000)py