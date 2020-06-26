import numpy as np
from flask import Flask, request, render_template, url_for
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt


GRAPH_FOLDER= os.path.join('static')
app = Flask(__name__)
app.config['upload']=GRAPH_FOLDER

x=0
y=0
def date_time(t):
            global x
            global y
            t=t + "  "+ str(y)+":"+str(x*10)+":"+"00" 
            if x<5:
                x=x+1
            else:
                x=0
                y=y+1
            return t    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    
    d=request.form['Date']
    ws=[]
    wd=[]
    m=[]
    for z in range(144):
        ws.append(np.random.uniform(0,25.5))
        wd.append(np.random.uniform(0,360))
        m.append(d)    
        
    data=np.concatenate([np.array(ws).reshape(144,1),np.array(wd).reshape(144,1),np.array(m).reshape(144,1)],1)
    df1=pd.DataFrame(data=data,columns=["Wind Speed","Wind direction","Date/Time"])
    
    
    
    df1["Date/Time"]=df1["Date/Time"].apply(date_time,1)
    global x
    x=0
    global y
    y=0 
    df1["Time"]=df1["Date/Time"].apply(lambda x:x[-8:-3],1)          
    df1["Hour"]=pd.to_timedelta(df1["Date/Time"].apply(lambda x:x[-8:],1)).dt.components["hours"]
    df1[['date','time']] = df1['Date/Time'].str.split(expand=True)
    df1['Date/Time'] = (pd.to_datetime(df1.pop('date'), format='%Y/%m/%d') + 
                  pd.to_timedelta(df1.pop('time') ))
    df1=df1.set_index("Date/Time")
    df1["Year"]=df1.index.year
    df1["Month"]=df1.index.month
    df1["Weekday"]=df1.index.weekday

    X1=df1[["Wind Speed","Wind direction"]]
    from sklearn.preprocessing import StandardScaler
    scaler1=StandardScaler()
    scaler1.fit(X1)
    scaled_data1=scaler1.transform(X1)

    df1["Theoretical_Power_Curve (KWh)"]=joblib.load('Wind Turbine(TPC).sav').predict(scaled_data1)

    X2=df1[["Wind Speed","Theoretical_Power_Curve (KWh)","Month","Hour"]]
    from sklearn.preprocessing import StandardScaler
    scaler2=StandardScaler()
    scaler2.fit(X2)
    scaled_data2=scaler2.transform(X2)

    df1["LV ActivePower (kW)"]=joblib.load('Wind Turbine(LV).sav').predict(scaled_data2)

    prediction=df1["LV ActivePower (kW)"].mean()
    fig, axes = plt.subplots(figsize=(80,20))
    x1=df1["Time"]
    y1=df1["LV ActivePower (kW)"].values
    z1=df1["Theoretical_Power_Curve (KWh)"].values
    axes.plot(x1, y1, 'b',label="LV ActivePower")
    axes.plot(x1,z1,"g",label="Theoretical_Power_Curve")
    axes.legend(fontsize=50,loc='upper right')
    plt.xlabel('Date/Time',fontsize=100)
    plt.ylabel('Power (kW)',fontsize=100)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=50)
    axes.set_xticklabels(x1,rotation=90)
    axes.set_title('Time Series Analysis')
    axes.title.set_size(100)
    plt.savefig("static/temp.png")
    

    return render_template('index.html', prediction_text=prediction)

@app.route('/display',  methods=["POST"])
def display():
    full_filename= os.path.join(app.config['upload'], 'temp.png')
    return render_template('graph.html', graph_image=full_filename )
    



if __name__ == "__main__":
    app.run(debug=True)