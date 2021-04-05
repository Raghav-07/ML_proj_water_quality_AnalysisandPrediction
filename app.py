# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:02:33 2021

@author: Lenovo
"""
 

from flask import Flask, request, render_template, redirect, url_for
app = Flask(__name__)

@app.route('/success/<name>')
def wqi(name):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    filename="E://Internship//water_dataX.csv"
    data = pd.read_csv(filename, encoding= 'unicode_escape')
    print(data.head())
    
    #conversions
    data['Temp']=pd.to_numeric(data['Temp'],errors='coerce')
    data['D.O. (mg/l)']=pd.to_numeric(data['D.O. (mg/l)'],errors='coerce')
    data['PH']=pd.to_numeric(data['PH'],errors='coerce')
    data['B.O.D. (mg/l)']=pd.to_numeric(data['B.O.D. (mg/l)'],errors='coerce')
    data['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(data['CONDUCTIVITY (µmhos/cm)'],errors='coerce')
    data['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(data['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
    data['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(data['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')
    print(data.dtypes)
    
    #initialization to remove column  FECAL COLIFORM (MPN/100ml)  and RENAME the features with simpler names
    start=2
    end=1779
    station=data.iloc [start:end ,0]
    location=data.iloc [start:end ,1]
    state=data.iloc [start:end ,2]
    do= data.iloc [start:end ,4].astype(np.float64)
    value=0
    ph = data.iloc[ start:end,5]  
    co = data.iloc [start:end ,6].astype(np.float64)   
      
    year=data.iloc[start:end,11]
    tc=data.iloc [2:end ,10].astype(np.float64)
    
    
    bod = data.iloc [start:end ,7].astype(np.float64)
    na= data.iloc [start:end ,8].astype(np.float64)
    print(na.dtype)
    
    data=pd.concat([station,location,state,do,ph,co,bod,na,tc,year],axis=1)
    data. columns = ['station','location','state','do','ph','co','bod','na','tc','year']
    print(data.head())
    
    #adding new Parameters according to their “WHO” standard limits from range(0-100) for WQI calculations(q_value).
    #Q VALUE NORMALIZATION
    
    
    #calulation of Ph
    data['npH']=data.ph.apply(lambda x: (100 if (8.5>=x>=7)  
                                     else(80 if  (8.6>=x>=8.5) or (6.9>=x>=6.8) 
                                          else(60 if (8.8>=x>=8.6) or (6.8>=x>=6.7) 
                                              else(40 if (9>=x>=8.8) or (6.7>=x>=6.5)
                                                  else 0)))))
    
    #calculation of dissolved oxygen
    data['ndo']=data.do.apply(lambda x:(100 if (x>=6)  
                                     else(80 if  (6>=x>=5.1) 
                                          else(60 if (5>=x>=4.1)
                                              else(40 if (4>=x>=3) 
                                                  else 0)))))
    #calculation of total coliform
    data['nco']=data.tc.apply(lambda x:(100 if (5>=x>=0)  
                                     else(80 if  (50>=x>=5) 
                                          else(60 if (500>=x>=50)
                                              else(40 if (10000>=x>=500) 
                                                  else 0)))))
    #calc of B.O.D
    data['nbdo']=data.bod.apply(lambda x:(100 if (3>=x>=0)  
                                     else(80 if  (6>=x>=3) 
                                          else(60 if (80>=x>=6)
                                              else(40 if (125>=x>=80) 
                                                  else 0)))))
    
    #calculation of electrical conductivity
    data['nec']=data.co.apply(lambda x:(100 if (75>=x>=0)  
                                     else(80 if  (150>=x>=75) 
                                          else(60 if (225>=x>=150)
                                              else(40 if (300>=x>=225) 
                                                  else 0)))))
    #Calulation of nitrate
    data['nna']=data.na.apply(lambda x:(100 if (20>=x>=0)  
                                     else(80 if  (50>=x>=20) 
                                          else(60 if (100>=x>=50)
                                              else(40 if (200>=x>=100) 
                                                  else 0)))))
    
    data.head()
    print(data.dtypes)
    
    
    #qvalue reflects the value of a parameter in the range of 0–100 and w_ f actor represents
    #the weight of a particular parameter as listed in Table 2. WQI is fundamentally calculated by initially
    #multiplying the q value of each parameter by its corresponding weight, adding them all up and then
    #dividing the result by the sum of weights of the employed parameters
    
    
    #we add new columns now by multiplying values to weights
    data['wph']=data.npH * 0.165
    data['wdo']=data.ndo * 0.281
    data['wbdo']=data.nbdo * 0.234
    data['wec']=data.nec* 0.009
    data['wna']=data.nna * 0.028
    data['wco']=data.nco * 0.281
    data['wqi']=data.wph+data.wdo+data.wbdo+data.wec+data.wna+data.wco 
    print(data.head())
    
    #calculation overall wqi for each year
    ag=data.groupby('year')['wqi'].mean()
    
    data=ag.reset_index(level=0,inplace=False)
    print(data)
    
    data = data[np.isfinite(data['wqi'])]
    print(data.head())
    
    #scatter plot of data points
    cols =['year']
    y = data['wqi']
    x=data[cols]
    
    plt.scatter(x,y)
    #plt.show()
    
    import matplotlib.pyplot as plt
    data=data.set_index('year')
    data.plot(figsize=(15,6))
    #plt.show()
    
    from sklearn import neighbors,datasets
    data=data.reset_index(level=0,inplace=False)
    data
    
    #using linear regression to predict
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    
    cols =['year']
    y = data['wqi']
    x=data[cols]
    reg=linear_model.LinearRegression()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
    reg.fit(x_train,y_train)
    
    a=reg.predict(x_test)
    print(a)
    
    print(y_test)
    
    from sklearn.metrics import mean_squared_error
    print('mse:%.2f'%mean_squared_error(y_test,a))
    
    dt = pd.DataFrame({'Actual': y_test, 'Predicted': a}) 
    
    
    print(x_test)
    
    x = (x - x.mean()) / x.std()
    x = np.c_[np.ones(x.shape[0]), x]
    print(x)
    
    alpha = 0.1 #Step size
    iterations = 3000 #No. of iterations
    m = y.size #No. of data points
    np.random.seed(4) #Setting the seed
    theta = np.random.rand(2) #Picking some random values to start with
    
    def gradient_descent(x, y, theta, iterations, alpha):
        past_costs = []
        past_thetas = [theta]
        for i in range(iterations):
            prediction = np.dot(x, theta)
            error = prediction - y
            cost = 1/(2*m) * np.dot(error.T, error)
            past_costs.append(cost)
            theta = theta - (alpha * (1/m) * np.dot(x.T, error))
            past_thetas.append(theta)
            
        return past_thetas, past_costs
    
    past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)
    theta = past_thetas[-1]
    
    #Print the results...
    print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))
    
    print(x_test)
    print(y_test)
    
    #prediction of january(2013-2015) across india
    import numpy as np
    newB=[74.76, 2.13]
    
    def rmse(y,y_pred):
        rmse= np.sqrt(sum(y-y_pred))
        return rmse
       
    
    y_pred=x.dot(newB)
    
    dt = pd.DataFrame({'Actual': y, 'Predicted': y_pred})  
    dt=pd.concat([data, dt], axis=1)
    print(dt)
    
    #testing the accuracy of the model
    
    from sklearn import metrics
    print(np.sqrt(metrics.mean_squared_error(y,y_pred)))
    
    #plotting the actual and predicted results
    x_axis=dt.year
    y_axis=dt.Actual
    y1_axis=dt.Predicted
    plt.scatter(x_axis,y_axis)
    #plt.plot(x_axis,y1_axis,color='r')
    plt.title("linear regression")
    
    #plt.show()
    #print(name)
    #name=list(name.split(" ")) 
    print(name)
    name=float(name)
    xx=[[name]]
    res=reg.predict(xx)
    res
    
    waterquality=""
    if(res<=100 and res>=90):
        waterquality=".....EXCELLENT, fit to drink"
    elif(res<=89 and res>=85):
        waterquality=".....GOOD, fit to drink"
    elif(res<=84 and res>=80):
        waterquality=".....ACCEPTABLE, drink at your own risk"
    elif(res<=79 and res>=60):
        waterquality=".....BAD, Not fit to drink"
    elif(res<=59 and res>=19):
        waterquality=".....POOR, Not fit to drink, chances of you dying is high"
    
    
    
    
    
    #res=res+"  "+waterquality;
    return '<h2 style="color:steelblue">The Predicted WQI for the input year is %s' % res + waterquality
























@app.route('/',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['year']
        return redirect(url_for('wqi',name = user))
    else:
        return render_template('proj_water_quality.html')

if __name__ == '__main__':
    app.run(debug = True)