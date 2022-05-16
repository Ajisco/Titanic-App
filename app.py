import pandas as pd
import numpy as np
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from flask import Flask, request, render_template

app= Flask(__name__)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data['Age'] = train_data['Age'].replace(np.NaN, train_data['Age'].mean())
train_data['Embarked'] = train_data['Embarked'].replace(np.NaN, statistics.mode(train_data['Embarked'] ) )                                            
train_data=pd.get_dummies(train_data,columns=['Embarked', "Pclass", "Sex", "SibSp", "Parch"],drop_first=True)
train_data =train_data.drop('Name', axis = 1).drop('Ticket', axis = 1).drop('Cabin', axis = 1)
x_train = train_data.drop('Survived', axis = 1) 
y_train = train_data['Survived']
test_data['Age'] = test_data['Age'].replace(np.NaN, test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].replace(np.NaN, test_data['Fare'].mean())
test_data['Embarked'] = test_data['Embarked'].replace(np.NaN, statistics.mode(test_data['Embarked'] ))
test_data=pd.get_dummies(test_data,columns=['Embarked', "Pclass", "Sex", "SibSp", "Parch"],drop_first=True)
test_data =test_data.drop('Name', axis = 1).drop('Ticket', axis = 1).drop('Cabin', axis = 1)
x_test = test_data
x_train['Parch_3'] = x_train['Parch_4'].apply(lambda x: 0)
x_train['Parch_9'] = x_train['Parch_4'].apply(lambda x: 0)

rf = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def index():
    pid= request.form['id']
    age= request.form['age']
    fare= request.form['fare']
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r  = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    (a.append(1) if request.form['sex'] == 1 else a.append(0)) #sex male
    (b.append(1) if request.form['port'] == 1 else b.append(0)) #Embark Q
    (c.append(1) if request.form['port'] == 2 else c.append(0)) #Embark S
    (d.append(1) if request.form['pclass'] == 2 else d.append(0)) #Pclass 2
    (e.append(1) if request.form['pclass'] == 3 else e.append(0)) #Pclass 3
    (f.append(1) if request.form['sib'] == 1 else f.append(0)) #sib 1
    (g.append(1) if request.form['sib'] == 2 else g.append(0)) #sib 2
    (h.append(1) if request.form['sib'] == 3 else h.append(0)) #sib 3
    (i.append(1) if request.form['sib'] == 4 else i.append(0)) #sib 4
    (j.append(1) if request.form['sib'] == 5 else j.append(0)) #sib 5
    (k.append(1) if request.form['sib'] == 8 else k.append(0)) #sib 8
    (l.append(1) if request.form['parch'] == 1 else l.append(0)) #parch 1
    (m.append(1) if request.form['parch'] == 2 else m.append(0)) #parch 2
    (n.append(1) if request.form['parch'] == 3 else n.append(0)) #parch 3
    (o.append(1) if request.form['parch'] == 4 else o.append(0)) #parch 4
    (p.append(1) if request.form['parch'] == 5 else p.append(0)) #parch 5
    (q.append(1) if request.form['parch'] == 6 else q.append(0)) #parch 6
    (r.append(1) if request.form['parch'] == 9 else r.append(0)) #parch 9
    sex,embq,embs,pclass2,pclass3,sib1,sib2,sib3,sib4,sib5,sib8,parch1,parch2,parch3,parch4,parch5,parch6,parch9 = a[0],b[0],c[0],d[0],e[0],f[0],g[0],h[0],i[0],j[0],k[0],l[0],m[0],n[0],o[0],p[0],q[0],r[0]
    arr = np.array([[pid,age,fare,embq,embs,pclass2,pclass3,sex,sib1,sib2,sib3,sib4,sib5,sib8,parch1,parch2,parch3,parch4,parch5,parch6,parch9]])
    pred= rf_random.predict(arr)
    return render_template('after.html', data=pred)
        

if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)
    
    
     




    