'''
XY1predict:make use of ['prolemCount','vedioCount'....], Y = ['train_label'] 
'''
from sklearn import svm
from preprocessor import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import numpy as np
'''
def XY1predict(Tstart,Tend,Pstart,Pend):
    
    enrollFile = "./rawdata/enrollment_list.csv"
    activityFile = "./rawdata/activity_log.csv"
    trainFile = "./rawdata/train_label.csv"
    sampleFile = "./rawdata/sample_submission.csv"
    X,Y = getXY1(Tstart,Tend,activityFile,trainFile)
    clf = svm.SVC()
    clf.fit(X,Y)
    print "finish train"   
    X = getXY1(Pstart,Pend,activityFile,sampleFile)[0]    
    Y = clf.predict(X)    
    print "finish predict"
    return Y
'''
def XY2predict(Tstart,Tend,Pstart,Pend):
    sampleFile = "./rawdata/sample_submission.csv"
    enrollFile = "./rawdata/enrollment_list.csv"
    activityFile = "./rawdata/activity_log.csv"
    trainFile = "./rawdata/train_label.csv"
    X,Y = getXY2(Tstart,Tend,activityFile,enrollFile,trainFile)
    clf = svm.SVC(probability = True)
    clf.fit(X,Y)
    print "finish train"   
    X = getXY2(Pstart,Pend,activityFile,enrollFile,sampleFile)[0]    
    #Y = clf.predict(X)
    P = clf.predict_proba(X)
    print "svm finish predict"
    return P

def XY3predict(Tstart,Tend,Pstart,Pend):
    sampleFile = "./rawdata/sample_submission.csv"
    enrollFile = "./rawdata/enrollment_list.csv"
    activityFile = "./rawdata/activity_log.csv"
    trainFile = "./rawdata/train_label.csv"
    X,Y = getXY2(Tstart,Tend,activityFile,enrollFile,trainFile)
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(1000,), random_state=1, activation = 'tanh')
    clf.fit(X,Y)
    print "finish train"   
    X = getXY2(Pstart,Pend,activityFile,enrollFile,sampleFile)[0]    
    #Y = clf.predict(X)
    P = clf.predict_proba(X)
    print "mlp finish predict"
    return P

def XY4predict(Tstart,Tend,Pstart,Pend):
    sampleFile = "./rawdata/sample_submission.csv"
    enrollFile = "./rawdata/enrollment_list.csv"
    activityFile = "./rawdata/activity_log.csv"
    trainFile = "./rawdata/train_label.csv"
    X,Y = getXY2(Tstart,Tend,activityFile,enrollFile,trainFile)
    clf = RandomForestClassifier(n_jobs = -1,random_state=0,n_estimators = 1000)
    clf.fit(X,Y)
    print "finish train"   
    X = getXY2(Pstart,Pend,activityFile,enrollFile,sampleFile)[0]    
    #Y = clf.predict(X)
    P = clf.predict_proba(X)
    print "rf finish predict"
    return P
'''
not works well
'''
def XY5predict(Tstart,Tend,Pstart,Pend):
    sampleFile = "./rawdata/sample_submission.csv"
    enrollFile = "./rawdata/enrollment_list.csv"
    activityFile = "./rawdata/activity_log.csv"
    trainFile = "./rawdata/train_label.csv"
    X,Y = getXY2(Tstart,Tend,activityFile,enrollFile,trainFile)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=60),algorithm="SAMME.R", n_estimators=500)
    clf.fit(X,Y)
    print "finish train"   
    X = getXY2(Pstart,Pend,activityFile,enrollFile,sampleFile)[0]    
    #Y = clf.predict(X)
    P = clf.predict_proba(X)
    print "ada finish predict"
    return P

def XY6predict(Tstart,Tend,Pstart,Pend):
    sampleFile = "./rawdata/sample_submission.csv"
    enrollFile = "./rawdata/enrollment_list.csv"
    activityFile = "./rawdata/activity_log.csv"
    trainFile = "./rawdata/train_label.csv"
    X,Y = getXY2(Tstart,Tend,activityFile,enrollFile,trainFile)
    clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5,n_estimators = 15)
    clf.fit(X,Y)
    print "finish train"   
    X = getXY2(Pstart,Pend,activityFile,enrollFile,sampleFile)[0]    
    #Y = clf.predict(X)
    P = clf.predict_proba(X)
    print "bag finish predict"
    return P

def XYpredict(Tstart,Tend,Pstart,Pend):
    trainFile = "./rawdata/train_label.csv"  
    X,Y = getXY(Tstart,Tend,trainFile)
    label = []
    P = []
    for x in X:
        p0 = 0
        p1 = 0
        c0 = 0
        c1 = 0 
        p = 0       
        x = np.array(x).reshape(-1,1)        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
        
        for i in range(5):            
	        if kmeans.labels_[i] == 0:
		        p0 += float(x[i][0])
		        c0 += 1
	        else:
		        p1 += float(x[i][0])
		        c1 += 1            
        if c1>c0:
	        p = p1/float(c1)
        else:
	        p = p0/float(c0)
        P.append(p)
        if p >=0.5:
            label.append(1)
        else:
            label.append(0)       
    return P
#print XYpredict(0,1,0,1)

def ADJpredict(Tstart,Tend,Pstart,Pend):
    trainFile = "./rawdata/train_label.csv"  
    X,Y = getXY(Tstart,Tend,trainFile)
    label = []
    P = []
    for x in X:
        p0 = 0
        p1 = 0
        c0 = 0
        c1 = 0 
        p = 0       
        x = np.array(x).reshape(-1,1)        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
        
        for i in range(5):            
	        if kmeans.labels_[i] == 0:
		        p0 += float(x[i][0])
		        c0 += 1
	        else:
		        p1 += float(x[i][0])
		        c1 += 1            
        if c1>3 and kmeans.labels_[1] == 0:            
	        p = p1/float(c1)
        elif c0>3 and kmeans.labels_[1] == 1:
	        p = p0/float(c0)
        else:
            p = float(x[1][0])
        P.append(p)
        if p >=0.5:
            label.append(1)
        else:
            label.append(0)       
    return P


    

    
    
    





