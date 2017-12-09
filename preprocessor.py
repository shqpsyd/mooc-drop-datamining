import csv
from sets import Set
from sklearn.preprocessing import StandardScaler
from datetime import datetime

events = {'problem':0,'video':1,'access':2,'wiki':3,'discussion':4,'navigate':5,'page_close':6}
'''
readcsv
input:.csv file
output: list of every row
'''
def readcsv(filename):
    rowcontainer = []
    with open (filename,'rb') as csvfile:
        rowreader = csv.reader(csvfile)
        for row in rowreader:
            rowcontainer.append(row)
    
    return rowcontainer[1:]
'''
seprateFile
input: .csv file, the percentage of where to start, the percentage of where to end
output: list of part of csv file  
'''
def sepratecsv(filename,start,end):
    originData = readcsv(filename)
    return originData[int(start*len(originData)):int(end*len(originData))]

'''
preXY1
input: activity_log.csv
output: {'enrollment_id':['prolemCount','vedioCount'....]}
e.g. {'1':[2,3,...],'2':[3,4...].....}
'''
def preXY1(filename):
    rowcontainer = readcsv(filename)
    actDict = {}
    durationDict = {}
    #Max = [0,0,0,0,0,0,0]
    for row in rowcontainer:
        if row[0] not in actDict:
            actDict[row[0]] = [0,0,0,0,0,0,0,0]        
        actDict[row[0]][events[row[2]]] += 1  
        actDict[row[0]][7] += 1
    '''
    for a in actDict: 
        for i in range(7):
            if actDict[a][i] > Max[i]:
                Max[i] = actDict[a][i]
    for a in actDict:
        for i in range(7):            
            actDict[a][i]/=float(Max[i]) 
    '''  

    '''
    make use of duration time 
    '''
    minT = datetime.strptime('2018-05-31T12:43:20','%Y-%m-%dT%H:%M:%S')
    for row in rowcontainer:
        if row[0] not in durationDict:
            start = datetime.strptime('2018-05-31T12:43:20','%Y-%m-%dT%H:%M:%S')
            end = datetime.strptime('1995-05-31T12:43:20','%Y-%m-%dT%H:%M:%S')
            durationDict[row[0]] = [start,end]
        time = datetime.strptime(row[1],'%Y-%m-%dT%H:%M:%S')
        if time < minT:
            minT = time    
        if time < durationDict[row[0]][0]:
            durationDict[row[0]][0] = time
        if time > durationDict[row[0]][1]:
            durationDict[row[0]][1] = time
    
    for row in durationDict:        
        start = (durationDict[row][0]-minT).total_seconds()        
        end = (durationDict[row][1]-minT).total_seconds()        
        duration = (end - start) 
             
        if duration == 0:
            actDict[row] += [duration, start, end, 0]
        else:
            actDict[row] += [duration, start, end, actDict[row][7]/duration]
        '''
        actDict[row] += [duration, start, end]
        '''
    '''
    make use of partitioin time
    
    partitioinDict = {}
    for row in rowcontainer:
        if row[0] not in partitioinDict:
            partitioinDict[row[0]] = [0,0,0]
        time = datetime.strptime(row[1],'%Y-%m-%dT%H:%M:%S')
        if ((time-minT).total_seconds() - actDict[row[0]][9]) <= actDict[row[0]][8]/3:
            partitioinDict[row[0]][0]+=1
        elif ((time-minT).total_seconds() - actDict[row[0]][9]) > actDict[row[0]][8]/3 and ((time-minT).total_seconds() - actDict[row[0]][9]) <= actDict[row[0]][8]*2/3:
            partitioinDict[row[0]][1]+=1
        else:
            partitioinDict[row[0]][2]+=1
    for row in partitioinDict:
        actDict[row] += partitioinDict[row]
    '''    

    return actDict
'''        
XY1
input: the percentage of where to start(of train data), the percentage of where to end(of traindata), activity_log.csv,train_label.csv
return:X = ['prolemCount','vedioCount'....], Y = ['train_label']
e.g. [2,4,....],[0]
'''
def getXY1(start,end,activityFile,trainFile):
    X = []
    Y = []
    actDict = preXY1(activityFile)
    trainDict = sepratecsv(trainFile,start,end)
    for tid in trainDict:
        eid = tid[0]
        label = tid[1]
        if eid not in actDict:
            continue
        X.append(actDict[eid])
        Y.append(label)
    return X,Y  

'''
enrollment_list.csv
add how many people have add before drop course by course 
'''
def preXY2(filename):  
    course = Set([])
    enrollDict = {}
    data = readcsv(filename)
    courseNum = {}  
    lessonNum = {}  
    for row in data: 
        if row[2] not in courseNum:
            courseNum[row[2]] = 0
        courseNum[row[2]] += 1
    
    #courseMax = max(courseNum.values()) 
    
    for row in data:
        if row[1] not in lessonNum:
            lessonNum[row[1]] = 0
        lessonNum[row[1]] += 1
    #lessonMax = max(lessonNum.values())
    
    for row in data:        
            #enrollDict[row[0]] = [float(courseNum[row[2]])/courseMax, float(lessonNum[row[1]])/lessonMax]   
            enrollDict[row[0]] = [courseNum[row[2]],lessonNum[row[1]]]
    return enrollDict

def getXY2(start,end,activityFile,enrollFile,trainFile):
    newDict = preXY1(activityFile)
    enrDict = preXY2(enrollFile)
    for i in newDict:
        newDict[i] += enrDict[i]
    X = []
    Y = []
    trainDict = sepratecsv(trainFile,start,end)
    for tid in trainDict:
        eid = tid[0]
        label = tid[1]
        if eid not in newDict:
            continue
        X.append(newDict[eid])
        Y.append(label)
        scaler = StandardScaler()        
    return scaler.fit_transform(X),Y 

def getXY(start,end,trainFile):
    SVM = sepratecsv("SVM.csv",start,end)
    MLP = sepratecsv("MLP.csv",start,end)
    RF = sepratecsv("RF.csv",start,end)
    BAG = sepratecsv("BAG.csv",start,end)
    ADA = sepratecsv("ADA.csv",start,end)   
    X = [] 
    Y = []
    trainDict = sepratecsv(trainFile,start,end)
    for tid in trainDict:        
        label = tid[1]       
        Y.append(label)  
    for i in range (len(SVM)):
        X.append([SVM[i][1],MLP[i][1],RF[i][1],BAG[i][1],ADA[i][1]])
    return X,Y

        

        
   
        






print preXY1("./rawdata/activity_log.csv")
#print getXY2(0,0.1,"./rawdata/activity_log.csv","./rawdata/enrollment_list.csv","./rawdata/train_label.csv")[0]





        
    
        
    





    
    
