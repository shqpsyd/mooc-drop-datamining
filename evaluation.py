from preprocessor import *
from classify import *
def eonline():
    Y = XY3predict(0,0.9,0.9,1)
    label = sepratecsv("./rawdata/train_label.csv",0.9,1)
    match = 0
    i = 0
    labeli = 0
    for predict in Y:
        
        if float(predict[1]) > 0.5:
            predict = 1
        else:
            predict = 0              
        if float(label[i][1])>0.5: 
            labeli = 1
        else:
            labeli = 0    
        if predict == labeli:
            match += 1        
        i += 1
    print match
    accuracy = float(match)/len(Y)
    print accuracy

def eoffline():
    Y = sepratecsv("ADJ2.csv",0,1)
    label = sepratecsv("./rawdata/train_label.csv",0,1)
    match = 0
    i = 0
    labeli = 0
    for predict in Y:
        if float(predict[1] > 0.5):
            predict = 1
        else:
            predict = 0   
        if float(label[i][1])>0.5: 
            labeli = 1
        else:
            labeli = 0    
        if predict == labeli:
            match += 1        
        i += 1
    print match
    accuracy = float(match)/len(Y)
    print accuracy  
eonline()  