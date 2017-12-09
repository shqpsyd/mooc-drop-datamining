from preprocessor import *
from classify import *
import csv
def XY():
    P = XY3predict(0,1,0,1)
    eid = readcsv("./rawdata/sample_submission.csv")

    with open("ADJ2.csv","wb") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['enrollment_id', 'dropout_prob'])
        count = 0
        for i in P:        
            csvwriter.writerow([eid[count][0],i])
            count+=1
def XY2():
    P = XY3predict(0,1,0,1)
    eid = readcsv("./rawdata/sample_submission.csv")

    with open("MLP3.csv","wb") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['enrollment_id', 'dropout_prob'])
        count = 0
        for i in P:        
            csvwriter.writerow([eid[count][0],i[1]])
            count+=1
XY2()




