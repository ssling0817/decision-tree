## you cannot use sklearn

import glob
import numpy as np
import csv
from Function import *
import math


def read_csv(filename):
    t = []
    csv_file = open(filename,'r')
    i =0
    for row in csv.reader(csv_file):
        # if (i > 2):
        t.append(row)
        # i+=1
    # print (t)
    t = np.array(t,dtype = np.float32)
    return t

def accF (x,y):
    acc = 0
    #print('predict',x)
    for i in range(len(x)):
        if x[i] == y[i]:
            acc +=1
    print('acc',acc)
    acc = float(acc) / len(x)
    return acc 

def feature_ex(x):
    t = []
    x = np.array(x)
    indexAx = 0
    indexAy = 1
    indexAz = 2
    indexGx = 3
    indexGy = 4
    indexGz = 5
    indexTotalAcc = 6
    indexTotalGyro = 7
    indexRoll = 8
    indexPitch = 9
    
    totalAcc = getTotalAxes(x[indexAx],x[indexAy],x[indexAz])
    totalGyro = getTotalAxes(x[indexGx],x[indexGy],x[indexGz])
    roll = getRoll(x[indexAx],x[indexAz])
    pitch = getPitch(x[indexAy],x[indexAz])

    processedKoalaData = np.ones((10,40))

    for i in range(6):
        processedKoalaData[i] = copy.deepcopy(x[i])
    processedKoalaData[6] = copy.deepcopy(totalAcc)
    processedKoalaData[7] = copy.deepcopy(totalGyro)
    processedKoalaData[8] = copy.deepcopy(roll)
    processedKoalaData[9] = copy.deepcopy(pitch)



    mean = getMean2D(processedKoalaData)
    t.append(mean)


    return t
#00
def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # label:the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val >= self.value
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        '''
        impurity -= prob_of_lbl**2
        '''
        impurity -= prob_of_lbl*math.log(prob_of_lbl,2)
    return impurity

def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
def find_best_split(rows):
    best_gain = 0  
    best_question = None 
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  

    for col in range(n_features): 
        values = set([row[col] for row in rows]) 
        for val in values: 
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
                
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)
        
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)

    
    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)
def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")
    
def classify(row, node):
    
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)




##################################
## load data
##################################

sensor = []
label = []
feature = []
f = glob.glob(r'40_data/down'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([0])

f = glob.glob(r'40_data/up'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([0])

f = glob.glob(r'40_data/left'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([1])
f = glob.glob(r'40_data/right'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([1])

f = glob.glob(r'40_data/CW'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([2])

f = glob.glob(r'40_data/CCW'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([3])



f = glob.glob(r'40_data/VLR'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([4])

f = glob.glob(r'40_data/VRL'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([5])


f = glob.glob(r'40_data/non'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([6])







sensor = np.array(sensor)
print ('sensor shape is :',sensor.shape)

feature_name = [ "x["+str(i)+"]" for i in range((sensor.shape[1]*sensor.shape[2]))]
sensor = np.reshape(sensor,(sensor.shape[0],sensor.shape[1]*sensor.shape[2]))
label  = np.array(label)
print ('sensor shape after is :',sensor.shape)
print ('label shape is :',label.shape)
#print ('label is :',label)

## 00
label=label.reshape(label.shape[0],1)
train_data = np.append(sensor,label,axis=1)
np.random.shuffle(train_data)
#feature choosing
fea=[ 0 , 1,  2,  3,  4,  7,  9, 10]
data_n=train_data.shape[0]
t=data_n//10 *8
v=data_n-t
train_X = train_data[:t]
train_X=train_X.take(fea,axis = 1)
train_y=label
val_X =train_data[t:,0:10]
val_X=val_X.take(fea[:-1],axis = 1)
val_y = train_data[t:,-1]



#train_X, val_X, train_y, val_y =?

#print ('train_y is :',train_y)
val_y.astype(int)
#print ('val_y is :',val_y)

#filter outliers
tolerance=6
import statistics
print("before filter",train_X.shape)
for i in range(train_X.shape[1]):    
    train_X = train_X[np.abs((train_X[:,i]-np.mean(train_X[:,i]))<=statistics.stdev(tolerance*train_X[:,i]))]
print("after filter",train_X.shape)  


####################################
# # build decision tree
my_tree = build_tree(train_X)
val_y_predicted=[]
error=0
for row in val_X:
    dic=classify(row, my_tree)
    for label in dic.keys():
       # print(int(label))
        val_y_predicted.append(int(label))

####################################


# # predict training data or testing data
#val_y_predicted = 
#train_X_predicted = 


# # # label is 
# # print(val_y)

acc = accF(val_y_predicted,val_y)
print ('val_y acc is :',acc)
#acc = accF(train_X_predicted,train_y)
#print ('train_y acc is :',acc)



### export your testing prediction to .csv
### upload to Kaggle and get your score 
f = glob.glob(r'testdata'+'/*.csv')
test_data=[None]*71
for i in range(len(f)):
    test = read_csv(f[i])
    if (len(test[0])) == 40:
        test = feature_ex(test)
        test_data[int(f[i][9:11])-1]=test
test_data = np.array(test_data)
test_data = np.reshape(test_data,(test_data.shape[0],test_data.shape[1]*test_data.shape[2]))
#print('test_data',test_data.shape)
test_data=test_data.take(fea[:-1],axis = 1)
#my_tree = build_tree(train_X)

with open('submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id','Category'])
    c=1
    for row in test_data:
        dic=classify(row, my_tree)
        for label in dic.keys():
            #print(int(label)) 
            writer.writerow([str(c).zfill(2)+".csv", int(label)])
        c+=1
#check tree depth
max_d=1;
def print_tree(node,depth, spacing=""):
    global max_d
    if isinstance(node, Leaf):
        if(depth>max_d):
            max_d=depth
        return 

    print_tree(node.true_branch,depth+1, spacing + "  ")
    print_tree(node.false_branch,depth+1, spacing + "  ")
print_tree(my_tree,1)
print(max_d)