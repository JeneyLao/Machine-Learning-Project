
"""
Created on Thu Mar 10 15:19:42 2016

@author: Jeney
"""
import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import mltools.dtree as dt
import mltools.linear

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Note: file is comma-delimited 
X1 = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',') 
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',') 
# also load features of the test data (to be predicted) 
Xe1= np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',') 

#Train_A, Train_B, Test_A, Test_B = ml.splitData(X1,Y,0.50)
Xtr,Xva,Ytr,Yva = ml.splitData(X1,Y,0.75)

featureXtr = []
featureXva = []
featureIndices = []
numberOfFeaturesAdded = []
count = 0
lowMSE = []
'''
def ForwardSelection(lowestMSE, featureXtr, featureXva, featureIndices):
    mseList = []    
    
    for i in range(X1.shape[1]):
        if (i not in featureIndices):
            if (len(featureXtr) == 0):
                lr = ml.linear.linearRegress(Xtr[:,i:i+1], Ytr)               
                #lr = dt.treeRegress(Xtr[:,i:i+1], Ytr, maxDepth=7)
                mseList.append(lr.mse(Xva[:,i:i+1], Yva))
            else:
                combine1 = np.column_stack((featureXtr, Xtr[:,i:i+1]))
                combine2 = np.column_stack((featureXva, Xva[:,i:i+1]))
                lr = ml.linear.linearRegress(combine1, Ytr)
                #lr = dt.treeRegress(combine1, Ytr, maxDepth=7)
                mseList.append(lr.mse(combine2, Yva))
        else:
            mseList.append(1.0)

    f = mseList.index(min(mseList))
    if (min(mseList) < lowestMSE):

        global numberOfFeaturesAdded
        global count
        global lowMSE
        numberOfFeaturesAdded.append(count)    
        count = count + 1
        lowMSE.append(min(mseList))   

        if (len(featureXtr) == 0):
            featureXtr = Xtr[:,f:f+1]
            featureXva = Xva[:,f:f+1]
        else:
            featureXtr = np.column_stack((featureXtr, Xtr[:,f:f+1]))
            featureXva = np.column_stack((featureXva, Xva[:,f:f+1]))
            
        featureIndices.append(f)
        ForwardSelection(min(mseList), featureXtr, featureXva, featureIndices)

ForwardSelection(1.0, featureXtr, featureXva, featureIndices)
plt.plot(numberOfFeaturesAdded, lowMSE, 'r')
plt.xlabel("Number of Features Added")
plt.ylabel("MSE")
plt.title("Using Forward Selection")
print(featureIndices)
'''
'''
lowMSE2=[]
iterations2=[]
count = 0
def foo2(lowestMSE, featureXtr, featureXva, featureIndices,count):
    mseList = []
    iterations2.append(count)
    count = count + 1
    for i in range(X1.shape[1]):
        if (i not in featureIndices):
            if (len(featureXtr) == 0):
                lr = ml.linear.linearRegress(Xtr[:,i:i+1], Ytr)               
                #lr = dt.treeRegress(Xtr[:,i:i+1], Ytr, maxDepth=7)
                mseList.append(lr.mse(Xva[:,i:i+1], Yva))
            else:
                combine1 = np.column_stack((featureXtr, Xtr[:,i:i+1]))
                combine2 = np.column_stack((featureXva, Xva[:,i:i+1]))
                lr = ml.linear.linearRegress(combine1, Ytr)
                #lr = dt.treeRegress(combine1, Ytr, maxDepth=7)
                mseList.append(lr.mse(combine2, Yva))
        else:
            mseList.append(1.0)
                
            
    #print(mseList)
    f = mseList.index(min(mseList))
    print("Lowest MSE: ", min(mseList))
    if (min(mseList) < lowestMSE):
        lowMSE2.append(min(mseList))
        if (len(featureXtr) == 0):
            featureXtr = Xtr[:,f:f+1]
            featureXva = Xva[:,f:f+1]
        else:
            featureXtr = np.column_stack((featureXtr, Xtr[:,f:f+1]))
            featureXva = np.column_stack((featureXva, Xva[:,f:f+1]))
        featureIndices.append(f)
        mse = min(mseList)
        foo2(min(mseList), featureXtr, featureXva, featureIndices, count)
'''
'''
foo2(mse, featureXtr, featureXva, featureIndices,count)
plt.plot(iterations2[0:-1], lowMSE2, 'g', label = "Tree Regression")
plt.plot(iterations[0:-1], lowMSE, 'r', label = "Linear Regression")
plt.xlabel("Number of Features Added")
plt.ylabel("MSE")
plt.title("Using Forward Selection")
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.0)

'''
'''
Xtr2 = np.zeros((Xtr.shape[0],2))
Xtr2[:,0] = Xtr[:,0]
Xtr2[:,1] = Xtr[:,0]**2

Xva2 = np.zeros((Xva.shape[0],2))
Xva2[:,0] = Xva[:,0]
Xva2[:,1] = Xva[:,0]**2

def foo2(startingFeatures, valFeatures, lowestMSE, featureXtr, featureXva, featureIndices):
    mseList = []
    for i in range(Xtr2.shape[1] + len(featureIndices)):
        if (i not in featureIndices):
            if (len(featureXtr) == 0):
                lr = ml.linear.linearRegress(Xtr[:,i:i+1], Ytr)               
                mseList.append(lr.mse(Xva[:,i:i+1], Yva))
            else:
                combine1 = np.column_stack((featureXtr, Xtr[:,i:i+1]))
                combine2 = np.column_stack((featureXva, Xva[:,i:i+1]))
                lr = ml.linear.linearRegress(combine1, Ytr)
                mseList.append(lr.mse(combine2, Yva))
        else:
            mseList.append(1.0)
                
            
    #print(mseList)
    f = mseList.index(min(mseList))
    print("Lowest MSE: ", min(mseList))
    if (min(mseList) < lowestMSE):
        if (len(featureXtr) == 0):
            featureXtr = Xtr[:,f:f+1]
            featureXva = Xva[:,f:f+1]
        else:
            featureXtr = np.column_stack((featureXtr, Xtr[:,f:f+1]))
            featureXva = np.column_stack((featureXva, Xva[:,f:f+1]))
        featureIndices.append(f)
        mse = min(mseList)
        foo(featureXtr, featureXva, min(mseList), featureXtr, featureXva, featureIndices)


#foo2(Xtr2[:,0:1], Xva2[:,0:1], 1.0, featureXtr, featureXva, featureIndices)
'''

''' Variables / Lists '''
featureXtr = []
featureXva = []
featureIndices = []
numberOfFeaturesAdded = []
count = 0
lowMSE = []

def BackwardElimination(lowestMSE, featureXtr, featureXva, featureIndices):
    mseList = []
    for i in range(X1.shape[1] - len(featureIndices)):
        if (len(featureXtr) == 0):
            RFT = np.delete(Xtr,i,1)
            RFV = np.delete(Xva,i,1)
            lr = ml.linear.linearRegress(RFT, Ytr)               
            mseList.append(lr.mse(RFV, Yva))
        else:
            RFT = np.delete(featureXtr,i,1)
            RFV = np.delete(featureXva,i,1)
            lr = ml.linear.linearRegress(RFT, Ytr)
            mseList.append(lr.mse(RFV, Yva))

    f = mseList.index(min(mseList))
    if (min(mseList) < lowestMSE):
        ''' Used to Plot Graphs '''
        global numberOfFeaturesAdded
        global count
        global lowMSE
        numberOfFeaturesAdded.append(count)    
        count = count + 1
        lowMSE.append(min(mseList))   
        ''' The above code is not needed for this function to work '''
        featureXtr = np.delete(Xtr,f,1)
        featureXva = np.delete(Xva,f,1)
        featureIndices.append(f)
        BackwardElimination(min(mseList), featureXtr, featureXva, featureIndices)


BackwardElimination(1.0, featureXtr, featureXva, featureIndices)
plt.plot(numberOfFeaturesAdded, lowMSE, 'r')
plt.xlabel("Number of Features Removed")
plt.ylabel("MSE")
plt.title("Using Backward Elimination with Linear Regression")



'''
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X11 = sel.fit_transform(X1)
Xtr,Xva,Ytr,Yva = ml.splitData(X11,Y,0.75)
lr = ml.linear.linearRegress(Xtr, Ytr)
print(lr.mse(Xva, Yva))
'''

'''
lasso = Lasso().fit(X1, Y)
model = SelectFromModel(lasso, prefit = True)
X_new = model.transform(X1)
Xtr,Xva,Ytr,Yva = ml.splitData(X_new,Y,0.75)
lr = ml.linear.linearRegress(Xtr, Ytr)
print(lr.mse(Xva, Yva))
'''
'''
MSEtr = []
MSEva = []
index = []
for i in range (92):
    index.append(i)
    X11 = SelectKBest(f_regression, k=i).fit_transform(X1, Y)
    Xtr,Xva,Ytr,Yva = ml.splitData(X11,Y,0.75)
    lr = ml.linear.linearRegress(Xtr, Ytr)
    #lr = dt.treeRegress(Xtr,Ytr,maxDepth=20,minParent=9)
    MSEva.append(lr.mse(Xva, Yva))
    MSEtr.append(lr.mse(Xtr, Ytr))
    print("Number of Features Not Removed: ", i, ", MSE is: ", lr.mse(Xva,Yva))

plt.plot(index, MSEtr, 'r', label ="Training Data")
plt.plot(index, MSEva, 'g', label ="Validation Data")
plt.xlabel("K")
plt.ylabel("MSE")
plt.title("Removes all but the K Highest Score Features")
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.0)
'''
'''
X11 = SelectKBest(f_regression, 50).fit_transform(X1, Y)
lr = ml.linear.linearRegress(X1,Y)
ys = lr.predict(Xe1)
'''
'''
def foo(startingFeatures, valFeatures, lowestMSE, featureXtr, featureXva, featureIndices):
    mseList = []
    for i in range(X11.shape[1]):
        if (i not in featureIndices):
            if (len(featureXtr) == 0):
                lr = ml.linear.linearRegress(Xtr[:,i:i+1], Ytr)               
                #lr = dt.treeRegress(Xtr[:,i:i+1], Ytr, maxDepth=7)
                mseList.append(lr.mse(Xva[:,i:i+1], Yva))
            else:
                combine1 = np.column_stack((featureXtr, Xtr[:,i:i+1]))
                combine2 = np.column_stack((featureXva, Xva[:,i:i+1]))
                lr = ml.linear.linearRegress(combine1, Ytr)
                #lr = dt.treeRegress(combine1, Ytr, maxDepth=7)
                mseList.append(lr.mse(combine2, Yva))
        else:
            mseList.append(1.0)
                
            
    #print(mseList)
    f = mseList.index(min(mseList))
    print("Lowest MSE: ", min(mseList))
    if (min(mseList) < lowestMSE):
        if (len(featureXtr) == 0):
            featureXtr = Xtr[:,f:f+1]
            featureXva = Xva[:,f:f+1]
        else:
            featureXtr = np.column_stack((featureXtr, Xtr[:,f:f+1]))
            featureXva = np.column_stack((featureXva, Xva[:,f:f+1]))
        featureIndices.append(f)
        mse = min(mseList)
        foo(featureXtr, featureXva, min(mseList), featureXtr, featureXva, featureIndices)

#foo(Xtr[:,0:1], Xva[:,0:1], 1.0, featureXtr, featureXva, featureIndices)

print(featureIndices)

featurestest = []
featurex1 = []
for i in featureIndices:
    if (len(featurestest) == 0):
        featurex1 = X1[:,i:i+1]
        featurestest = Xe1[:,i:i+1]
    else:
        featurex1 = np.column_stack((featurex1, X1[:,i:i+1]))
        featurestest = np.column_stack((featurestest, Xe1[:,i:i+1]))
        
#lineartest = ml.linear.linearRegress(featurex1, Y)
        
lineartest = dt.treeRegress(featurex1, Y, maxDepth=7)
ys = lineartest.predict(featurestest)
'''
'''
fh = open('predictions.csv', 'w')
fh.write('ID,Prediction\n')
for i,yi in enumerate(ys):
    fh.write('{},{}\n'.format(i+1,yi))
fh.close()
print("done")
'''
        
'''
f = mseList.index(min(mseList))
firstArray = Xtr[:,f:f+1]
firstArray2 = Xva[:,f:f+1]

mseList2 = []
for i in range(X1.shape[1] - 1):
    if (i != f):
        feature.append(i)
        combinedF = np.column_stack((firstArray, Xtr[:,i:i+1]))
        lr = ml.linear.linearRegress(combinedF, Ytr)
        combinedF2 = np.column_stack((firstArray2, Xva[:,i:i+1]))
        mseList2.append(lr.mse(combinedF2, Yva))
        


for i in feature:
    print("Using Feature ", i, " - MSE is: ", mseList2[i])


for i in range(X1.shape[1]):
    
'''




'''
Train_AA, Train_AB, Test_AA, Test_AB = ml.splitData(Train_A, Test_A, 0.50)
Train_BA, Train_BB, Test_BA, Test_BB = ml.splitData(Train_B, Test_B, 0.50)

ensemble = [0]

index = [1,5,10,25]
TrainingErrorList = []
ValidationErrorList = []
'''
'''
def boost(num, Xtr, Xva, Ytr, Yva):
    nBoost = num
    learner = ensemble * nBoost
    alpha = [1.0] * nBoost
    mu = Ytr.mean()
    dY = Ytr - mu
    for k in range(nBoost):
        print(k)
        learner[k] = (dt.treeRegress(Xtr,dY, maxDepth=2)) 
        ys = learner[k].predict(Xtr);
        learner[k] = (dt.treeRegress(Xtr,ys, maxDepth=3)) 
        ys = learner[k].predict(Xtr);
        learner[k] = (dt.treeRegress(Xtr,ys, maxDepth=4)) 
        ys = learner[k].predict(Xtr);        
        learner[k] = ml.linear.linearRegress(Xtr,ys)
        alpha[k] = 1.0
        
        dY = dY - alpha[k] * learner[k].predict(Xtr)[:,0]
    mTest = Xva.shape[0]
    predict1 = np.zeros((mTest,)) + mu
    for k in range(nBoost):
        predict1 += alpha[k] * learner[k].predict(Xva)[:,0]
    mTrain = Xtr.shape[0]
    predict2 = np.zeros((mTrain,)) + mu
    for k in range(nBoost):
        predict2 += alpha[k] * learner[k].predict(Xtr)[:,0]
    mseTrain = np.mean( (Ytr - predict2)**2 , axis=0)
    mseTest = np.mean( (Yva - predict1)**2 , axis=0)
    TrainingErrorList.append(mseTrain)
    ValidationErrorList.append(mseTest)

boost(25, Train_A, Train_B, Test_A, Test_B)
print(TrainingErrorList)
print(ValidationErrorList)
'''
'''
learner11 = dt.treeRegress(Train_AA, Test_AA, maxDepth=7)
learner12 = dt.treeRegress(Train_AB, Test_AB, maxDepth=7)
learner21 = dt.treeRegress(Train_BA, Test_BA, maxDepth=7)
learner22 = dt.treeRegress(Train_BB, Test_BB, maxDepth=7)

lr = ml.linear.linearRegress(Train_A, Test_A)
print(lr.mse(Train_A, Test_A))
print(lr.mse(Train_B, Test_B))
'''

'''
learner11 = ml.linear.linearRegress(Train_AA, Test_AA)
learner12 = ml.linear.linearRegress(Train_AB, Test_AB)
learner21 = ml.linear.linearRegress(Train_BA, Test_BA)
learner22 = ml.linear.linearRegress(Train_BB, Test_BB)
'''
'''
ys11 = learner11.predict(Train_AA)
ys12 = learner12.predict(Train_AB)
ys21 = learner21.predict(Train_BA)
ys22 = learner22.predict(Train_BB)

ys1122 = []
for i in ys11:
    ys1122.append(i)
for i in ys12:
    ys1122.append(i)
for i in ys21:
    ys1122.append(i)
for i in ys22:
    ys1122.append(i)
#learner12 = ml.linear.linearRegress(X1, ys1122)
print(len(ys1122))
print(X1.shape)
learner12 = dt.treeRegress(X1, ys1122, maxDepth = 7)
print(learner12.mse(X1,Y))
print(learner12.mse(Train_A, Test_A))
print(learner12.mse(Train_B, Test_B))
'''
'''
xs = np.linspace(0,10,200)[:,np.newaxis]


ys = learner12.predict(Xe1)

fh = open('predictions.csv', 'w')
fh.write('ID,Prediction\n')
for i,yi in enumerate(ys):
    fh.write('{},{}\n'.format(i+1,yi))
fh.close()
print("done")
'''