# Project 5 Implement a KNN learner
# Chintan sheth

#imports
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import operator
import math
class KNNLearner:
	def __init__(self,k,method):
                self.k=k
		self.method=method
	def addEvidence(self,Xtr,Ytr):
		self.Xtrain=Xtr
		self.Ytrain=Ytr
	def predict(self,Xtest):
		self.Xtest=Xtest
		Ypredict=()
        	Ypredict=zeros((len(self.Xtest)))
		for i in range(0,402):
			testtempdata= []
			for j in range(0,600):
				dist = ((self.Xtest[i][0]-self.Xtrain[j][0])**2 + (self.Xtest[i][1]-self.Xtrain[j][1])**2) ** 0.5
		#	print dist
		#	print Xtrain
				testtempdata.append(list(self.Xtrain[j]) + [dist])
		#print testtempdata
			testtempdata.sort(key=operator.itemgetter(3))
		#print testtempdata
			if (self.method=="mode"):
				freq=()
				freq=zeros((3))
		#	print freq
				for l in testtempdata[0:self.k]:
			#		print l[2]
        				if (l[2]==0):	
						freq[0] = freq[0] + 1
					elif (l[2]==1):
                                       		 freq[1] = freq[1] + 1
					else:
                                       		freq[2]=freq[2]+1
		        #print freq
				if(freq[0]>freq[1]):	
					 if(freq[0]>freq[2]):
						mode=0
					 else: 
						mode=-1
				else:
					if (freq[1]>freq[2]):
						mode=1
					else:
						mode=-1
				if(mode==0 or mode==-1):
					if (freq[0]==freq[2]):
						mode=0
				Ypredict[i]=mode
		
			mean=0
			if (self.method=="mean"):
				for l in testtempdata[0:self.k]:
					mean+=l[4]/4
				Ypredict[i]=mean
			median=0
			if (self.method=="median"):
				for l in testtempdata[0:self.k]:
					if (self.k%2==0):
						median=(l[self.k/2] + l[self.k/2 + 1])/2
					else:
						 median=l[self.k/2 + 1]
				Ypredict[i]=median			
		return Ypredict




data1 = np.loadtxt('data1.csv',delimiter=',',comments='#')
Xtrain1=data1[:600,:3]
Xtest1=data1[600:1002,:3]
Ytrain1=data1[:600,2:3]
Ytest1=data1[600:1002,2:3]

Xtrain=array(Xtrain1,dtype=float_)
Ytrain=array(Ytrain1,dtype=float_)
Xtest=array(Xtest1,dtype=float_)
Ytest=array(Ytest1,dtype=float_)

error=()
error=zeros((len(Ytest)))
errormean=0
k=600
Y=zeros((600))
minerr=2
optk=0
#plot k vs error
for a in range(1,k):
	print a
	learner=KNNLearner(a,"mode")
	learner.addEvidence(Xtrain,Ytrain)
	result=learner.predict(Xtest)
	for i in range (0,len(Ytest)):
		if (result[i]==Ytest[i]):
			error[i]=0
       		else:
			error[i]=1
	Y[a]=mean(error)
	if (Y[a]<minerr):
		minerr=Y[a]
		optk=a
print "Optimal values"
print minerr
print optk		
#plotting
plt.clf()
plt.plot(range(0,600),Y)
plt.xlabel('K')
plt.ylabel('Error')

savefig("report.pdf", format='pdf')






#for plot2ddata

learner=KNNLearner(3,"mode")
learner.addEvidence(Xtrain, Ytrain)
result=learner.predict(Xtest)


#plot 2ddata

data1 = np.loadtxt('data1.csv',delimiter=',',comments='#')
Xtest=data1[600:1002,:3]
Y=result
X1 = Xtest[:,0]
X2 = Xtest[:,1]


#
# Choose colors
#
miny = min(Y)
maxy = max(Y)
Y = (Y-miny)/(maxy-miny)
colors =[]
for i in Y:
        if (i>0.5):
                j = (i-.5)*2
                colors.append([j,(1-j),0])
        else:
                j = i*2
                colors.append([0,j,(1-j)])


# scatter plot X1 vs X2 and colors are Y
#
plt.clf()
plt.scatter(X1,X2,c=colors,edgecolors='none')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(-2,2)  # set x scale
plt.ylim(-2,2)  # set y scale
savefig("scatterdata.pdf", format='pdf')
