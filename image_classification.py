import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split

features=pandas.read_csv('textures_data.csv',sep=',')
data=numpy.array(features)
X=(data[:,:-1]).astype('float64')
Y=data[:,-1]

x_transform=PCA(n_components=3)
Xt=x_transform.fit_transform(X)

red=Y=='drewno'
blue=Y=='fotel'
cyan=Y=='plytki'

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(Xt[red,0],Xt[red,1],Xt[red,2],c='r')
ax.scatter(Xt[blue,0],Xt[blue,1],Xt[blue,2],c='b')
ax.scatter(Xt[cyan,0],Xt[cyan,1],Xt[cyan,2],c='c')

# classification module -----------------------------------------------------------------
classifier=svm.SVC(gamma='auto')
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33)

classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(acc)

cm=confusion_matrix(y_test,y_pred,normalize='true')
print(cm)

disp=plot_confusion_matrix(classifier,x_test,y_test,cmap=plt.cm.Blues)

plt.show()