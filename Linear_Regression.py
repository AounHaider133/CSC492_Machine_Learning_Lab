import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pandas.read_csv(url,names = names)

print(dataset.shape)
print(dataset.head())
print(dataset.describe())
print(dataset.groupby('class').size())

dataset.plot(kind='box',subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()

dataset.hist()
plt.show()
scatter_matrix(dataset)
plt.show()

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = validation_size, random_state = seed)
scoring = 'accuracy'

models = []
models.append(("LR",LogisticRegression()))
models.append(("LDA",LinearDiscriminantAnalysis()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("CART",DecisionTreeClassifier()))
models.append(("NB",GaussianNB()))
models.append(("SVM",SVC()))

#Evaluate each model in turn
results = []
names = []
seed = 6
for names,model in models:
    kfold = model_selection.KFold(n_splits = 10)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv = kfold, scoring = scoring)
    results.append(names)
    msg = "%s: %f (%f)" % (names, cv_results.mean(),cv_results.std())
    print(msg)