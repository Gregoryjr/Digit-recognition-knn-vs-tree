import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def results(test, ans):
    ctf_score = 0
    knn_score = 0
    ctfwrong = []
    knnwrong = []
    for entry in range(len(test)):
        knn_score = knn_score + 1 if knn.predict([test[entry]]) == ans[entry] else knn_score
        ctf_score = ctf_score + 1 if clf.predict([test[entry]]) == ans[entry] else ctf_score

        if knn.predict([test[entry]]) != ans[entry]:
            knnwrong.append(entry)
        if clf.predict([test[entry]]) != ans[entry]:
            ctfwrong.append(entry)

    both_wrong = list(set(knnwrong).intersection(ctfwrong))

    print("knn score = ", knn_score / 1000 * 100, "%")
    print("ctf score = ", ctf_score / 1000 * 100, "%")

    print(" the following entry is where knn failed ", knnwrong)

    print(" the following entry is where the decison tree failed ", ctfwrong)
    print("both are wrong in these index: ", both_wrong)


def exam(num, test, act):
    d = test[num]
    d.shape = (28, 28)
    print("The KNeighborsClassifier predicted: ", knn.predict([test[num]]))
    print("The DecisionTreeClassifier predicted: ", clf.predict([test[num]]))
    print("the right answer ", ans[num])
    plt.imshow(255 - d, cmap='gray')
    plt.show()


# data comes from this link https://www.kaggle.com/competitions/digit-recognizer
# Switch to your own paths
training_path = "C:\\Users\\ezike\\OneDrive\\Documents\\train.csv"
testing_path = "C:\\Users\\ezike\\OneDrive\\Documents\\test.csv"

data = pd.read_csv(training_path).values
# organize data
# xtrain=data[0:41999, 1:]
xtrain = data[0:41000, 1:]
train_label = data[0:41000, 0]
# training Decision tree
clf = DecisionTreeClassifier()

clf.fit(xtrain, train_label)

# trining knn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain, train_label)

# testing data DT
tdata = pd.read_csv(testing_path).values  # 28000
xtest = data[41000:42000, 1:]
ans = data[41000:42000, 0]
# actual=tdata[0:30000]
results(xtest, ans)
# user testing machine
repeat = True
while (repeat == True):
    num = int(input("select a number: "))
    exam(num, xtest, 0)
    # print(ans[num])

    answer = input("do you want to try again [y/n]: ")
    if (answer == "n"):
        repeat = False
    else:
        repeat = True



