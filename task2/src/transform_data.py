import csv
import numpy as np

def map_(f, l):
    return list(map(f,l))


trainFile = open('X_train.csv', 'rU')
testFile = open('X_test.csv', 'rU')
labelsFile = open('y_train.csv', 'rU')
csvReader = csv.reader(trainFile)
csvReader2 = csv.reader(testFile)
csvReaderY = csv.reader(labelsFile)
X = list(csvReader)
y = list(csvReader)

y = np.array(map_(lambda l: map_(lambda x: int(x) if x != "" else float('NaN'), l), list(csvReaderY)[1:]))[:, 1:]
y = y.reshape((len(y),))
testFile.close()
trainFile.close()
labelsFile.close()

for i in range(1, len(X)):
    X[i][0] = y[i - 1]

out = open('out/X_transformed.csv', 'w+')

for x in X:
    for x_ in x:
        out.write("{0}, ".format(x_))
    out.write("\n")

out.close()