import pandas as pd

titanicData = pd.read_csv('titanic.csv', delimiter=';', decimal=',')

titanicFeaturesNames = ['Pclass', 'Sex', 'Age']
titanicDataFeatures = pd.DataFrame(titanicData, columns=titanicFeaturesNames)
titanicFeaturesList = titanicDataFeatures.values.tolist()

titanicDataResults = pd.DataFrame(titanicData, columns=['Survived'])
titanicResulstsList = titanicDataResults.values.tolist()

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=3)  # note: results looks wierd with more
clf = clf.fit(titanicFeaturesList, titanicResulstsList)
tree.export_graphviz(clf,
                     out_file="diagram.dot",
                     feature_names=titanicFeaturesNames,
                     class_names=['dead', 'alive'],
                     rounded=True,
                     rotate=True,
                     filled=True,
)

print("Decision tree saved to diagram.dot")
# note: -> webgraphviz.com <-
