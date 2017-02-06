#----------------------------------Decision tree using scikit learn---------------------------------------
import pandas as pd
target = pd.read_csv("dt-data.txt",
names=['Size', 'Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy'], skipinitialspace=True,
skiprows=[0],index_col=False)
target['Size'] = target['Size'].str.replace('\d+:', '')
target['Enjoy'] = target['Enjoy'].str.replace(';', '')
from sklearn import preprocessing
label_processor = preprocessing.LabelEncoder()
target.VIP = label_processor.fit_transform(target.VIP)
target.Enjoy = label_processor.fit_transform(target.Enjoy)
target.Size = label_processor.fit_transform(target.Size)
target.Occupied = label_processor.fit_transform(target.Occupied)
target.Price = label_processor.fit_transform(target.Price)
target.Music = label_processor.fit_transform(target.Music)
target.Location = label_processor.fit_transform(target.Location)
target['Favorite Beer'] = label_processor.fit_transform(target['Favorite Beer'])

lables = target['Enjoy']
target = target.drop(labels='Enjoy', axis=1)

#print(target)
#print(lables)
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf = clf.fit(target, lables)
print(clf.predict([[0,2,0,0,0,0,0]]))
#(size = Large; occupied = Moderate; price = Cheap; music = Loud; location = City-Center; VIP = No; favorite beer = No).

# import pydotplus
# dot_data = tree.export_graphviz(clf, out_file=None, feature_names = target.columns, class_names=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("clf.pdf")



