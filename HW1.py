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

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(target, lables)

# expected = target
# predicted = clf.predict(target)
# out = []

# Test with training data
# print("Test case outputs")
# for i in range(0, len(predicted)):
#     if predicted[i] == 0:
#         out.insert(i,'No')
#     else:
#         out.insert(i,'Yes')
#     print(out[i])




