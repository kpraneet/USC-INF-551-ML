import pandas as pd
import math
#-----------------------------calculates entropy of given attribute-----------------------------------------------------
#calculates entropy of given attribute
def entropy(example, classification_attribute):
    attributePositiveCounter= {}
    attributeNegativeCounter = {}
    for index, data in enumerate(example):
        if classification_attribute[index]:
            if(data in attributePositiveCounter):
                attributePositiveCounter[data] += 1
            else:
                attributePositiveCounter[data] = 1
        else:
            if (data in attributeNegativeCounter):
                attributeNegativeCounter[data] += 1
            else:
                attributeNegativeCounter[data] = 1
    averageEntropy = 0
    for singleAttribute in example.unique():
        total = len(example)
        if singleAttribute in attributePositiveCounter:
            posCounter = attributePositiveCounter[singleAttribute]
        else:
            posCounter = 0
        if singleAttribute in attributeNegativeCounter:
            negCounter = attributeNegativeCounter[singleAttribute]
        else:
            negCounter = 0
        totalSingleAttribute = posCounter + negCounter
        if posCounter==0 or negCounter ==0:
            averageEntropy +=0
        else:
            positiveEntropy = ((posCounter/totalSingleAttribute) * math.log2(totalSingleAttribute / posCounter))
            negativeEntropy = ((negCounter / totalSingleAttribute) * math.log2(totalSingleAttribute / negCounter))
            averageEntropy += (totalSingleAttribute/total)*(positiveEntropy + negativeEntropy)
    return  averageEntropy

#-----------------seperates given attribute into dataframes of all of its values----------------------------------------

def seperateBranchesOfAttribute(example, attribute):
    branchOfAttributes = {}
    # numberOfUniqueValues = len(originalData[attribute].unique())
    # for index in range (numberOfUniqueValues):
    numberOfUniqueValues = originalData[attribute].unique()
    for index in numberOfUniqueValues:
        branchOfAttributes[index] = example.loc[example[attribute] == index]
        branchOfAttributes[index] = branchOfAttributes[index].drop(labels=attribute, axis=1)
        branchOfAttributes[index] = branchOfAttributes[index].reset_index(drop=True)
        # print(branchOfAttributes[index])
    return branchOfAttributes

#-----------------------prepares initial data frame---------------------------------------------------------------------

def prepareData():
    target = pd.read_csv("dt-data.txt",
                         names=['Size', 'Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy'],
                         skipinitialspace=True, skiprows=[0], index_col=False)
    target['Size'] = target['Size'].str.replace('\d+:', '')
    target['Enjoy'] = target['Enjoy'].str.replace(';', '')
    from sklearn import preprocessing
    label_processor = preprocessing.LabelEncoder()
    target.VIP = label_processor.fit_transform(target.VIP)
    vipKeys = {}
    for val in target.VIP.unique():
        vipKeys[val] = label_processor.inverse_transform(val)
    target.Enjoy = label_processor.fit_transform(target.Enjoy)
    target.Size = label_processor.fit_transform(target.Size)
    sizeKeys = {}
    for val in target.Size.unique():
        sizeKeys[val] = label_processor.inverse_transform(val)
    target.Occupied = label_processor.fit_transform(target.Occupied)
    occKeys = {}
    for val in target.Occupied.unique():
        occKeys[val] = label_processor.inverse_transform(val)
    target.Price = label_processor.fit_transform(target.Price)
    priceKeys = {}
    for val in target.Price.unique():
        priceKeys[val] = label_processor.inverse_transform(val)
    target.Music = label_processor.fit_transform(target.Music)
    musicKeys = {}
    for val in target.Music.unique():
        musicKeys[val] = label_processor.inverse_transform(val)
    target.Location = label_processor.fit_transform(target.Location)
    locKeys = {}
    for val in target.Location.unique():
        locKeys[val] = label_processor.inverse_transform(val)
    target['Favorite Beer'] = label_processor.fit_transform(target['Favorite Beer'])
    beerKeys = {}
    for val in target['Favorite Beer'].unique():
        beerKeys[val] = label_processor.inverse_transform(val)
    global inverseKeys
    inverseKeys = {'Size':sizeKeys, 'Occupied':occKeys, 'Price':priceKeys, 'Music':musicKeys, 'Location':locKeys, 'VIP':vipKeys, 'Favorite Beer':beerKeys}


# -------------------------------------------Tennis Data--------------------------------------------------------------
#     target = pd.read_csv("tennis.csv",
#                          names=['outlook', 'temp', 'humidity', 'windy', 'play'],
#                          skipinitialspace=True, skiprows=[0], index_col=False)
#     from sklearn import preprocessing
#     label_processor = preprocessing.LabelEncoder()
#     target.outlook = label_processor.fit_transform(target.outlook)
#     outlookKeys = {}
#     for val in target.outlook.unique():
#         outlookKeys[val]=label_processor.inverse_transform(val)
#     target.temp = label_processor.fit_transform(target.temp)
#     tempKeys = {}
#     for val in target.temp.unique():
#         tempKeys[val] = label_processor.inverse_transform(val)
#     target.humidity = label_processor.fit_transform(target.humidity)
#     humKeys = {}
#     for val in target.humidity.unique():
#         humKeys[val] = label_processor.inverse_transform(val)
#     target.windy = label_processor.fit_transform(target.windy)
#     winKeys = {}
#     for val in target.humidity.unique():
#         winKeys[val] = label_processor.inverse_transform(val)
#     target.play = label_processor.fit_transform(target.play)
#     global inverseKeys
#     inverseKeys = {"outlook":outlookKeys, "temp":tempKeys, "humidity":humKeys, "windy":winKeys}
    return target
#Make a prediction for (size = Large; occupied = Moderate; price = Cheap; music = Loud; location = City-Center; VIP = No; favorite beer = No).

#-----------------------------------------------------------------------------------------------------------------------

def findRootNode(target, labels):
    # branch_values = seperateBranchesOfAttribute(target, 'Size')
    initialEntropies = [0]*len(target.columns)
    for index, column in enumerate(target):
       initialEntropies[index] = entropy(target[column], labels)
    # print("printing entropy")
    # print(min(initialEntropies))
    # if(min(initialEntropies)==0):
    #     return -1
    # else:
    return initialEntropies.index(min(initialEntropies))

#----------------------------------ID3 Algorithm -----------------------------------------------------------------------

#main algorithm
def id3(examples, classification_attribute ,attributes):
    #create a root node for the tree
   decisionTree = DecisionTree()
   isAllPositive = True
   posCounter = 0
   isAllNegative = True
   negCounter = 0
   for classificationAttribute in classification_attribute:
       if classificationAttribute == 0:
           isAllPositive=False
           negCounter += 1
       if classificationAttribute == 1:
           isAllNegative =False
           posCounter += 1
    #if all examples are positive/yes: return root node with positive/yes label
   if isAllPositive:
       # print("All Positive and returning")
       decisionTree.label = 'yes'
       return decisionTree
    #else if all examples are negative/no: return root node with negative/nolabel
   if isAllNegative:
       # print("All Negative and returning")
       decisionTree.label = 'no'
       return decisionTree
    #else if there are no attributes left : return root node with most popular
   if not len(attributes):
       if posCounter>negCounter:
           decisionTree.label = 'yes'
       if posCounter==negCounter:
           decisionTree.label = 'tie'
       else:
           decisionTree.label = 'no'
       return decisionTree
   else:
       #print("searching for best attribute")
       example = examples.drop(labels='Enjoy', axis=1)
       best_attribute = findRootNode(example, classification_attribute)
       decisionTree.label = [example.columns[best_attribute]]
       # printTree(decisionTree)
       # if entropy(example[example.columns[best_attribute]], classification_attribute) == 0:
       #     print("Terminating at")
       #     print(example.columns[best_attribute])
       #     return 0
       #assign best_attribute to root node
       # print(best_attribute)
       branch_examples = seperateBranchesOfAttribute(examples, example.columns[best_attribute])
       # print(branch_examples)
       #for each value in best_attribute: add branch below root node for the value
       if not branch_examples:
           if posCounter > negCounter:
               decisionTree.label = 'yes'
           if posCounter == negCounter:
               decisionTree.label = 'tie'
           else:
               decisionTree.label = 'no'
               return decisionTree
       #if branch_examples is empty : add leaf npde with most popular label
        #else subtree id3(branch_examples, classification_attribute, attributes-best attribute)
       for branch in branch_examples:
           label = branch_examples[branch]['Enjoy']
           retVal = id3(branch_examples[branch], label, branch_examples[branch].columns)
           decisionTree.children.append(retVal)
           # if(retVal==-1):
               # print("Terminating at")
               # print(inverseKeys[example.columns[best_attribute]][branch])
   return decisionTree

# ----------------------------------------------------------------------------------------------------------------------

class DecisionTree(object):
    def __init__(self):
        self.label = None
        self.children = []
# ---------------------------------print the decision tree - level order traversal--------------------------------------

def printTree(tree):
    if tree is None:
        return
    count = [1,0]    # Number of nodes in each level
    from collections import deque
    queue = deque()
    queue.append(tree)
    node = None
    level = 0  # current level
    nodeCount = 0 # number of nodes dequeued in current level
    while (len(queue) > 0):
        node = queue.popleft()
        for child in node.children:
            queue.append(child)
        count[level+1] = count[level+1]+len(node.children)
        print (node.label,end=" ")
        nodeCount += 1
        if (nodeCount == count[level]):
            nodeCount = 0
            level += 1
            count.append(0)
            print("\n")

# --------------------------------------------------------------------------------------------------------------------

# def predict():
#
#     (size = Large;
#     occupied = Moderate;
#     price = Cheap;
#     music = Loud;
#     location = City - Center;
#     VIP = No;
#     favorite
#     beer = No).

# get root node label
# get value of the data for that label
#
#

#---------------------------------------------------------------------------------------------------------------------
#main method

target = prepareData()
global originalData
originalData = target
labels = target['Enjoy']
#target = target.drop(labels='play', axis=1)
# global decisionTree
# decisionTree = DecisionTree()
id3(target, labels, target.columns)
printTree(id3(target, labels, target.columns))
# indexOfMinAttribute = findRootNode(target, labels)
# branches = seperateBranchesOfAttribute(originalData, target.columns[indexOfMinAttribute])
# plays = branches[0].play
# branches[0] = branches[0].drop(labels='play', axis=1)
# print(findRootNode(branches[0], plays))






















