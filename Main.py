
# coding: utf-8

# In[ ]:


#the beginning of the code 
#input: raw data in json format
#output: result with emoji number

#import other files
from HandleRawData import HandleRawData
from Ngram import NgramModel

#open raw data file
with open('./newUniData.json', 'r') as f:
    rawData = f.readlines()

#handle raw data
#output: a txt file 'count_sort_emoji.txt' --> count the number of comment with emoji and sort in descending order
handleRawData = HandleRawData(rawData)

#"classifyComment" function: classify comments by emoji
#parameter: top_number: int -> how many kind of comments<emoji>
#return: comments[][] -> row is categoty; col is comment
comments = handleRawData.classifyComment(10)

#"writeToFile" function: write classifyComment" function result to file
#parameter: 1. top_number: int -> how many kind of comments<emoji>
#           2. dirctory path: str -> file will be named as 0.txt; 1.txt...
handleRawData.writeToFile(10, './rawClassifyData/')


#输出为二维数组,[traingData][testData]
#"sperateRawData" function: sperate data to training data and test data(8:2)
#parameter: 1. classifyComment:[][] ->  row is category; col is comments <result from "classifyComment" function>
#           2. dataTotalNumber:int -> how many data do you want to put
#return: [] -> [0] is training data([][]); [1]: test data
data = handleRawData.sperateRawData(comments, 2500)

#traing data
train = data[0]
#test data
test = data[1]

#Ngram Model
#input: training data -> calculate probability of each kind of emoji
#"start" function -> start to calculate test data
model = NgramModel(train)

result = model.start(test)
perplexity = model.perplexity(test)


