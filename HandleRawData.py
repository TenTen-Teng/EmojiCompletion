
# coding: utf-8

# In[29]:


#handle raw data

import json
from nltk import word_tokenize
import re


class JSONObject:
    def __init__(self, d):
        self.__dict__ = d


class HandleRawData:
    def __init__(self, rawData):
        self.rawData = rawData
        self.sortedEmoji = HandleRawData.getSortedEmojiData(self)
    
    ##############################################################
    #category raw data by emoji and sort
    #return a list(top emoji title)
    ##############################################################
    def getSortedEmojiData(self):     
        #store top emoji title
        title = []
        
        #store emoji and its number: key -> emoji; value: number
        emoji_pair = {}
        
        #output path
        emoji_sort_path = './count_sort_emoji.txt'
        
        count = 0
        for item in self.rawData:
            count += 1
            
            #tranfer raw data to json object
            pair = json.loads(item, object_hook=JSONObject)
    
            #add emoji and its number to emoji_pair
            emoji_pair[pair.title] = emoji_pair.setdefault(pair.title, 0) + 1

        #sort emoji_pair by value(emoji number)
        emoji_sort = sorted(emoji_pair.items(), key=lambda d:d[1], reverse=True) 
        
        #store top emoji titles
        for item in emoji_sort:
            title.append(''.join(str(v) for v in item[0]))

        #write result to file
        with open(emoji_sort_path, 'w') as f:
            top_number = 0
            for item in emoji_sort:
                output = []
                output_title = ''.join(str(v) for v in item[0])
                output_count = str(item[1])+'\n'
                output.append(str(top_number))
                output.append(output_title)
                output.append(output_count)
                output = '**'.join(output)
                f.write(output)
                top_number += 1
        return title
    
    #################################################################
    #category comments by emoji
    #input: top number: int
    #return: a two-dimensional array -> row: emoji kind; col: comment
    #################################################################
    def classifyComment(self, top_number):
        #a list for top emoji titles
        top = self.sortedEmoji[0:top_number]
        
        #a two-dimensional array -> row: emoji kind; col: comment
        comments = [[]for i in range(top_number)] 

        #category comments
        for item in self.rawData:
            pair = json.loads(item, object_hook=JSONObject)
            for i in range(len(top)):
                if pair.title == top[i]:
                    comments[i].append(pair.content) 
        return comments
    
    
    #################################################################
    #write "classifyComment" function result to file
    #input: top number: int; dirctory path: str -> file will be named
    #                                              as 0.txt; 1.txt...
    #################################################################
    def writeToFile(self, top_number, path):
        kinds = HandleRawData.classifyComment(self, top_number)
        
        for i in range(len(kinds)):
            fileName = path + str(i) + '.txt'
            with open(fileName , 'w') as f:
                for item in kinds[i]:
                    f.write(item)
                    f.write('\n')
                    
    #################################################################
    #sperate data to training data and test data(8:2)
    #input: classifyComment:[][] ->  row is category; col is comments 
    #       <result from "classifyComment" function>
    #       dataTotalNumber:int -> how many data do you want to put
    #return: [] -> [0] is training data([][]); [1]: test data
    #################################################################
    def sperateRawData(self, classifyComment, dataTotalNumber):
        kindsNumber = len(classifyComment)
        test_data = []
        traingData = [[] for i in range(kindsNumber)]
        training_number = int(dataTotalNumber * 0.8)
        test_number = int(dataTotalNumber * 0.2)
        
        for i in range(kindsNumber):
            count_train = 0
            for element in classifyComment[i]:
                element = re.sub(r'\n', r'', element)
                preprocess_ele = word_tokenize(element)
                if len(preprocess_ele) >= 4 and count_train < training_number:
                    traingData[i].append(element)
                    count_train += 1
            
            count_test = 0
            for element in classifyComment[i][count_train: ]:
                element = re.sub(r'\n', r'', element)
                preprocess_ele = word_tokenize(element)
                if len(preprocess_ele) >= 4 and count_test < test_number:
                    test_data.append(element)
                    count_test += 1

    
    
        trainPath = './trainingData/'
        for i in range(len(traingData)):
            fileName = trainPath + str(i) + '.txt'
            with open(fileName , 'w') as f:
                for item in traingData[i]:
                    f.write(item)
                    f.write('\n')

        testPath = './testData/test.txt'
        with open(testPath , 'w') as f:
            for item in test_data:
                f.write(item)
                f.write('\n')

        goldPath = './gold.txt'
        with open(goldPath , 'w') as f:
            count = 0
            i = -1
            for item in test_data:
                if count % 1000 == 0:
                    i += 1
                    f.write(str(i))
                    f.write('\n')
                else:
                    f.write(str(i))
                    f.write('\n')
                count += 1
        
        return traingData, test_data

    # #################################################################
    # # sperate data to training data and test data(8:2)
    # #remove stop words
    # # input: classifyComment:[][] ->  row is category; col is comments
    # #       <result from "classifyComment" function>
    # #       dataTotalNumber:int -> how many data do you want to put
    # # return: [] -> [0] is training data([][]); [1]: test data
    # #################################################################
    # def sperateRawData(self, classifyComment, dataTotalNumber):
    #     kindsNumber = len(classifyComment)
    #     test_data = []
    #     traingData = [[] for i in range(kindsNumber)]
    #     training_number = int(dataTotalNumber * 0.8)
    #     test_number = int(dataTotalNumber * 0.2)
    #     stop_word = set(stopwords.words("English"))
    #
    #     for i in range(kindsNumber):
    #         count_train = 0
    #         for element in classifyComment[i]:
    #             element = re.sub(r'\n', r'', element)
    #             preprocess_ele = word_tokenize(element)
    #
    #             filtered_word = [w for w in preprocess_ele if w not in stop_word]
    #
    #             if len(filtered_word) >= 4 and count_train < training_number:
    #                 # remove stop words
    #                 # traingData[i].append(' '.join(filtered_word))
    #                 traingData[i].append(element)
    #                 count_train += 1
    #
    #         count_test = 0
    #         for element in classifyComment[i][count_train:]:
    #             element = re.sub(r'\n', r'', element)
    #             preprocess_ele = word_tokenize(element)
    #
    #             filtered_word = [w for w in preprocess_ele if w not in stop_word]
    #
    #             if len(filtered_word) >= 4 and count_test < test_number:
    #                 # remove stop words
    #                 # test_data.append(' '.join(filtered_word))
    #                 test_data.append(element)
    #                 count_test += 1
    #
    #     trainPath = './trainingData/'
    #     for i in range(len(traingData)):
    #         fileName = trainPath + str(i) + '.txt'
    #         with open(fileName, 'w') as f:
    #             for item in traingData[i]:
    #                 f.write(item)
    #                 f.write('\n')
    #
    #     testPath = './testData/test.txt'
    #     with open(testPath, 'w') as f:
    #         for item in test_data:
    #             f.write(item)
    #             f.write('\n')
    #
    #     goldPath = './gold.txt'
    #     with open(goldPath, 'w') as f:
    #         count = 0
    #         i = -1
    #         for item in test_data:
    #             if count % 1000 == 0:
    #                 i += 1
    #                 f.write(str(i))
    #                 f.write('\n')
    #             else:
    #                 f.write(str(i))
    #                 f.write('\n')
    #             count += 1
    #
    #     return traingData, test_data
