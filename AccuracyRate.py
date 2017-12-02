
# coding: utf-8

# In[1]:


gold = []
with open('./gold.txt', 'r') as f:
    gold = f.readlines()
    
result = []
with open('./result.txt', 'r') as f:
    result = f.readlines()
    
accuracy_count = 0
for i in range(len(gold)):
    if gold[i] == result[i]:
        accuracy_count += 1
print('accuracy_count = ', accuracy_count)

accuracy_rate = accuracy_count / len(gold)
print('accuracy_rate = ' , accuracy_rate)

