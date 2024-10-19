#!/usr/bin/env python
# coding: utf-8

# In[5]:

'''
# import libraries
import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import Main
#from Main import test_data, test, MODELS, encoderRNN, decoderRNN, attention
import pickle


# In[9]:


if not torch.cuda.is_available():
    modelIP = torch.load('SavedModel/model1.h5', map_location=lambda storage, loc: storage)
else:
    modelIP = torch.load('SavedModel/model1.h5')


# In[10]:


files_dir = 'testing_data/feat'
i2w,w2i,dictonary = Main.dictonaryFunc()


test_dataset = Main.test_dataloader(files_dir)
test_dataloader = Main.DataLoader(dataset = test_dataset, batch_size=1, shuffle=True, num_workers=8)

# with open('i2wData.pickle', 'rb') as f:
#     i2w = pickle.load(f)

model = modelIP.cuda()

ss = Main.test(test_dataloader, model, i2w)

with open('test_output.txt', 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))


# In[11]:


# Bleu Eval
test = json.load(open('testing_label.json','r'))
#output = 'testing_data.txt'
output = 'test_output.txt'
result = {}

with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
#count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))


# In[ ]:





# In[ ]:

'''

#!/usr/bin/env python
# coding: utf-8

# Import libraries
import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import Main
import pickle

# Model loading
if not torch.cuda.is_available():
    # Loading the model without GPU and with weights_only set to False for safety
    modelIP = torch.load('SavedModel/model1.h5', map_location=lambda storage, loc: storage)
else:
    # Loading the model with CUDA if available
    modelIP = torch.load('SavedModel/model1.h5')

# Setting up files directory and initializing dictionary
files_dir = 'testing_data/feat'
word_min = 5  # Assuming word_min is 5; adjust as needed
i2w, w2i, dictionary = Main.dictonaryFunc(word_min)

# Preparing data loader for testing, reducing num_workers to 1
test_dataset = Main.test_dataloader(files_dir)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=1)

# Ensure model is on the correct device (CUDA if available)
if torch.cuda.is_available():
    model = modelIP.cuda()
else:
    model = modelIP

# Running the test function
ss = Main.test(test_dataloader, model, i2w)

# Writing the test output to a file
with open('test_output.txt', 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))

# BLEU Evaluation
test = json.load(open('testing_label.json', 'r'))
output = 'test_output.txt'
result = {}

# Reading output from the test file
with open(output, 'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma + 1:]
        result[test_id] = caption

# Computing BLEU score
bleu = []
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']], captions, True))
    bleu.append(score_per_video[0])

# Calculating and printing the average BLEU score
average = sum(bleu) / len(bleu)
print("Average BLEU score is " + str(average))

