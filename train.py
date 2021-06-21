from __future__ import unicode_literals
from flask import Flask, render_template, request, redirect, url_for
import urllib
#import unirest #In an attempt to update the library, unirest requires Python 2.7
                #which is getting updated to Python 3 which means we can now use requests instead
import difflib
from goose3 import Goose
import io
import os
import sys, csv
import tensorflow as tf
import requests
import json
import re
#import importlib #Used to reload modules
#importlib.reload(sys)
#sys.setdefaultencoding('utf8')
csv.field_size_limit(sys.maxsize)

#import fake_news_rn
import nltk
from defend import *
defend = Defend("abc")

# I. Antrenam un model cateva epoci
# iei inputul din politifact
# chemi train

newsdic = {}
platform = 'politifact'
saved_model_filename = 'politifact_Defend_model.h5'

comments = []
contents = []
labels = np.array(0)

file_title = './static/' + platform + '_title_no_ignore.tsv'
with open(file_title) as tsvfile:
    reader = csv.reader(tsvfile, delimiter=str(u'\t'))
    for row in reader:
        id = row[0]
        newsdic[id] = {}
        newsdic[row[0]]['title'] = row[1]

file_comment_our = './static/' + platform + '_comment_no_ignore.tsv'
with open(file_comment_our) as tsvfile:
    reader = csv.reader(tsvfile, delimiter=str(u'\t'))
    for row in reader:
        if row[0] in newsdic.keys():
            newsdic[row[0]]['comment_our'] = row[1].split("::")
            comments.append(newsdic[row[0]]['comment_our'])

file_content = './static/' + platform + '_content_no_ignore.tsv'
with open(file_content) as tsvfile:
    reader = csv.reader(tsvfile, delimiter=str(u'\t'))
    for row in reader:
        if row[0] in newsdic.keys():
            data = row[2].replace("?", ".")
            data = data.replace("!", ".")

            newsdic[row[0]]['content'] = data.split(".")
            newsdic[row[0]]['label'] = row[1]
            contents.append(newsdic[row[0]]['content'])
            labels = np.append(labels, newsdic[row[0]]['label'])

#print(len(labels))
train_y = np.concatenate((labels[1:210], labels[272:372]), axis=0)
train_y = np.vstack((train_y, train_y)).T
val_y = np.concatenate((labels[211:271], labels[373:416]), axis=0)
val_y = np.vstack((val_y, val_y)).T
#print(comments[2])
#print(np.shape(val_y))
#print(contents[1])
#print(train_y)
#print(newsdic['politifact14856'])
defend.train(contents[0:209] + contents[271:371], train_y, comments[0:209] + comments[271:371], comments[210:270] + comments[372:415], contents[210:270] + contents[372:415], val_y, saved_model_filename)
