
# All the necessary libraries
import re
from nltk.stem.snowball import SnowballStemmer
import string
import time
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from Spell_Checker_Dict import spell_check_dict as spellcheck_dict # a dictionary of all words that are spelt wrong 
from Google_Search import spell_check as spellcheck # google spell checker from kaggle kernel
from tqdm import tqdm # really cool module for showing progress of loops


# Load data  into dataframes
df_train = pd.read_csv(".../home-depot/data/train.csv",encoding="ISO-8859-1")
df_test = pd.read_csv(".../home-depot/data/test.csv",encoding="ISO-8859-1")
df_pro_desc  = pd.read_csv(".../home-depot/data/product_descriptions.csv")
df_attr = pd.read_csv(".../home-depot/data/attributes.csv")

stopwords = open("stopwords.txt",'r').readlines()
stopwords = [x.strip() for x in stopwords]

def remove_punct(data):
    # Remove all punctuation using lambda function
    data = data.apply(
        lambda x: ''.join([i for i in x if i not in string.punctuation])
    )
    return data

# Remove stopwords.txt from the data
def remove_stopwords(data):
    # Convert to lower case
    data = data.str.lower()

    # Replace all instances of stopwords
    index = 0

    # Had to brute force with a for loop
    for index in range(len(data)):
        # Break up the sentence
        sentence = data.iloc[index].split(" ")
        # Remove stopwords
        sentence = [word for word in sentence if word not in stopwords]
        # Rejoin the sentence
        sentence = " ".join(sentence)
        # Reassign sentence
        data.iloc[index] = sentence

    # Return the data
    return data



# helol wold -> hello world
# only uses predefined spell check dict atm and it seems to be enough
# google spellcheck does not make much difference
def spellchecker(searchwords):
    if searchwords in spellcheck_dict.keys():
        return spellcheck_dict[searchwords]
    else:
        return searchwords




# predefined regex process function to make it easier to run all the below functions
def regex_processor(text, replace_list):
    for pattern, replace in replace_list:
        try:
            text = re.sub(pattern, replace, text)
        except:
            pass
    return re.sub(r"\s+", " ", text).strip()


# change units like inches to in, pounds to lb
def convertunits(text):
    replace_list = [
        (r"([0-9]+)( *)(inches|inch|in|in.|')\.?", r"\1 in. "),
        (r"([0-9]+)( *)(pounds|pound|lbs|lb|lb.)\.?", r"\1 lb. "),
        (r"([0-9]+)( *)(foot|feet|ft|ft.|'')\.?", r"\1 ft. "),
        (r"([0-9]+)( *)(square|sq|sq.) ?\.?(inches|inch|in|in.|')\.?", r"\1 sq.in. "),
        (r"([0-9]+)( *)(square|sq|sq.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 sq.ft. "),
        (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(inches|inch|in|in.|')\.?", r"\1 cu.in. "),
        (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 cu.ft. "),
        (r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gal. "),
        (r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 oz. "),
        (r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 cm. "),
        (r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 mm. "),
        (r"([0-9]+)( *)(minutes|minute)\.?", r"\1 min. "),
        (r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. "),
        (r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. "),
        (r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. "),
        (r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. "),
        (r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. "),
        (r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr "),
        (r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?",
         r"\1 gal. per min. "),
        (r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr "),
    ]
    return regex_processor(text, replace_list)







# 1x1 -> 1 x 1
def digitsplitters(text):
    replace_list = [
        (r"(\d+)[\.\-]*([a-zA-Z]+)", r"\1 \2"),
        (r"([a-zA-Z]+)[\.\-]*(\d+)", r"\1 \2"),
    ]
    return regex_processor(text, replace_list)





# convert string numbers to numberss like one to 1
def numberconverter(text):
    numbers = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"
    ]
    digits = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 140, 15,
        16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000
    ]
    replace_list = [
        (r"%s" % n, str(d)) for n, d in zip(numbers, digits)
        ]
    return regex_processor(text, replace_list)


# remove special characters just in case
def specialcharcleaner(text):
    replace_list = [
        (r"<.+?>", r""),
        (r"&nbsp;", r" "),
        (r"&amp;", r"&"),
        (r"&#39;", r"'"),
        (r"/>/Agt/>", r""),
        (r"</a<gt/", r""),
        (r"gt/>", r""),
        (r"/>", r""),
        (r"<br", r""),
        (r"[ &<>)(_,;:!?\+^~@#\$]+", r" "),
        ("'s\\b", r""),
        (r"[']+", r""),
        (r"[\"]+", r""),
    ]
    return regex_processor(text, replace_list)


# handle all HTML tags if any exist
def HtmlCleaner(text, parser='html.parser'):
    bs = BeautifulSoup(text, parser)
    text = bs.get_text(separator=" ")
    return text




# stemming
def stem(data):
    # Use English Snowball Stemmer (from NLTK)
    stemmer = SnowballStemmer("english")
    token = [stemmer.stem(token) for token in data.split(" ")]

    return " ".join(token)


# Apply all data cleaning
def preprocess1(data):
    data=remove_punct(data)
    data = remove_stopwords(data)
    return data
def preprocess2(df_col):
    cleaner_funcs = [
        spellchecker,
        convertunits,
        digitsplitters,
        numberconverter,
        specialcharcleaner,
        HtmlCleaner,
        stem
    ]

    for func in tqdm(cleaner_funcs):
        df_col = df_col.apply(lambda x: func(str(x)))
    return df_col

    # ==============================================================================


start_train_time = time.time()
print("--- Cleaning Train Data---")

# Clean train data
df_train.product_title = preprocess1(df_train.product_title)
df_train.product_title = preprocess2(df_train.product_title)
df_train.search_term = preprocess1(df_train.search_term)
df_train.search_term = preprocess2(df_train.search_term)


print(" :) Train Data Cleaning Finished in %s minutes" % round(((time.time() - start_train_time)/60),2))

start_test_time = time.time()
print("--- Cleaning test Data---")

# Clean test data
df_test.product_title = preprocess1(df_test.product_title)
df_test.product_title = preprocess2(df_test.product_title)
df_test.search_term = preprocess1(df_test.search_term)
df_test.search_term = preprocess2(df_test.search_term)


print(" :) test Data Cleaning Finished in %s minutes" % round(((time.time() - start_test_time)/60),2))



start_description_time = time.time()
print("--- Cleaning desc Data---")

# Clean desc data
df_pro_desc.product_description = preprocess1(df_pro_desc.product_description)
df_pro_desc.product_description = preprocess2(df_pro_desc.product_description)



print(" :) product description Data Cleaning Finished in %s minutes" % round(((time.time() - start_description_time)/60),2))

start_att_time = time.time()
print("--- Cleaning attr Data---")

# Clean attribute data

df_attr.name = preprocess2(df_attr.name)


df_attr.value = preprocess2(df_attr.value)

print(" :) product attribute Data Cleaning Finished in %s minutes" % round(((time.time() - start_att_time)/60),2))

df_train.to_csv('.../home-depot/data/train_clean.csv')
df_test.to_csv('.../home-depot/data/test_clean.csv')
df_pro_desc.to_csv('.../data/product_descriptions_clean.csv', index=False)
df_attr.to_csv('.../data/attributes_clean.csv',index=False)
