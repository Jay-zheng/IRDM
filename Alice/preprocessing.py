import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import string

# Fill in name of CSV file
file = "test.csv"
# Dataframe for the data
data = pd.read_csv(file, encoding="ISO-8859-1")

# Get stopwords from a list of stopwords
# NLTK's stopword list does not remove some common words such as "the"
stopwords = open("stopwords.txt",'r').readlines()
stopwords = [x.strip() for x in stopwords]

# Remove punctuation
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

# Stem words (e.g. pooing -> poo)
def stem(data):
    # Use English Snowball Stemmer (from NLTK)
    stemmer = SnowballStemmer("english")
    # Stem all words in data
    data = data.apply(lambda x: stemmer.stem(x))
    return data

# Put it all together!
def preprocess(data):
    data = remove_punct(data)
    data = remove_stopwords(data)
    data = stem(data)
    return data

# Preprocess the columns you want
data["product_title"] = preprocess(data["product_title"])
data["search_term"] = preprocess(data["search_term"])

# Save to CSV
data.to_csv("test_stemmed.csv", index=False)