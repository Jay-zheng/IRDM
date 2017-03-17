import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as sw
import string

# Fill in name of CSV file
file = "train.csv"
# Dataframe for the data
data = pd.read_csv(file, encoding="ISO-8859-1")

# Remove punctuation
def remove_punct(data):
    # Remove all punctuation using a lambda function
    data = data.apply(
        lambda x: ''.join([i for i in x if i not in string.punctuation])
    )
    return data

# Remove stopwords from the data
def remove_stopwords(data):
    # Convert to lower case
    data = data.str.lower()
    # Get the stopwords
    stopwords = set(sw.words("english"))
    # Use pandas to replace all instances of stopwords
    data = data.replace(to_replace=stopwords, value="")
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

# Preprocess the "product_title" and "search_term" columns of the data
data["product_title"] = preprocess(data["product_title"])
data["search_term"] = preprocess(data["search_term"])

data.to_csv("train_stemmed.csv", index=False)