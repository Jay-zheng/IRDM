import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances

# All the data
data = pd.read_csv("all_info.csv")
poo = data.iloc[0]

# Initialise vectoriser with an N-Gram range of 2 (bigram)
vec = TfidfVectorizer()

"""
Find the cosine distance of the TF-IDF vectors of data in 2 columns

Cosine dist = 1 - cosine similarity = 1 - ((dt1.dt2)/||dt1||||dt2||)
"""
def cosine_dist(data,column1,column2):
    # Get the data from the relevant columns
    d1 = data[column1] # Column 1
    d2 = data[column2] # Column 2

    # Deal with missing data
    d2 = d2.fillna(value="none")

    # Learn the vocabulary and IDF of the two columns/document
    vec.fit(d1 + " " + d2)

    # Turn each document into a document term matrix
    dt1 = vec.transform(d1)
    dt2 = vec.transform(d2)

    # Compute the cosine distance of the two vectors
    result = paired_distances(dt1,dt2,"cosine")

    # Return the result
    return result

# Make a new, blank dataframe
df = pd.DataFrame()

# Make cosine distances for:
#   - search term vs. product title
#   - search term vs. brand
#   - search term vs. attr
#   - search term vs. attr_title

df["id"] = data["id"]
df["search_vs_pro_title"] = cosine_dist(data,"search_term","product_title")
df["search_vs_brand"] = cosine_dist(data,"search_term","brand")
df["search_vs_attr"] = cosine_dist(data,"search_term","attr")
df["search_vs_attr_title"] = cosine_dist(data,"search_term","attr_title")
df["relevance"] = data["relevance"]

df.to_csv("alice_feats.csv",index=False,encoding="utf-8")
























