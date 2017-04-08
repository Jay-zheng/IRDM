import numpy as np
import pandas as pd

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")
testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")
attribute_data = pd.read_csv('../input/attributes.csv')
descriptions = pd.read_csv('../input/product_descriptions.csv')

print("Test data columns")
print(testing_data.columns)
print("Attribute data columns")
print(attribute_data.columns)
print("Description data columns")
print(descriptions.columns)

training_data = pd.merge(training_data, descriptions,
                         on="product_uid", how="left")
print(training_data.columns)
print(training_data.head())

product_counts = pd.DataFrame(pd.Series(training_data.groupby(
['product_uid']).size(), name='product_count'))

training_data = pd.merge(training_data, product_counts,
                        left_on="product_uid", right_index=True,
                        how="left")
print(training_data[:20])

brand_names = attribute_data[attribute_data.name == "MFG Brand Name"][['product_uid', 'value']].rename(columns={"value": "brand_name"})
training_data = pd.merge(training_data, brand_names, on="product_uid", how="left")

print(brand_names)

print(str(training_data.info()))

# We note that brand_name has null values and replace them with "Unknown"
# Fill all products with no brand name
training_data.brand_name.fillna("Unknown", inplace=True)
print(str(training_data.describe()))

(descriptions.product_description.str.len()/5).hist(bins=30)
(training_data.product_title.str.len() / 5).hist(bins=30)
(training_data.search_term.str.count("\\s+") + 1).hist(bins=30)

(training_data.search_term.str.count("\\s+") + 1).describe()
attribute_counts = attribute_data.name.value_counts()
print(attribute_counts)