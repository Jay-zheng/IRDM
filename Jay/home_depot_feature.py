import numpy as np
import pandas as pd
import utils

df_train = pd.read_csv(".../home-depot/data/train_clean.csv",encoding="ISO-8859-1")
df_test = pd.read_csv(".../home-depot/data/test_clean.csv",encoding="ISO-8859-1")
df_desc = pd.read_csv(".../home-depot/data/product_descriptions_clean.csv")
df_attr = pd.read_csv(".../home-depot/data/attributes_clean.csv",encoding = "ISO-8859-1")

df_train = df_train.drop(df_train.columns[0],axis = 1)
df_test = df_test.drop(df_test.columns[0],axis = 1)
numtrain = df_train.shape[0]

df_brand = df_attr[df_attr.name == "mfg brand name"][["product_uid", "value"]].rename(columns={"value": "brand"})

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

for var in df_all.keys():
    df_all[var]=df_all[var].fillna("na")

attr_all = df_attr[df_attr.name != 'mfg brand name'][['product_uid', 'value']].rename(columns={'value': 'attr'})
attr_all['attr'] = attr_all['attr'].apply(lambda x: x if isinstance(x, str) else 'None')
attr_all = attr_all.groupby('product_uid')['attr'].apply(lambda x: ' '.join(x))
attr_all = pd.DataFrame(attr_all)
attr_all = attr_all.reset_index()

attr_all_title = df_attr[df_attr.name != 'mfg brand name'][['product_uid', 'name']].rename(columns={'name': 'attr_title'})
attr_all_title['attr_title'] = attr_all_title['attr_title'].apply(lambda x: x if isinstance(x, str) else 'None')
attr_all_title = attr_all_title.groupby('product_uid')['attr_title'].apply(lambda x: ' '.join(x))
attr_all_title = pd.DataFrame(attr_all_title)
attr_all_title = attr_all_title.reset_index()

df_all = pd.merge(df_all, attr_all, how='left', on='product_uid')
df_all = pd.merge(df_all, attr_all_title, how='left', on='product_uid')

df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(str(x).split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x: len(str(x).split())).astype(np.int64)
df_all['len_of_attr'] = df_all['attr'].map(lambda x: len(str(x).split())).astype(np.int64)
df_all['len_of_attr_title'] = df_all['attr_title'].map(lambda x: len(str(x).split())).astype(np.int64)

df_all['query_in_title'] = df_all.apply(lambda x: utils.counter_appearance(x['search_term'].split(), x['product_title'].split()), axis=1)
df_all['query_in_description'] = df_all.apply(lambda x: utils.counter_appearance(x['search_term'].split(), x['product_description'].split()), axis=1)
df_all['query_in_attr'] = df_all.apply(lambda x: utils.counter_appearance(str(x['search_term']).split(), str(x['attr']).split()), axis=1)
df_all['query_in_attr_title'] = df_all.apply(lambda x: utils.counter_appearance(str(x['search_term']).split(), str(x['attr_title']).split()), axis=1)
df_all['query_last_word_in_title'] = df_all.apply(lambda x: utils.counter_appearance(x['search_term'].split()[-1], x['product_title'].split()), axis=1)
df_all['query_last_word_in_description'] = df_all.apply(lambda x: utils.counter_appearance(x['search_term'].split()[-1], x['product_description'].split()), axis=1)
df_all['query_last_word_in_attr'] = df_all.apply(lambda x: utils.counter_appearance(str(x['search_term']).split()[-1], str(x['attr']).split()), axis=1)
df_all['query_last_word_in_attr_title'] = df_all.apply(lambda x: utils.counter_appearance(str(x['search_term']).split()[-1], str(x['attr_title']).split()), axis=1)
df_all['word_in_title'] = df_all.apply(lambda x: utils.counter_appear_times(x['search_term'].split(), x['product_title'].split()), axis=1)
df_all['word_in_description'] = df_all.apply(lambda x: utils.counter_appear_times(x['search_term'].split(), x['product_description'].split()), axis=1)
df_all['word_in_attr'] = df_all.apply(lambda x: utils.counter_appear_times(str(x['search_term']).split(), str(x['attr']).split()), axis=1)
df_all['word_in_attr_title'] = df_all.apply(lambda x: utils.counter_appear_times(str(x['search_term']).split(), str(x['attr_title']).split()), axis=1)
df_all['word_in_brand'] = df_all.apply(lambda x: utils.counter_appear_times(x['search_term'].split(), x['brand'].split()), axis=1)

desc_tf, desc_idf, desc_length, desc_ave_length = utils.tfidf(df_all, 'product_description')
df_all['desc_BM25_score'] = df_all.apply(lambda x: utils.BM25_score(x, desc_tf, desc_idf, desc_length, desc_ave_length), axis=1)

attr_tf, attr_idf, attr_length, attr_ave_length = utils.tfidf(df_all, 'attr')
df_all['attr_BM25_score'] = df_all.apply(lambda x: utils.BM25_score(x, attr_tf, attr_idf, attr_length, attr_ave_length), axis=1)

title_tf, title_idf, title_length, title_ave_length = utils.tfidf(df_all, 'product_title')
df_all['title_BM25_score'] = df_all.apply(lambda x: utils.BM25_score(x, title_tf, title_idf, title_length, title_ave_length), axis=1)

attr_title_tf, attr_title_idf, attr_title_length, attr_title_ave_length = utils.tfidf(df_all, 'attr_title')
df_all['attr_title_BM25_score'] = df_all.apply(lambda x: utils.BM25_score(x, attr_title_tf, attr_title_idf, attr_title_length, attr_title_ave_length), axis=1)

brand_tf, brand_idf, brand_length, brand_ave_length = utils.tfidf(df_all, 'brand')
df_all['brand_BM25_score'] = df_all.apply(lambda x: utils.BM25_score(x, brand_tf, brand_idf, brand_length, brand_ave_length), axis=1)


df_brand = pd.unique(df_all.brand.ravel())
d = {}
i = 1000
for s in df_brand:
    d[s] = i
    i += 3
df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(x))

df_all.to_csv(".../home-depot/data/df_all.csv",index = False)
