import pandas as pd
import numpy as np
import Feature_Extraction_Functions as f
 
attributes = pd.read_csv('input_clean/attributes_clean.csv',encoding="ISO-8859-1")
product_descriptions = pd.read_csv('input_clean/product_descriptions_clean.csv')
train_original = pd.read_csv('input_clean/train_clean.csv', encoding="ISO-8859-1",index_col='id')
test_original = pd.read_csv('input_clean/test_clean.csv', encoding="ISO-8859-1",index_col='id')

#for each product, I have put all its data in a dataframe...
product_dataframe = f.load_product_dataframe(train_original,test_original,attributes,product_descriptions)
    
#I have added all the product info into the train and test data for feature calculation efficiency
#try to load this 'train_all' and 'test_all' data, or if it has not yet been made, make it and save it.
train, test = f.load_train_and_test_data_with_product_info(train_original, test_original, product_dataframe, attributes)
train = train.fillna('')
test = test.fillna('')

#try to load search terms
search_terms = f.load_search_terms(train,test)
product_title_terms = f.load_terms(product_dataframe,'product_title')
prod_descr_terms = f.load_terms(product_dataframe,'prod_descr')
attr_names_terms = f.load_terms(product_dataframe,'attr_names')
attr_values_terms = f.load_terms(product_dataframe,'attr_values')
brand_name_terms = f.load_terms(product_dataframe,'brand_name')
    
#try to load the dictionary containing average document lengths, for BM25 calculations
doc_len_dict = f.load_dict_of_avg_doc_lengths(product_dataframe, train, test)

#represent search terms as a document, for use in reverse idf calculation
search_term_df = pd.concat([train['search_term'],test['search_term']])

#try to load the defaultdicts containing idf values for (query,doc) pairs
idf_dicts = {}
idf_dicts['search_term-product_title'] = f.load_idf_default_dict(search_terms,product_dataframe['product_title'],'search_term','product_title')
idf_dicts['search_term-prod_descr'] = f.load_idf_default_dict(search_terms,product_dataframe['prod_descr'],'search_term','prod_descr')
idf_dicts['search_term-attr_names'] = f.load_idf_default_dict(search_terms,product_dataframe['attr_names'],'search_term','attr_names')
idf_dicts['search_term-attr_values'] = f.load_idf_default_dict(search_terms,product_dataframe['attr_values'],'search_term','attr_values')
idf_dicts['product_title-search_term'] = f.load_idf_default_dict(product_title_terms,search_term_df,'product_title','search_term')
idf_dicts['prod_descr-search_term'] = f.load_idf_default_dict(prod_descr_terms,search_term_df,'prod_descr','search_term')
idf_dicts['attr_names-search_term'] = f.load_idf_default_dict(attr_names_terms,search_term_df,'attr_names','search_term')
idf_dicts['attr_values-search_term'] = f.load_idf_default_dict(attr_values_terms,search_term_df,'attr_values','search_term')
idf_dicts['brand_name-search_term'] = f.load_idf_default_dict(brand_name_terms,search_term_df,'attr_values','search_term')


#have a look at data for the first product uid
#attributes.loc[attributes['product_uid']==100001]
#product_descriptions.loc[product_descriptions['product_uid']==100001]
#train.loc[train['product_uid']==100001]

#%%

load_data = True

#try to load pre-processed data (to avoid recomputing features), if this fails create new dataframes
if load_data == True:
    print('Attempting to load pre-processed data...')
    try:
        #load pre-processed training data
        X_train = f.load_obj('input_clean/X_train')
        X_test = f.load_obj('input_clean/X_test')
        Y_train = f.load_obj('input_clean/Y_train')
        print('Pre-processed data successfully loaded...')
    except:
        print('Pre-processed data failed to load, creating new dataframe...')
        #create a data frame to store feature vectors
        X_train = pd.DataFrame(index = train.index)
        X_test = pd.DataFrame(index = test.index)
        #retrieve labels for training data
        Y_train = pd.DataFrame(data = {'relevance':train['relevance']}, index = train.index)
else:
    print('Creating dataframes to store new features...')
    #create a data frame to store feature vectors
    X_train = pd.DataFrame(index = train.index)
    X_test = pd.DataFrame(index = test.index)
    #retrieve labels for training data
    Y_train = pd.DataFrame(data = {'relevance':train['relevance']}, index = train.index)


#%%

#list of features to add to the data
#note: features in the list are only added if they are not already in the dataframe
features = [
            #tf features should contain: doc, query & tf_type
            #tf_type is the type of tf to calculate
            ('_tf_','product_title','search_term','natural'),
            ('_tf_','product_title','search_term','natural_avg'),
            ('_tf_','product_title','search_term','log_norm_avg'),
            ('_tf_','product_title','search_term','double_norm_0.5_avg'),
            ('_tf_','prod_descr','search_term','natural'),
            ('_tf_','prod_descr','search_term','natural_avg'),
            ('_tf_','prod_descr','search_term','log_norm_avg'),
            ('_tf_','prod_descr','search_term','double_norm_0.5_avg'),
            ('_tf_','attr_names','search_term','natural'),
            ('_tf_','attr_names','search_term','natural_avg'),
            ('_tf_','attr_names','search_term','log_norm_avg'),
            ('_tf_','attr_names','search_term','double_norm_0.5_avg'),
            ('_tf_','attr_values','search_term','natural'),
            ('_tf_','attr_values','search_term','natural_avg'),
            ('_tf_','attr_values','search_term','log_norm_avg'),
            ('_tf_','attr_values','search_term','double_norm_0.5_avg'),

            #length features should contain: doc
            ('_length','product_title'),
            ('_length','prod_descr'),
            ('_length','attr_names'),
            ('_length','attr_values'),
            ('_length','search_term'),


            #tf_idf features should contain: doc, query, tf_type, idf_type, tf_idf_type
            #tf_idf is the type of tf_idf to calculate (sum/average across query words)
            ('_tf_idf_','product_title','search_term','natural','smooth','sum'),
            ('_tf_idf_','product_title','search_term','natural','max','sum'),
            ('_tf_idf_','product_title','search_term','natural','prob','sum'),
            ('_tf_idf_','prod_descr','search_term','natural','smooth','sum'),
            ('_tf_idf_','prod_descr','search_term','natural','max','sum'),
            ('_tf_idf_','prod_descr','search_term','natural','prob','sum'),
            ('_tf_idf_','attr_names','search_term','natural','smooth','sum'),
            ('_tf_idf_','attr_names','search_term','natural','max','sum'),
            ('_tf_idf_','attr_names','search_term','natural','prob','sum'),
            ('_tf_idf_','attr_values','search_term','natural','smooth','sum'),
            ('_tf_idf_','attr_values','search_term','natural','max','sum'),
            ('_tf_idf_','attr_values','search_term','natural','prob','sum'),

            ('_tf_idf_','product_title','search_term','log_norm','smooth','sum'),
            ('_tf_idf_','product_title','search_term','log_norm','max','sum'),
            ('_tf_idf_','product_title','search_term','log_norm','prob','sum'),
            ('_tf_idf_','prod_descr','search_term','log_norm','smooth','sum'),
            ('_tf_idf_','prod_descr','search_term','log_norm','max','sum'),
            ('_tf_idf_','prod_descr','search_term','log_norm','prob','sum'),
            ('_tf_idf_','attr_names','search_term','log_norm','smooth','sum'),
            ('_tf_idf_','attr_names','search_term','log_norm','max','sum'),
            ('_tf_idf_','attr_names','search_term','log_norm','prob','sum'),
            ('_tf_idf_','attr_values','search_term','log_norm','smooth','sum'),
            ('_tf_idf_','attr_values','search_term','log_norm','max','sum'),
            ('_tf_idf_','attr_values','search_term','log_norm','prob','sum'),

            #tf-idf features, treating 'search_term' as the document
            ('_tf_idf_','search_term','product_title','natural','smooth','sum'),
            ('_tf_idf_','search_term','prod_descr','natural','smooth','sum'),
            ('_tf_idf_','search_term','attr_names','natural','smooth','sum'),
            ('_tf_idf_','search_term','attr_values','natural','smooth','sum'),


            #bm 25 features should contain: doc, query k1, b
            ('_bm25_','product_title','search_term', 'sum', 1.5, 0.75),
            ('_bm25_','prod_descr','search_term', 'sum', 1.5, 0.75),
            ('_bm25_','attr_names','search_term', 'sum', 1.5, 0.75),
            ('_bm25_','attr_values','search_term', 'sum', 1.5, 0.75),
            ('_bm25_','product_title','search_term', 'avg', 1.5, 0.75),
            ('_bm25_','prod_descr','search_term', 'avg', 1.5, 0.75),
            ('_bm25_','attr_names','search_term', 'avg', 1.5, 0.75),
            ('_bm25_','attr_values','search_term', 'avg', 1.5, 0.75),
            #bm25 features where we treat 'search_term' as the document
            ('_bm25_','search_term', 'product_title','sum', 1.5, 0.75),
            ('_bm25_','search_term', 'prod_descr','sum', 1.5, 0.75),
            ('_bm25_','search_term', 'attr_names','sum', 1.5, 0.75),
            ('_bm25_','search_term', 'attr_values','sum', 1.5, 0.75),
            ('_bm25_','search_term', 'product_title','avg', 1.5, 0.75),
            ('_bm25_','search_term', 'prod_descr','avg', 1.5, 0.75),
            ('_bm25_','search_term', 'attr_names','avg', 1.5, 0.75),
            ('_bm25_','search_term', 'attr_values','avg', 1.5, 0.75),

            #adding brand name tf
            ('_tf_','search_term','brand_name','natural'),
            ('_tf_idf_','search_term','brand_name','natural','smooth','sum'),
            ('_bm25_','search_term','brand_name', 'sum', 1.5, 0.75),
            ]

f.save_obj(Y_train,'input_clean/Y_train')
#add each feature to the data
for feature in features:
    X_train = f.add_feature(X_train, train, product_dataframe, feature, 0.5, idf_dicts, 124428, doc_len_dict, data = 'train')
    X_test = f.add_feature(X_test, test, product_dataframe, feature, 0.5, idf_dicts,124428, doc_len_dict, data = 'test')
    

#save the pre-processed data
print('Saving data...')
f.save_obj(X_train,'input_clean/X_train')
f.save_obj(X_test,'input_clean/X_test')

#%%

#split the data by relevance value and view the difference in means for each feature
#retrieve feature importance scores, calculated by 
scores = []
for col in X_train.columns:
    less_rel_score = X_train.loc[Y_train['relevance']<2.0][col].describe()['mean']
    more_rel_score = X_train.loc[Y_train['relevance']>=2.0][col].describe()['mean']
    print('Feature: {} \n Mean for relevance > 2.5: {} \n Mean for relevance < 1.5: {}'.format(col,
          more_rel_score,
          less_rel_score))
    scores.append(abs(more_rel_score - less_rel_score)/min(less_rel_score,more_rel_score))
    

import matplotlib.pyplot as plt
import seaborn as sns
#number the attributes from 1 to 68, store these as k,v pairs
feature_importance_scores = [(i+1,scores[i]) for i in range(len(scores))]
#sort the attributes by importance, descending
sorted_feats = sorted(feature_importance_scores, key = lambda x:x[1], reverse=True)
#retrieves sorted keys and values
sorted_keys = [feat[0] for feat in sorted_feats]
sorted_vals = [feat[1] for feat in sorted_feats]
sorted_names = [X_train.columns[i-1] for i in sorted_keys]


fig, (ax1, ax2) = plt.subplots(1,2,sharey=True)
#plot 10 most important features
sns.barplot(x = sorted_keys[:10],y = sorted_vals[:10], order = sorted_keys[:10], ax=ax1)
#plot 10 least important features
sns.barplot(x = sorted_keys[-10:],y = sorted_vals[-10:], order = sorted_keys[-10:], ax=ax2)
ax1.set_title('10 Most Important Features')
ax2.set_title('10 Least Important Features')
fig.savefig('output/FeatureImportance_UsingRelativeDifference')

feature_importance = pd.DataFrame({'Feature':sorted_names,'% Importance':sorted_vals})
feature_importance.to_csv('output/FeatureImportance_UsingRelativeDifference.csv')
        
