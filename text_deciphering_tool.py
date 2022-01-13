#!/usr/bin/env python
# coding: utf-8

# # Text Deciphering Tool
# James Pickett

# In[139]:


import pickle, re, nltk, os


# In[140]:


import numpy as np
import pandas as pd

from pandas import DataFrame, Series


# In[141]:


# general function in Pandas to set maximum number of rows; otherwise only shows a few

pd.set_option('display.max_rows', 300)


# In[142]:


#set home directory path
hdir = os.path.expanduser('~')


# Sister files:
# - Pickled corpora cleaned in text_cleaning_tokenizing (optional: run `text_cleaning_tokenizing.py` - may take a few minutes to execute)
# - Corpora stats in corpora_statistics

# In[143]:


print ("run 'text_cleaning_tokenizing.py' to re-tokenize corpus")


# ### Help Function

# In[241]:


def tool_help ():
    print ("Eurasia Corpus Tool: Functions and Explanation\n\n           \tlist_corpora: lists all of the sub-corpora options\n           \tfreq: returns the most likely terms matching the search term\n           \tindex_kwic: key word in context; optional arguments: 'category' to specify a specific corpus                   'exclusion=False' to also include the Persian literature corpus\n            \n\n\n"
          )
    


# In[223]:


#tool_help()


# ## Importing Corpora
# 
# 

# In[144]:


pickle_path = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Corpora/pickled_tokenized_cleaned_corpora"


# In[230]:


# import dataframe corpus

df_eurcorp = pd.read_csv (os.path.join(pickle_path,r'eurasia_corpus.csv'))
#df_eurcorp.sample(5)


# In[219]:


def list_corpora():
    categories = df_eurcorp['Category'].unique()
    print ("Corpora\n:", sorted(categories))


# ## Importing Datasets

# In[159]:


# dataset path

ds_path = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Datasets"


# In[161]:


# Von Melzer
meltzer = pd.read_csv(ds_path + "/von_melzer.csv")
#meltzer.sample(5)


# In[173]:


dehkhoda = pd.read_csv(ds_path + "/dehkhoda_dictionary.csv", names=['Term', 'Definition'])
#dehkhoda.sample(5)


# In[162]:


# Locations
locations = pd.read_csv(ds_path + '/exported_database_data/locations.csv', names=['UID', 'Ar_Names',                                                 'Lat_Name', 'Nickname', 'Type'])
# Social Roles
roles = pd.read_csv(ds_path + '/exported_database_data/roles.csv', names=['UID', 'Term', 'Emic', 'Etic', 'Scope'])

# Glossary
glossary = pd.read_csv(ds_path + '/exported_database_data/glossary.csv', names=['UID', 'Term',                                                 'Eng_Term', 'Translation', 'Transliteration', 'Scope', 'Tags'])


# ## Dictionary Search Functions

# In[216]:


def multi_dic (term):
    query_mask = dehkhoda["Term"].str.contains(term, na=False)
    query = dehkhoda[query_mask]
    result = query.sample(5)
    return (result)


# In[231]:


#multi_dic('د.رو')


# In[232]:


# ?? how to return multiple dataframes without losing the dataframe pretty format


# ## Custom KWIC

# In[187]:


def index_kwic (term, category=None, exclusion=True):
    
    """This function returns a dataframe filtered by the search term."""
    df_eurcorp_lite = df_eurcorp

    # default state is to exclude the vast Persian literature corpus
    if exclusion:
        df_eurcorp_lite = df_eurcorp_lite[df_eurcorp_lite['Category'] != 'pers_lit_toks']
    
    # default state is to include all of the sub-corpora; but can also specify one of them
    if category is not None:
        df_eurcorp_lite = df_eurcorp_lite[df_eurcorp_lite['Category'] == category]

        
    result = df_eurcorp_lite[df_eurcorp_lite['Token'].str.match(term)]
    return result
    
    # str.match; the str part is telling match how to behave; .match is a method specific to pandas
    
    


# In[199]:


#index_kwic('اژدها', exclusion = False)


# In[218]:


def kwic (term, category=None, exclusion=True):
    # optional arguments refer to index_kwic function
    
    for i, item in index_kwic(term, category, exclusion).iterrows():
        
        title = item["Text"]
        category = item["Category"]
        loc = item['No']
        
        filtered = df_eurcorp[(df_eurcorp['Text']==title)]
        filt_toks = filtered[(filtered['No']>=(loc-5))&(filtered['No']<=(loc+5))]

        
        #filtered = filtered.sort_values("index")
        # probably already sorted, but better to be on the safe side
                
        text = " ".join(filt_toks["Token"])
        
        print(f'{title}: {loc} {category}\n{text}\n')
        
        # task: figure out how to color code results; termcolor package, has to be installed



        
# iterrows(): research what this does exactly, has something to do with dataframes being composed of series
        


# In[ ]:


#kwic('اژ.ها$', exclusion=False)


# ## Frequency

# In[111]:


freq_dic = pd.value_counts(df_eurcorp.Token).to_frame().reset_index()


# In[238]:


#freq_dic.sample(5)


# In[236]:


def freq (term):
    query_mask = freq_dic["index"].str.contains(term, na=False)
    query = freq_dic[query_mask]
    result = query.head()
    return (result)


# In[240]:


#freq("اژد.ا")


# In[239]:


# ++ can use methodology from Pahlavi tool to merge dictionary and frequency data
## but first need to solve the problem of how to prettily return multiple dataframes (per above)


# ## Conditional Frequency

# In[131]:


def confreq (term, group=False):
    sel = df_eurcorp[df_eurcorp['Token']==term].copy()
    sel['index_next'] = sel['No'] + 1
    sel = sel.join(
        df_eurcorp.set_index(['Text', 'Category', 'No'])['Token'].rename('token_next'),
        on=['Text', 'Category', 'index_next']
    )
    # If there are only 1-frequency results, it will still show them;
    # but if there are enough higher frequency results, it will omit the 1-frequency results.
    result = sel['token_next'].value_counts()
    short_result = [(x,y) for x,y in result.items() if y > 1]
    if len(short_result) > 5:
        result = short_result
    # improvement: create a list of omitted words (e.g. ud, ī, etc.), and make a flag1=False
    # optional argument to omit them.
    
    if group == True:
        result = sel.groupby('Category')['token_next'].value_counts().rename("count").reset_index()
    
    return (result)
    


# In[225]:


#confreq("اژدها", group = True)


# ---

# In[226]:


tool_help()


# In[227]:


list_corpora()


# ---

# In[242]:


# next steps:

## conditional trigram frequency
## reverse conditional frequency


# ---

# def multi_dic (term):
#     
#     match_freq(term)
#     
#     search_term = re.compile(term)
#     
#     glos_query_mask = glossary["Term"].str.contains(search_term, na=False)
#     glos_query = glossary[glos_query_mask][["UID", "Term", "Translation"]]
#     glos_query
#     
#     
#     dehkhoda_query_mask = dehkhoda["Term"].str.contains(search_term, na=False)
#     dehkhoda_query = dehkhoda[dehkhoda_query_mask]
#     dehkhoda_query    
#     
#     melz_query_mask = meltzer["Präs.-Stamm"].str.contains(search_term, na=False)
#     melz_query = meltzer[melz_query_mask]
#     melz_query[["Präs.-Stamm", "Deutsch"]]
#     
#     
#     result = print ("Glossary \n\n", glos_query,"\n\n\n", \
#                     "Dehkhoda \n\n", dehkhoda_query,"\n\n\n",\
#                     "Von_Meltzer \n\n", melz_query[["Präs.-Stamm", "Deutsch"]])
# 
#     return result
# 

# [Another way of doing max value](https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary):
# 
# ```python
# def keywithmaxval(d):
#      """ a) create a list of the dict's keys and values; 
#          b) return the key with the max value"""  
#      v=list(d.values())
#      k=list(d.keys())
#      return k[v.index(max(v))]
# ```

# def best_match (term, corpus):
#     
#     """Takes a search term and frequency dictionary, returns the most frequently\
#     appearing match within the specified corpus as [matching term, frequency of appearnace]."""
#     
#     search_term = re.compile(term)
#     toks = {k:v for (k,v) in corpus.items() if re.match(search_term, k)}
#     if len(toks) > 0:
#         match = sorted(toks, key=toks.get, reverse=True)[0]
#         freq = corpus[match]
#         pair = [match, freq]
#     
#     else:
#         pair = None
#     
#     return pair

# def match_freq(term):
#     
#    
#     
#     if best_match(term, combo_freq) is not None:
#         print ("Most likely match in corpus:\n\n\t",\
#               best_match(term, combo_freq)[0], "appearing ", best_match(term, combo_freq)[1], "times;\n")
#     
#     search_term = re.compile(term)
#     toks = {k:v for (k,v) in combo_freq.items() if re.match(search_term, k)}
#     if len(toks) > 3:
#         cf2 = sorted(toks, key=toks.get, reverse=True)[1]
#         cf3 = sorted(toks, key=toks.get, reverse=True)[2]
#         print ("\tfollowed by:\n\t",\
#                     cf2, "appearing ", combo_freq[cf2], "times, and \n\t",\
#                    list(sorted(toks))[2], "appearing ", combo_freq[list(sorted(toks))[2]], "times\n\n")
#     
#     
#     print ("Most likely matches in sub-corpora:\n")
#     
#     if best_match(term, doc_freq) is not None:
#            print("\tDocuments:", best_match(term, doc_freq)[0], "appearing ", best_match(term, doc_freq)[1], "times;\n")
#     
#     if best_match(term, nar_freq) is not None:
#            print ("\tNarrative texts:", best_match(term, nar_freq)[0], "apprearing", best_match(term, nar_freq)[1], "times;\n\n")
# 
#     if best_match(term, indo_freq) is not None:
#            print ("\tIndic texts:", best_match(term, indo_freq)[0], "appearing ", best_match(term, indo_freq)[1], "times;\n")
#     
#     if best_match(term, trans_freq) is not None:
#            print ("\tTransoxania texts:", best_match(term, trans_freq)[0], "appearing ", best_match(term, trans_freq)[1], "times;\n")
#     
# 
#     print ("\nMost likely matches in Persian literature corpus:\n\t")
#            
#     if best_match(term, pers_lit_freq) is not None:
#            print ("\t",best_match(term, pers_lit_freq)[0], "appearing ", best_match(term, pers_lit_freq)[1], "times;\n")
#     
#     
# 
#     
# 

# ## Simple Conditional Frequency Tool

# def confreq (term, refine="none"):
#     
#     """Conditional frequency across multiple corpora and sub-corpora.
#         Takes a search term (no regex) for the first word in the bigram, 
#         as well as an optional regex filter to narrow down the second word
#         in the bigram sequence.
#             
#     """
#     
#     if refine == "none" and len(combo_cfd[term]) > 0:
#     
#         if len(combo_cfd[term]) > 0:
#             print (term, " is most commonly followed by:\n\n", combo_cfd[term].most_common(10))
# 
#         
# 
#         if len(combo_cfd[term]) > 0:
#             print("\nWithin sub-corpora:\n")
# 
#             # Still need to fill out sub-corpora
# 
#             if len(doc_cfd[term]) > 0 :
#                 print ("\tDocuments:", term, " is most commonly followed by:\n\n\t", doc_cfd[term].most_common(5))
#             if len(nar_cfd[term]) > 0 :
#                 print ("\n\tNarrative texts:", term, " is most commonly followed by:\n\n\t", nar_cfd[term].most_common(5))
#             if len(indo_cfd[term]) > 0 :
#                 print ("\n\n\tIndic texts:", term, " is most commonly followed by:\n\n\t", indo_cfd[term].most_common(5))
#             if len(trans_cfd[term]) > 0 :
#                 print ("\n\tTransoxania texts:", term, " is most commonly followed by:\n\n\t", trans_cfd[term].most_common(5))
#     
#     # Optional regex refinement of the results:
#     if refine != "none" and len(combo_cfd[term]) > 0:
#         
#         filt = re.compile(refine)
#         
#         filt_toks = [(x, y) for (x, y) in combo_cfd[term].items() if re.match(refine, x)]
#         
#         print ("With the results filtered by the regex search (", refine, "), the most likely words following,", term, "are:\n\t", filt_toks)
#         
#     elif len(combo_cfd[term]) == 0:
#             print ("no results")
#         
# 
#    

# def regcf (term):
#     
#     """
#         From a regex search term finds the most frequent possible word, then returns 
#         conditional frequency (from bigrams) across multiple corpora.
#     
#     """
#     
#     if best_match(term, combo_freq) is not None:
#         
#         local_match = best_match(term, combo_freq)[0]
#         
#         print ("The most likly match (based on word frequency) for ", term, " is ", local_match,\
#               "(with frequency", best_match(term, combo_freq)[1], ").\n")
#         print ("Conditional frequency of", local_match, "(combined corpus):\n\t", combo_cfd[local_match].most_common(5))
#         
#         print ("\nSub-Corpora:\n")
#         
#         print ("\n\tDocuments:\n\t", doc_cfd[local_match].most_common(5))
#         print ("\n\tNarrative texts:\n\t", nar_cfd[local_match].most_common(5))
#         
#         print ("\n\n\tIndic texts:\n\t", indo_cfd[local_match].most_common(5))
#         print ("\n\tTransoxania texts:\n\t", trans_cfd[local_match].most_common(5))
#         
#        
#     

# ### Third term, if first two known:

# *Document Corpus (Meta-Corpus simply too computationally costly)*

# def tricfd (first_term, second_term, refine="none"):
#     
#     """
#     Given two words in a row, conditional frequency of the third word in the sequence.
#     Inputs: two words (in order), and an optional regex filter on the third word.
#             
#     """
#     
#     #print ("The pair ", first_term, second_term, " is most commonly followed by :\n")
#     #output = combo_tricfd[(first_term, second_term)].most_common(10)
#     #print (output)
#     
#     
#     
#     if refine == "none" and len(combo_tricfd[(first_term, second_term)]) > 0:
#     
#         if len(combo_tricfd[(first_term, second_term)]) > 0:
#             print ("The pair ", first_term, second_term, " is most commonly followed by:\n\n", combo_tricfd[(first_term, second_term)].most_common(10))
# 
# 
#         if len(combo_tricfd[(first_term, second_term)]) > 0:
#             print("\nWithin sub-corpora:\n")
# 
#             # Still need to fill out sub-corpora
# 
#             if len(doc_tricfd[(first_term, second_term)]) > 0 :
#                 print ("\tDocuments:\n\n\t", doc_tricfd[(first_term, second_term)].most_common(5))
#             if len(nar_tricfd[(first_term, second_term)]) > 0 :
#                 print ("\n\tNarrative texts:\n\n\t", nar_tricfd[(first_term, second_term)].most_common(5))
#             if len(indo_tricfd[(first_term, second_term)]) > 0 :
#                 print ("\n\n\tIndic texts:\n\n\t", indo_tricfd[(first_term, second_term)].most_common(5))
#             if len(trans_tricfd[(first_term, second_term)]) > 0 :
#                 print ("\n\tTransoxania texts::\n\n\t", trans_tricfd[(first_term, second_term)].most_common(5))
#     
#     # Optional regex refinement of the results:
#     if refine != "none" and len(combo_tricfd[(first_term, second_term)]) > 0:
#         
#         filt = re.compile(refine)
#         
#         filt_toks = [(x, y) for (x, y) in combo_tricfd[(first_term, second_term)].items() if re.match(refine, x)]
#         
#         print ("With the results filtered by the regex search (", refine, "), the most likely words following the pair ", first_term, second_term, "are:\n\t", filt_toks)
#         
#     elif len(combo_tricfd[(first_term, second_term)]) == 0:
#             print ("no results")

# ### Reversed conditional frequency, i.e. if second word in sequence known but not first

# *Meta-Corpus*

# def revcfd (term, refine="none"):
#     
#     
#     """Reverse conditional frequency (bigrams) across multiple corpora and sub-corpora.
#         Takes a search term (no regex) for the first word in the bigram, 
#         as well as an optional regex filter to narrow down the second word
#         in the bigram sequence.
#             
#     """
#     
#     if refine == "none" and len(rev_combo_cfd[term]) > 0:
#     
#         if len(rev_combo_cfd[term]) > 0:
#             print (term, " is most commonly preceded by:\n\n", rev_combo_cfd[term].most_common(10))
# 
#         
# 
#         if len(rev_combo_cfd[term]) > 0:
#             print("\nWithin sub-corpora:\n")
# 
#             # Still need to fill out sub-corpora
# 
#             if len(rev_doc_cfd[term]) > 0 :
#                 print ("\tDocuments:", term, " is most commonly preceded by:\n\n\t", rev_doc_cfd[term].most_common(5))
#             if len(rev_nar_cfd[term]) > 0 :
#                 print ("\n\tNarrative texts:", term, " is most commonly preceded by:\n\n\t", rev_nar_cfd[term].most_common(5))
#             if len(rev_indo_cfd[term]) > 0 :
#                 print ("\n\n\tIndic texts:", term, " is most commonly preceded by:\n\n\t", rev_indo_cfd[term].most_common(5))
#             if len(rev_trans_cfd[term]) > 0 :
#                 print ("\n\tTransoxania texts:", term, " is most commonly preceded by:\n\n\t", rev_trans_cfd[term].most_common(5))
#     
#     # Optional regex refinement of the results:
#     if refine != "none" and len(rev_combo_cfd[term]) > 0:
#         
#         filt = re.compile(refine)
#         
#         filt_toks = [(x, y) for (x, y) in rev_combo_cfd[term].items() if re.match(refine, x)]
#         
#         print ("With the results filtered by the regex search (", refine, "), the most likely words preceding,", term, "are:\n\t", filt_toks)
#         
#     elif len(rev_combo_cfd[term]) == 0:
#             print ("no results")
#         
# 
#    

# ----
# ----
# ----
# # Graveyard
# (i.e. code saved for posterity, no longer active)

# ### Keyword in Context

# ### NLTK Concordance

# ```python
# # for whatever reason you can't just use the concordance method on a string;
# # you have to convert it to an NLTK Text type one way or another
# 
# trans_corpus = nltk.Text(raw_combo_toks)
# 
# #trans_corpus.concordance('خانه')
# ```

# ### Regex Concordance

# *Tokens in corpus regex matching the string:*
# 
# (obsolete with custom KWIC)

# ```python
# toks = [x for x in combo_freq if re.match(r'...خوی', x)]
# toks[:5]
# ```

# ```python
# conc0 = sum([trans_corpus.concordance_list(x) for x in toks], [])
# conc1 = [c.line for c in conc0]
# print('\n'.join(conc1))
# ```

# ### Custom KWIC (beta)

# (drafting, active version now as a function, saved in markdown for posterity)

# ```python
# # Creating 5-Grams
# 
# five_grams = {k:list(nltk.ngrams(v, 5)) for (k,v) in combined_corpus_toks.items() if len(v) >= 5}
# ```

# ```python
# # Find in Document
# ## This function takes a dictionary of 5-grams as the first argument,
# ## a regex search term as the second argument, and returns the sequence of 5 words
# 
# def find_doc(d, s):
#     for v in d:
#         m = re.match(s, v[2])
#         if m is not None:
#             yield ' '.join(v)
#             
# 
# # Note: Return sends a specified value back to its caller
# # whereas Yield can produce a sequence of values.
# 
# 
# # Example:
# ## list(find_doc(five_grams['al_biruni_card_catalog_suleimanov_fond'], 'ف.'))
# ```

# ```
# # Find Corpus
# ## Produces a generator object with the KWIC with associated work title
# 
# def find_corpus(c, s):
#     for k, d in five_grams.items():
#         for m in find_doc(d, s):
#             yield f'{k:50s}: {m}'
# ```

# ```python
# # Formatting
# 
# def print_align(v, m):
#     plen = max([sum([len(z)+1 for z in x[:m]]) for x in v])
#     for x in v:
#         pre = ' '.join(x[:m])
#         mid = x[m]
#         pos = ' '.join(x[m+1:])
#         print(f'{pre:>{plen}s} \033[1m{mid}\033[0m {pos}')
# ```

# ```python
# print('\n'.join(find_corpus(five_grams, '^من.قر?$')))
# ```

# ___
# ___

# # Conditional Frequency

# *Meta-Corpus*

# ```python
# # ConditionalFreqDist() takes a list of pairs.
# # Generator variable uses itself up upon assignment, so need to recreate above
# 
# bigrams_cfd = nltk.ngrams(raw_combo_toks, 2)
# 
# cfd = nltk.ConditionalFreqDist(bigrams_cfd)
# ```

# ### Simple Conditional Frequency:

# *Meta-Corpus*

# ```python
# search_term = r"جهد"
# ```

# ```python
# print (search_term, " is most commonly followed by:\n")
# cfd[search_term].most_common(5)
# ```

# *Document Corpus*

# ```python
# bigrams_doc_fd = nltk.ngrams(raw_doc_toks, 2)
# 
# cfd_doc = nltk.ConditionalFreqDist(bigrams_doc_fd)
# ```

# ```python
# search_term = "بداند"
# ```

# ```python
# print ("\nin the documents corpus, ", search_term, " is most commonly followed by: \n")
# cfd_doc[search_term].most_common(5)
# ```
