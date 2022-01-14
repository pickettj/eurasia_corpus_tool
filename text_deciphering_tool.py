#!/usr/bin/env python
# coding: utf-8

# # Text Deciphering Tool
# James Pickett

# In[120]:


import pickle, re, nltk, os


# In[121]:


import numpy as np
import pandas as pd

from pandas import DataFrame, Series


# In[122]:


# general function in Pandas to set maximum number of rows; otherwise only shows a few

pd.set_option('display.max_rows', 300)


# In[123]:


#set home directory path
hdir = os.path.expanduser('~')


# Sister files:
# - Pickled corpora cleaned in text_cleaning_tokenizing (optional: run `text_cleaning_tokenizing.py` - may take a few minutes to execute)
# - Corpora stats in corpora_statistics

# In[124]:


print ("run 'text_cleaning_tokenizing.py' to re-tokenize corpus")


# ### Help Function

# In[197]:


def tool_help ():
    print ("Eurasia Corpus Tool: Functions and Explanation\n\n           \ttool_help: lists functions\n           \tlist_corpora: lists all of the sub-corpora options\n           \tfreq: returns the most likely terms matching the search term\n           \tindex_kwic: key word in context; optional arguments: 'category' to specify a specific corpus                   'exclusion=False' to also include the Persian literature corpus\n           \tmulti_dic: searches through all of the dictionary corpuses for the term\n           \tdic_freq: searches through dictionary corpuses, returns most frequent term appearing in the text corpuses\n            \n\n\n"
          )
    


# In[126]:


#tool_help()


# ## Importing Corpora
# 
# 

# In[127]:


pickle_path = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Corpora/pickled_tokenized_cleaned_corpora"


# In[128]:


# import dataframe corpus

df_eurcorp = pd.read_csv (os.path.join(pickle_path,r'eurasia_corpus.csv'))
#df_eurcorp.sample(5)


# In[129]:


def list_corpora():
    categories = df_eurcorp['Category'].unique()
    print ("Corpora\n:", sorted(categories))


# ## Importing Datasets

# In[130]:


# dataset path

ds_path = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Datasets"


# In[131]:


# Von Melzer
meltzer = pd.read_csv(ds_path + "/von_melzer.csv")
#meltzer.sample(5)


# In[132]:


dehkhoda = pd.read_csv(ds_path + "/dehkhoda_dictionary.csv", names=['Term', 'Definition'])
dehkhoda = dehkhoda.rename(columns={'Term': 'Emic_Term'})
#dehkhoda.sample(5)


# In[133]:


# Locations
locations = pd.read_csv(ds_path + '/exported_database_data/locations.csv', names=['UID', 'Ar_Names',                                                 'Lat_Name', 'Nickname', 'Type'])
# Social Roles
roles = pd.read_csv(ds_path + '/exported_database_data/roles.csv', names=['UID', 'Term', 'Emic', 'Etic', 'Scope'])


# In[134]:


# Glossary
glossary = pd.read_csv(ds_path + '/exported_database_data/glossary.csv', names=['UID', 'Term',                                                 'Eng_Term', 'Translation', 'Transliteration', 'Scope', 'Tags'])
glossary = glossary.rename(columns={'Term': 'Emic_Term', 'Eng_Term':'Transliteration', 'Translation':'Definition'})
#glossary.sample(5)


# ## Dictionary Search Functions

# In[135]:


# simplify / standardize dictionaries

simp_meltzer = meltzer.rename(columns={'Präs.-Stamm': 'Emic_Term', 'Transkription': 'Transcription', 'Deutsch':'Definition'})                .drop(['Volume', 'Unnamed: 2', 'Persisch',  'Bemerkung'], axis=1)
simp_meltzer['Dataset']='meltzer'
#simp_meltzer.sample(5)


# In[136]:


simp_dehkhoda = dehkhoda
simp_dehkhoda['Dataset']='dehkhoda'


# In[137]:


simp_glossary = glossary
simp_glossary['Dataset']='glossary'
simp_glossary = simp_glossary.drop(['Transliteration'], axis=1)
#simp_glossary.sample(5)


# In[164]:


concat_dics = pd.concat([simp_dehkhoda, simp_glossary, simp_meltzer], axis=0)
#concat_dics.sample(5)


# In[174]:


def multi_dic (term):
    query_mask = concat_dics["Emic_Term"].str.contains(term, na=False)
    query = concat_dics[query_mask]
    result = query
    
    if len(result) > 50:
        result = result.sample(50)
    
    return (result)


# In[173]:


#multi_dic("دار")


# In[140]:


term = "چقر"
query_mask = simp_glossary["Emic_Term"].str.contains(term, na=False)
query1 = simp_glossary[query_mask]
query1


# In[141]:


multi_dic('چقر')


# In[142]:


# ?? how to return multiple dataframes without losing the dataframe pretty format
# ++ need to use concat function for this


# ## Custom KWIC

# In[143]:


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
    
    


# In[144]:


#index_kwic('اژدها', exclusion = False)


# In[145]:


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
        


# In[146]:


#kwic('اژ.ها$', exclusion=False)


# ## Frequency

# In[147]:


freq_dic = pd.value_counts(df_eurcorp.Token).to_frame().reset_index()


# In[176]:


freq_dic.sample(5)


# In[149]:


def freq (term):
    query_mask = freq_dic["index"].str.contains(term, na=False)
    query = freq_dic[query_mask]
    result = query.head()
    return (result)


# In[150]:


#freq("اژد.ا")


# In[189]:


# addiing the frequency data into the dictionary

# need to turn the transcribed word into a unique identifier (will lose some data):
concat_dics_clean = concat_dics.drop_duplicates(subset=['Emic_Term'], keep='first')

# merge with frequency data based on the unique identifier:
merged_dics_freq = pd.merge(left=concat_dics_clean, right=freq_dic, how='left', left_on='Emic_Term', right_on='index')

# clean up for readability
cleaned_merged_dics_freq = merged_dics_freq.drop(columns=['index']).rename(columns={'Token': 'Frequency'})

#cleaned_merged_dics_freq.sample(5)

# ?? how to get rid of those decimal points / why did they enter in the first place?


# ### Get Definition with Frequency

# In[190]:


def dic_freq (term):
    query_mask = cleaned_merged_dics_freq["Emic_Term"].str.contains(term, na=False)
    query = cleaned_merged_dics_freq[query_mask]
    result = query.sort_values('Frequency', ascending=False).head()
    return (result)
    


# In[196]:


#dic_freq('چ.ر$')


# ## Conditional Frequency

# In[152]:


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
    


# In[153]:


#confreq("اژدها", group = True)


# ---

# In[154]:


tool_help()


# In[155]:


list_corpora()


# ---

# In[156]:


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
