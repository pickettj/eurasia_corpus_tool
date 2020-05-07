#!/usr/bin/env python
# coding: utf-8

# # Text Deciphering Tool

# In[73]:


import pickle, re, nltk, os


# In[74]:


import numpy as np
import pandas as pd

from pandas import DataFrame, Series


# In[75]:


#set home directory path
hdir = os.path.expanduser('~')


# Sister files:
# - Pickled corpora cleaned in text_cleaning_tokenizing
# - Corpora stats in corpora_statistics

# ## I. Importing Corpora
# 
# 

# In[76]:


pickle_path = hdir + "/Box/Notes/Digital_Humanities/Corpora/pickled_tokenized_cleaned_corpora"


# In[77]:


with open(pickle_path + "/corpora.pkl", "rb") as f:
    unsorted_doc_toks,                indo_xml_toks, hyd_xml_toks, trans_xml_toks,                trans_nar_toks, indo_nar_toks,                trans_nar_ext_toks, indo_nar_ext_toks = pickle.load(f)


# In[78]:


with open(pickle_path + "/meta_corpora.pkl", "rb") as f:
    comb_india_nar_toks, comb_trans_nar_toks, nar_corpus_toks, doc_corpus_toks,                combined_corpus_toks, mega_corpus_toks = pickle.load(f)


# In[79]:



#"خان" in combined_corpus_toks["tarikh_i_baljuvan_al_biruni_2663iii_ser412"]
        


# ## II. Importing Raw Tokens
# I.e. tokens without parent text designation, i.e. format necessary for many NLTK routines.

# In[80]:


with open(pickle_path + "/raw_tokens.pkl", "rb") as f:
    raw_doc_toks, raw_nar_toks, raw_lit_toks, raw_combo_toks = pickle.load(f)


# In[81]:


#indo_nar_toks.keys()


# ## III. Importing Datasets

# - Von Melzer Persian Lexicon
# - Glossary
# - Place Names

# In[82]:


# dataset path

ds_path = hdir + "/Box/Notes/Digital_Humanities/Datasets"


# In[83]:


# Von Melzer
meltzer = pd.read_csv(ds_path + "/von_melzer.csv")


# In[84]:


#meltzer["Präs.-Stamm"].sample(5)
#meltzer.sample(10)


# In[85]:


# Locations
locations = pd.read_csv(ds_path + '/exported_database_data/locations.csv', names=['UID', 'Ar_Names',                                                 'Lat_Name', 'Nickname', 'Type'])
# Social Roles
roles = pd.read_csv(ds_path + '/exported_database_data/roles.csv', names=['UID', 'Term', 'Emic', 'Etic', 'Scope'])

# Glossary
glossary = pd.read_csv(ds_path + '/exported_database_data/glossary.csv', names=['UID', 'Term',                                                 'Eng_Term', 'Translation', 'Transliteration', 'Scope', 'Tags'])


# ___
# ___

# In[86]:


dehkhoda = pd.read_csv(ds_path + "/dehkhoda_dictionary.csv", names=['Term', 'Definition'])


# In[87]:


#dehkhoda.sample(10)


# # Basic Search

# Regex reminders:
# - Just the word itself: `^مال$`

# In[88]:


search_term = re.compile(r"ب.د")


# ### Von Melzer Persian Dictionary

# In[89]:


melz_query_mask = meltzer["Präs.-Stamm"].str.contains(search_term, na=False)
melz_query = meltzer[melz_query_mask]
melz_query


# ### Database Terms

# #### (a) Technical Lexicon

# In[90]:


glos_query_mask = glossary["Term"].str.contains(search_term, na=False)
glos_query = glossary[glos_query_mask]
glos_query


# #### (b) Social Roles

# In[91]:


roles_query_mask = roles["Emic"].str.contains(search_term, na=False)
roles_query = roles[roles_query_mask]
roles_query


# #### (c) Place Names

# In[92]:


loc_query_mask = locations["Ar_Names"].str.contains(search_term, na=False)
loc_query = locations[loc_query_mask]
loc_query


# ### Corpus Tokens

# In[93]:


search_term = re.compile(r"کوبکار.?")


# In[94]:


combo_freq = nltk.FreqDist(raw_doc_toks)
toks = [x for x in combo_freq if re.match(search_term, x)]
toks[:5]


# ### Keyword in Context

# ### NLTK Concordance

# In[95]:



# for whatever reason you can't just use the concordance method on a string;
# you have to convert it to an NLTK Text type one way or another

trans_corpus = nltk.Text(raw_combo_toks)

#trans_corpus.concordance('خانه')


# ### Regex Concordance

# *Tokens in corpus regex matching the string:*

# In[97]:


toks = [x for x in combo_freq if re.match(r'...خوی', x)]
toks[:5]


# In[98]:


conc0 = sum([trans_corpus.concordance_list(x) for x in toks], [])
conc1 = [c.line for c in conc0]
print('\n'.join(conc1))


# ### Custom KWIC (beta)

# In[ ]:


# Better KWIC: need to (a) list source,
# and (b) have the ability to have multiple tokens in a row.


# In[29]:


combined_corpus_toks["al_biruni_card_catalog_suleimanov_fond"][5]


# In[43]:


#list(nltk.ngrams(raw_doc_toks, 5))


# In[46]:


five_grams = {k:list(nltk.ngrams(v, 5)) for (k,v) in combined_corpus_toks.items()}


# In[ ]:





# In[36]:


five_grams = {k:list(v) for (k,v) in five_grams.items()}


# In[57]:


five_grams = list(five_grams)
five_grams[5][2] == "پانصد"


# In[30]:


five_grams[5][3]


# In[58]:


search_toks = [x for x in five_grams if x[2] == "پانصد"]
search_toks[:5]


# ___
# ___

# # Conditional Frequency

# *Meta-Corpus*

# In[50]:


# ConditionalFreqDist() takes a list of pairs.
# Generator variable uses itself up upon assignment, so need to recreate above

bigrams_cfd = nltk.ngrams(raw_combo_toks, 2)

cfd = nltk.ConditionalFreqDist(bigrams_cfd)


# ### Simple Conditional Frequency:

# *Meta-Corpus*

# In[51]:


search_term = r"جهد"


# In[52]:


print (search_term, " is most commonly followed by:\n")
cfd[search_term].most_common(5)


# *Document Corpus*

# In[105]:


bigrams_doc_fd = nltk.ngrams(raw_doc_toks, 2)

cfd_doc = nltk.ConditionalFreqDist(bigrams_doc_fd)


# In[162]:


search_term = "بداند"


# In[163]:


print ("\nin the documents corpus, ", search_term, " is most commonly followed by: \n")
cfd_doc[search_term].most_common(5)


# ### Third term, if first two known:

# *Document Corpus (Meta-Corpus simply too computationally costly)*

# In[195]:


tri0 = nltk.ngrams(raw_doc_toks, 3)
tri1 = [((a, b), c) for (a, b, c) in tri0]
cfd1 = nltk.ConditionalFreqDist(tri1)


# In[196]:


first_term = "بکار"
second_term = "برد"


# In[197]:


print ("The pair ", first_term, second_term, " is most commonly followed by :\n")

cfd1[(first_term, second_term)]


# ### Reversed conditional frequency, i.e. if second word in sequence known but not first

# *Meta-Corpus*

# In[61]:


search_term = "دلربا"


# In[62]:


bi0 = nltk.ngrams(raw_lit_toks, 2)
bir = [(b, a) for (a, b) in bi0]
cfdr = nltk.ConditionalFreqDist(bir)


# In[63]:


print ("The term ", search_term, " is most commonly preceded by:\n")

cfdr[search_term].most_common(15)


# ## Functions

# ## Multi-Search

# In[53]:


def multi_dic (term):
    
    search_term = re.compile(term)
    
    glos_query_mask = glossary["Term"].str.contains(search_term, na=False)
    glos_query = glossary[glos_query_mask][["UID", "Term", "Translation"]]
    glos_query
    
    
    dehkhoda_query_mask = dehkhoda["Term"].str.contains(search_term, na=False)
    dehkhoda_query = dehkhoda[dehkhoda_query_mask]
    dehkhoda_query    
    
    melz_query_mask = meltzer["Präs.-Stamm"].str.contains(search_term, na=False)
    melz_query = meltzer[melz_query_mask][["Präs.-Stamm", "Deutsch"]]
    melz_query
    
    
    result = print ("Glossary \n\n", glos_query,"\n\n\n",                     "Dehkhoda \n\n", dehkhoda_query,"\n\n\n",                    "Von_Meltzer \n\n", melz_query)

    return result



# In[55]:


#multi_dic ("ب.د")


# ## Simple Conditional Frequency Tool

# In[116]:


def confreq (term, corpus=raw_combo_toks):
        
    bigrams_cfd = nltk.ngrams(corpus, 2)
    cfd = nltk.ConditionalFreqDist(bigrams_cfd)
    output = cfd[term].most_common(5)
    result = print (term, " is most commonly followed by:\n\n", output)
    
    return result


# In[121]:


confreq ("خان")


# In[113]:


bigrams_cfd = nltk.ngrams(raw_combo_toks, 2)
cfd = nltk.ConditionalFreqDist(bigrams_cfd)
output = cfd["خانه"].most_common(5)
result = print ( " is most commonly followed by:\n\n", output)


# ## Regex Concordance

# In[118]:


def regcon (term, corpus=raw_combo_toks):
    # corpus="raw_combo_toks" provides a default argument, which can be overruled.

    search_term = re.compile(term)
    
    freq = nltk.FreqDist(corpus)
    toks = [x for x in combo_freq if re.match(search_term, x)]
    #toks[:5]

    toks = [x for x in combo_freq if re.match(search_term, x)]
    #toks[:5]

    conc0 = sum([trans_corpus.concordance_list(x) for x in toks], [])
    conc1 = [c.line for c in conc0]
    
    result = print('\n'.join(conc1))
    
    return result


# In[119]:


regcon ("خ.ن", raw_nar_toks)


# In[ ]:




