#!/usr/bin/env python
# coding: utf-8

# # Text Deciphering Tool

# In[2]:


import pickle, re, nltk, os


# In[3]:


import numpy as np
import pandas as pd

from pandas import DataFrame, Series


# In[4]:


#set home directory path
hdir = os.path.expanduser('~')


# Sister files:
# - Pickled corpora cleaned in text_cleaning_tokenizing
# - Corpora stats in corpora_statistics

# ## I. Importing Corpora
# 
# 

# In[5]:


pickle_path = hdir + "/Box/Notes/Digital_Humanities/Corpora/pickled_tokenized_cleaned_corpora"


# In[6]:


with open(pickle_path + "/corpora.pkl", "rb") as f:
    unsorted_doc_toks,                indo_xml_toks, hyd_xml_toks, trans_xml_toks,                trans_nar_toks, indo_nar_toks,                trans_nar_ext_toks, indo_nar_ext_toks, khiva_doc_toks = pickle.load(f)


# In[7]:


with open(pickle_path + "/meta_corpora.pkl", "rb") as f:
    comb_india_nar_toks, comb_trans_nar_toks, nar_corpus_toks, doc_corpus_toks,                comb_india_toks, comb_trans_toks, comb_turk_toks,                combined_corpus_toks, mega_corpus_toks = pickle.load(f)


# In[218]:


# Test:
#"خان" in combined_corpus_toks["tarikh_i_baljuvan_al_biruni_2663iii_ser412"]


# ## II. Importing Raw Tokens
# I.e. tokens without parent text designation, i.e. format necessary for many NLTK routines.

# In[9]:


with open(pickle_path + "/raw_tokens.pkl", "rb") as f:
    raw_doc_toks, raw_nar_toks, raw_indo_toks,                 raw_trans_toks, raw_lit_toks, raw_combo_toks, raw_turk_toks = pickle.load(f)


# In[219]:


# Test:
#indo_nar_toks.keys()


# ## III. Importing Pre-processed NTLK Data

# In[11]:


pickle_data_path = hdir + "/Box/Notes/Digital_Humanities/Corpora/pickled_nltk_data"


# In[16]:


#NLTK Word Frequencies

with open(pickle_data_path + "/frequencies.pkl", "rb") as f:
    combo_freq, pers_lit_freq,                indo_freq, trans_freq,                nar_freq, doc_freq,                turk_freq = pickle.load(f)


# In[344]:


#NLTK Conditional Frequency Dictionaries (raw tokens)

with open(pickle_data_path + "/cfd.pkl", "rb") as f:
    combo_cfd,                indo_cfd, trans_cfd,                nar_cfd, doc_cfd,                turk_cfd,                rev_combo_cfd,                rev_indo_cfd, rev_trans_cfd,                rev_nar_cfd, rev_doc_cfd,                rev_turk_cfd = pickle.load(f)
    


# In[14]:


#NLTK 5-grams (tokens by work)

with open(pickle_data_path + "/fivegrams.pkl", "rb") as f:
    combo_five_grams,                indo_five_grams, trans_five_grams,                nar_five_grams, doc_five_grams,                turk_five_grams = pickle.load(f)


# In[15]:


#Three-way Conditional Frequency Dictionaries (raw tokens)

with open(pickle_data_path + "/tri_cfd.pkl", "rb") as f:
    combo_tricfd,                indo_tricfd, trans_tricfd,                nar_tricfd, doc_tricfd,                turk_tricfd = pickle.load(f)


# ## IV. Importing Datasets

# - Von Melzer Persian Lexicon
# - Glossary
# - Place Names

# In[83]:


# dataset path

ds_path = hdir + "/Box/Notes/Digital_Humanities/Datasets"


# In[121]:


# Von Melzer
meltzer = pd.read_csv(ds_path + "/von_melzer.csv")


# In[355]:


#meltzer["Präs.-Stamm"].sample(5)
#meltzer.sample(10)


# In[86]:


# Locations
locations = pd.read_csv(ds_path + '/exported_database_data/locations.csv', names=['UID', 'Ar_Names',                                                 'Lat_Name', 'Nickname', 'Type'])
# Social Roles
roles = pd.read_csv(ds_path + '/exported_database_data/roles.csv', names=['UID', 'Term', 'Emic', 'Etic', 'Scope'])

# Glossary
glossary = pd.read_csv(ds_path + '/exported_database_data/glossary.csv', names=['UID', 'Term',                                                 'Eng_Term', 'Translation', 'Transliteration', 'Scope', 'Tags'])


# ___
# ___

# In[87]:


dehkhoda = pd.read_csv(ds_path + "/dehkhoda_dictionary.csv", names=['Term', 'Definition'])


# In[88]:


#dehkhoda.sample(10)


# # Basic Search

# Regex reminders:
# - Just the word itself: `^مال$`

# In[89]:


search_term = re.compile(r"ب.د")


# ### Von Melzer Persian Dictionary

# In[207]:


melz_query_mask = meltzer["Präs.-Stamm"].str.contains(search_term, na=False)
melz_query = meltzer[melz_query_mask]
#melz_query


# ### Database Terms

# #### (a) Technical Lexicon

# In[208]:


glos_query_mask = glossary["Term"].str.contains(search_term, na=False)
glos_query = glossary[glos_query_mask]
#glos_query


# #### (b) Social Roles

# In[209]:


roles_query_mask = roles["Emic"].str.contains(search_term, na=False)
roles_query = roles[roles_query_mask]
#roles_query


# #### (c) Place Names

# In[210]:


loc_query_mask = locations["Ar_Names"].str.contains(search_term, na=False)
loc_query = locations[loc_query_mask]
#loc_query


# # Utility Functions
# 
# ----

# In[353]:


def corpora_guide ():
    print(
        "\tCombined Token Corpora:\n\
        \t Narrative Sources from India: comb_india_nar_toks\n\
        \t Narrative Sources from Transoxania: comb_trans_nar_toks\n\n\
        \t All Narrative Sources: nar_corpus_toks\n\
        \t All Document Sources: doc_corpus_toks\n\n\
        \t Documents and Narrative Sources: combined_corpus_toks\n\
        \t Mega Corpus including Persian lit. corpus: mega_corpus_toks\n\n\n\
        Individual Corpora:\n\
        \t External Indic Corpus: indo_nar_ext_toks\n\
        \t External Transoxania Corpus: trans_nar_ext_toks\n\n\
        \t Khiva Turkic Document Corpus: khiva_doc_toks\n\n\
        \t Internal India Narrative Corpus: indo_nar_toks\n\
        \t Internal Transoxania Narrative orpus: trans_nar_toks\n\n\
        \t XML-stage Transoxania Documents: trans_xml_toks\n\
        \t XML-stage Indic Documents: indo_xml_toks\n\
        \t XML-stage Hyderabad Documents: hyd_xml_toks\n\n\
        \t"
                 
    )


# ### Frequency

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

# In[101]:


def best_match (term, corpus):
    
    """Takes a search term and frequency dictionary, returns the most frequently    appearing match within the specified corpus as [matching term, frequency of appearnace]."""
    
    search_term = re.compile(term)
    toks = {k:v for (k,v) in corpus.items() if re.match(search_term, k)}
    if len(toks) > 0:
        match = sorted(toks, key=toks.get, reverse=True)[0]
        freq = corpus[match]
        pair = [match, freq]
    
    else:
        pair = None
    
    return pair


# In[165]:


#help (best_match)

#best_match ("error", nar_freq)

best_match("د.رو", doc_freq)


# In[242]:


def match_freq(term):
    
   
    
    if best_match(term, combo_freq) is not None:
        print ("Most likely match in corpus:\n\n\t",              best_match(term, combo_freq)[0], "appearing ", best_match(term, combo_freq)[1], "times;\n")
    
    search_term = re.compile(term)
    toks = {k:v for (k,v) in combo_freq.items() if re.match(search_term, k)}
    if len(toks) > 3:
        cf2 = sorted(toks, key=toks.get, reverse=True)[1]
        cf3 = sorted(toks, key=toks.get, reverse=True)[2]
        print ("\tfollowed by:\n\t",                    cf2, "appearing ", combo_freq[cf2], "times, and \n\t",                   list(sorted(toks))[2], "appearing ", combo_freq[list(sorted(toks))[2]], "times\n\n")
    
    
    print ("Most likely matches in sub-corpora:\n")
    
    if best_match(term, doc_freq) is not None:
           print("\tDocuments:", best_match(term, doc_freq)[0], "appearing ", best_match(term, doc_freq)[1], "times;\n")
    
    if best_match(term, nar_freq) is not None:
           print ("\tNarrative texts:", best_match(term, nar_freq)[0], "apprearing", best_match(term, nar_freq)[1], "times;\n\n")

    if best_match(term, indo_freq) is not None:
           print ("\tIndic texts:", best_match(term, indo_freq)[0], "appearing ", best_match(term, indo_freq)[1], "times;\n")
    
    if best_match(term, trans_freq) is not None:
           print ("\tTransoxania texts:", best_match(term, trans_freq)[0], "appearing ", best_match(term, trans_freq)[1], "times;\n")
    

    print ("\nMost likely matches in Persian literature corpus:\n\t")
           
    if best_match(term, pers_lit_freq) is not None:
           print ("\t",best_match(term, pers_lit_freq)[0], "appearing ", best_match(term, pers_lit_freq)[1], "times;\n")
    
    

    


# In[111]:


#match_freq("error")


# ### Multi-Search

# In[239]:


def multi_dic (term):
    
    match_freq(term)
    
    search_term = re.compile(term)
    
    glos_query_mask = glossary["Term"].str.contains(search_term, na=False)
    glos_query = glossary[glos_query_mask][["UID", "Term", "Translation"]]
    glos_query
    
    
    dehkhoda_query_mask = dehkhoda["Term"].str.contains(search_term, na=False)
    dehkhoda_query = dehkhoda[dehkhoda_query_mask]
    dehkhoda_query    
    
    melz_query_mask = meltzer["Präs.-Stamm"].str.contains(search_term, na=False)
    melz_query = meltzer[melz_query_mask]
    melz_query[["Präs.-Stamm", "Deutsch"]]
    
    
    result = print ("Glossary \n\n", glos_query,"\n\n\n",                     "Dehkhoda \n\n", dehkhoda_query,"\n\n\n",                    "Von_Meltzer \n\n", melz_query[["Präs.-Stamm", "Deutsch"]])

    return result


# In[253]:


#multi_dic ("دارو+")


# ### Custom KWIC

# In[187]:


# Find in Document

def find_doc(d, s):
    
    """This function takes a dictionary of 5-grams as the first argument,    a regex search term as the second argument, and returns the sequence of 5 words"""
    
    for v in d:
        m = re.match(s, v[2])
        if m is not None:
            yield ' '.join(v)
            

# Note: Return sends a specified value back to its caller
# whereas Yield can produce a sequence of values.


# Example:
## list(find_doc(five_grams['al_biruni_card_catalog_suleimanov_fond'], 'ف.'))


# In[199]:


# Find Corpus
## Produces a generator object with the KWIC with associated work title

def find_corpus(c, s):
    for k, d in c.items():
        for m in find_doc(d, s):
            yield f'{k:50s}: {m}'
            


# In[200]:


def kwic(term, corpus=combo_five_grams):
    
    print('\n'.join(find_corpus(corpus, term)))
    
    # todo: organize this by best match
    


# In[312]:


#kwic("د.رو", indo_five_grams)


# ## Simple Conditional Frequency Tool

# In[290]:


def confreq (term, refine="none"):
    
    """Conditional frequency across multiple corpora and sub-corpora.
        Takes a search term (no regex) for the first word in the bigram, 
        as well as an optional regex filter to narrow down the second word
        in the bigram sequence.
            
    """
    
    if refine == "none" and len(combo_cfd[term]) > 0:
    
        if len(combo_cfd[term]) > 0:
            print (term, " is most commonly followed by:\n\n", combo_cfd[term].most_common(10))

        

        if len(combo_cfd[term]) > 0:
            print("\nWithin sub-corpora:\n")

            # Still need to fill out sub-corpora

            if len(doc_cfd[term]) > 0 :
                print ("\tDocuments:", term, " is most commonly followed by:\n\n\t", doc_cfd[term].most_common(5))
            if len(nar_cfd[term]) > 0 :
                print ("\n\tNarrative texts:", term, " is most commonly followed by:\n\n\t", nar_cfd[term].most_common(5))
            if len(indo_cfd[term]) > 0 :
                print ("\n\n\tIndic texts:", term, " is most commonly followed by:\n\n\t", indo_cfd[term].most_common(5))
            if len(trans_cfd[term]) > 0 :
                print ("\n\tTransoxania texts:", term, " is most commonly followed by:\n\n\t", trans_cfd[term].most_common(5))
    
    # Optional regex refinement of the results:
    if refine != "none" and len(combo_cfd[term]) > 0:
        
        filt = re.compile(refine)
        
        filt_toks = [(x, y) for (x, y) in combo_cfd[term].items() if re.match(refine, x)]
        
        print ("With the results filtered by the regex search (", refine, "), the most likely words following,", term, "are:\n\t", filt_toks)
        
    elif len(combo_cfd[term]) == 0:
            print ("no results")
        

   


# In[292]:


#confreq("قوش")
#help(confreq)


# In[310]:


def regcf (term):
    
    """
        From a regex search term finds the most frequent possible word, then returns 
        conditional frequency (from bigrams) across multiple corpora.
    
    """
    
    if best_match(term, combo_freq) is not None:
        
        local_match = best_match(term, combo_freq)[0]
        
        print ("The most likly match (based on word frequency) for ", term, " is ", local_match,              "(with frequency", best_match(term, combo_freq)[1], ").\n")
        print ("Conditional frequency of", local_match, "(combined corpus):\n\t", combo_cfd[local_match].most_common(5))
        
        print ("\nSub-Corpora:\n")
        
        print ("\n\tDocuments:\n\t", doc_cfd[local_match].most_common(5))
        print ("\n\tNarrative texts:\n\t", nar_cfd[local_match].most_common(5))
        
        print ("\n\n\tIndic texts:\n\t", indo_cfd[local_match].most_common(5))
        print ("\n\tTransoxania texts:\n\t", trans_cfd[local_match].most_common(5))
        
       
    


# In[309]:


#regcf("ق.ش")


# In[ ]:


# TODO: functions for 3-part confreq, and reverse confreq


# ### Third term, if first two known:

# *Document Corpus (Meta-Corpus simply too computationally costly)*

# In[350]:


def tricfd (first_term, second_term, refine="none"):
    
    """
    Given two words in a row, conditional frequency of the third word in the sequence.
    Inputs: two words (in order), and an optional regex filter on the third word.
            
    """
    
    #print ("The pair ", first_term, second_term, " is most commonly followed by :\n")
    #output = combo_tricfd[(first_term, second_term)].most_common(10)
    #print (output)
    
    
    
    if refine == "none" and len(combo_tricfd[(first_term, second_term)]) > 0:
    
        if len(combo_tricfd[(first_term, second_term)]) > 0:
            print ("The pair ", first_term, second_term, " is most commonly followed by:\n\n", combo_tricfd[(first_term, second_term)].most_common(10))


        if len(combo_tricfd[(first_term, second_term)]) > 0:
            print("\nWithin sub-corpora:\n")

            # Still need to fill out sub-corpora

            if len(doc_tricfd[(first_term, second_term)]) > 0 :
                print ("\tDocuments:\n\n\t", doc_tricfd[(first_term, second_term)].most_common(5))
            if len(nar_tricfd[(first_term, second_term)]) > 0 :
                print ("\n\tNarrative texts:\n\n\t", nar_tricfd[(first_term, second_term)].most_common(5))
            if len(indo_tricfd[(first_term, second_term)]) > 0 :
                print ("\n\n\tIndic texts:\n\n\t", indo_tricfd[(first_term, second_term)].most_common(5))
            if len(trans_tricfd[(first_term, second_term)]) > 0 :
                print ("\n\tTransoxania texts::\n\n\t", trans_tricfd[(first_term, second_term)].most_common(5))
    
    # Optional regex refinement of the results:
    if refine != "none" and len(combo_tricfd[(first_term, second_term)]) > 0:
        
        filt = re.compile(refine)
        
        filt_toks = [(x, y) for (x, y) in combo_tricfd[(first_term, second_term)].items() if re.match(refine, x)]
        
        print ("With the results filtered by the regex search (", refine, "), the most likely words following the pair ", first_term, second_term, "are:\n\t", filt_toks)
        
    elif len(combo_tricfd[(first_term, second_term)]) == 0:
            print ("no results")


# In[341]:


#tricfd ("بعد", "از", "خ+")


# ### Reversed conditional frequency, i.e. if second word in sequence known but not first

# *Meta-Corpus*

# In[345]:


def revcfd (term, refine="none"):
    
    
    """Reverse conditional frequency (bigrams) across multiple corpora and sub-corpora.
        Takes a search term (no regex) for the first word in the bigram, 
        as well as an optional regex filter to narrow down the second word
        in the bigram sequence.
            
    """
    
    if refine == "none" and len(rev_combo_cfd[term]) > 0:
    
        if len(rev_combo_cfd[term]) > 0:
            print (term, " is most commonly preceded by:\n\n", rev_combo_cfd[term].most_common(10))

        

        if len(rev_combo_cfd[term]) > 0:
            print("\nWithin sub-corpora:\n")

            # Still need to fill out sub-corpora

            if len(rev_doc_cfd[term]) > 0 :
                print ("\tDocuments:", term, " is most commonly preceded by:\n\n\t", rev_doc_cfd[term].most_common(5))
            if len(rev_nar_cfd[term]) > 0 :
                print ("\n\tNarrative texts:", term, " is most commonly preceded by:\n\n\t", rev_nar_cfd[term].most_common(5))
            if len(rev_indo_cfd[term]) > 0 :
                print ("\n\n\tIndic texts:", term, " is most commonly preceded by:\n\n\t", rev_indo_cfd[term].most_common(5))
            if len(rev_trans_cfd[term]) > 0 :
                print ("\n\tTransoxania texts:", term, " is most commonly preceded by:\n\n\t", rev_trans_cfd[term].most_common(5))
    
    # Optional regex refinement of the results:
    if refine != "none" and len(rev_combo_cfd[term]) > 0:
        
        filt = re.compile(refine)
        
        filt_toks = [(x, y) for (x, y) in rev_combo_cfd[term].items() if re.match(refine, x)]
        
        print ("With the results filtered by the regex search (", refine, "), the most likely words preceding,", term, "are:\n\t", filt_toks)
        
    elif len(rev_combo_cfd[term]) == 0:
            print ("no results")
        

   


# In[349]:


#revcfd("قوش", "خد")


# In[354]:


corpora_guide()


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
