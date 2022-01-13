#!/usr/bin/env python
# coding: utf-8

# # Corpora Cleaning, Tokenizing, Pickling

# ### Libraries

# In[8]:


import arabic_cleaning as ac
import pandas as pd


# In[9]:


import nltk, glob, os, pickle


# ### Paths

# Home Directory

# In[10]:


#set home directory path
hdir = os.path.expanduser('~')

#external relative path
ext_corp_path = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Corpora"

#internal relative path
int_corp_path = hdir + "/Dropbox/Active_Directories/Notes/Primary_Sources"

#pickle path
pickle_path = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Corpora/pickled_tokenized_cleaned_corpora"


# ##### Pre-existing Corpora

# In[11]:


# Indic Narrative
indo_path = ext_corp_path + "/indo-persian_corpora"

# Transoxania Narrative (Persian)
trans_path = ext_corp_path + "/machine_readable_persian_transoxania_texts"

# Khiva documents
khiva_path = ext_corp_path + "/khiva_khanate_chancery_corpus"

# Muscovite Persian diplomatic documents
musc_path = ext_corp_path + "/khorezm_muscovy_diplomatic"

# Persian Lit
perslit_path = ext_corp_path + "/pickled_tokenized_cleaned_corpora"

# Turkic Narrative sources
turk_path = ext_corp_path + "/turkic_corpora"


# ##### Self-created Corpora

# In[12]:


# Indian Narrative
indo_man_path = int_corp_path + "/non-machine-readable_notes/indian_manuscripts"

# Transoxania Narrative
trans_man_path = int_corp_path + "/non-machine-readable_notes/bactriana_notes"

# Transoxania Documents
trans_man_docs_path = int_corp_path + "/xml_notes_stage3_final/bukhara_xml"

# Hyderabad Documents
hyd_man_docs_path = int_corp_path + "/xml_notes_stage3_final/hyderabad_xml"

# Indian Documents (misc. transcribed)
indo_man_docs_path = int_corp_path + "/xml_notes_stage3_final/indic_corpus_xml"

# Qajar Documents (misc. transcribed)
qajar_man_docs_path = int_corp_path + "/xml_notes_stage3_final/qajar_xml"

# Qajar Documents (misc. transcribed)
saf_man_docs_path = int_corp_path + "/xml_notes_stage3_final/qajar_xml"

# Misc Documents (misc. transcribed)
misc_man_docs_path = int_corp_path + "/xml_notes_stage3_final/misc_xml"


# ##### Unorganized Documents

# In[13]:


# Converted to XML, pre-sorted, Stage 2
parser_xml_path = int_corp_path + "/xml_notes_stage2/parser_depository"

# Converted to XML, pre-sorted, Stage 3
updated_docs_path = int_corp_path + "/xml_notes_stage3_final/updater_repository"

# Old system, yet to update
xml_old_sys_path = int_corp_path + "/xml_notes_stage2/xml_transcriptions_old_system"

# Markdown stage
markdown_path = int_corp_path + "/transcription_markdown_drafting_stage1"

# Markdown backlog (old system)
md_backlog_path = int_corp_path + "/transcription_markdown_drafting_stage1/document_conversion_backlog"


# ## Corpus Globbing Section

# ### Pre-existing Corpora

# #### Indic Narrative
# Thackston corpus

# In[14]:


indo_corpus_files = glob.glob(indo_path + r'//**/*.txt', recursive=True)

indo_corpus = {}
for longname in indo_corpus_files:
    with open(longname) as f:
        txt = f.read()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    indo_corpus[short[0]] = txt
    
#indo_corpus.keys()


# #### Transoxania Narrative

# In[15]:


trans_corpus_files = glob.glob(trans_path + r'//**/*.txt', recursive=True)

trans_corpus = {}
for longname in trans_corpus_files:
    with open(longname) as f:
        txt = f.read()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    trans_corpus[short[0]] = txt
    
    
#trans_corpus.keys()


# #### Persian Literature
# *See below*

# #### Khiva Documents

# In[16]:


khiva_corpus_files = glob.glob(khiva_path + r'//**/*.txt', recursive=True)

khiva_corpus = {}
for longname in khiva_corpus_files:
    with open(longname) as f:
        txt = f.read()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    khiva_corpus[short[0]] = txt
    
    
#khiva_corpus.keys()


# #### Turkic Documents
# *TBD*

# #### Muscovite Persian diplomatic documents
# *TBD*

# ### Self-created Corpora

# *Note: need to update processes below to reflect new file organization*

# #### Indic Narrative

# In[17]:


indo_man_files = glob.glob(indo_man_path + r'//**/*.txt', recursive=True)

indo_man = {}
for longname in indo_man_files:
    with open(longname) as f:
        txt = f.read()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    indo_man[short[0]] = txt
    
    
#indo_man.keys()


# #### Transoxania Narrative
# Corpus based on partially transcribed manuscripts from early modern Transoxania.

# In[18]:


trans_man_files = glob.glob(trans_man_path + r'/*.txt')

trans_man = {}
for longname in trans_man_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    trans_man[short[0]] = txt

trans_man.keys()


# #### Transoxania Documents
# Qushbegi documents at XML stage

# In[19]:


trans_man_doc_files = glob.glob(trans_man_docs_path + r'/*.xml')

trans_man_docs = {}
for longname in trans_man_doc_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    trans_man_docs[short[0]] = txt

trans_man_docs.keys()


# #### Hyderabad Documents

# In[20]:


# Hyderabad Documents

hyd_man_doc_files = glob.glob(hyd_man_docs_path + r'/*.xml')

hyd_man_docs = {}
for longname in hyd_man_doc_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    hyd_man_docs[short[0]] = txt

hyd_man_docs.keys()
#Note: nothing in that folder yet.


# #### Indic Documents
# Misc. Indic documents other than those from the Nizam State collection

# In[21]:


ind_man_doc_files = glob.glob(indo_man_docs_path + r'/*.xml')

ind_man_docs = {}
for longname in ind_man_doc_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    ind_man_docs[short[0]] = txt

ind_man_docs.keys()


# ### Unorganized Documents
# E.g. documents still at the markdown stage, and not yet sorted by region.

# #### XML, pre-sorted

# In[22]:


xml_presort_files = glob.glob(parser_xml_path + r'/*.xml')

xml_presort_docs = {}
for longname in xml_presort_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    xml_presort_docs[short[0]] = txt

xml_presort_docs.keys()


# In[23]:


xml_updated_files = glob.glob(updated_docs_path + r'/*.xml')

xml_updated_docs = {}
for longname in xml_updated_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    xml_updated_docs[short[0]] = txt

xml_updated_docs.keys()


# #### XML, old system

# In[24]:


xml_oldsys_files = glob.glob(xml_old_sys_path + r'//**/*.xml', recursive=True)

xml_oldsys_docs = {}
for longname in xml_oldsys_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    xml_oldsys_docs[short[0]] = txt

xml_oldsys_docs.keys()


# ### Pickling XML Corpora

# In[25]:


# Merges

## All final stage XML documents
combo_xml_final = {**ind_man_docs, **hyd_man_docs, **trans_man_docs}
## All XML all stages
combo_xml_all = {**combo_xml_final, **xml_oldsys_docs, **xml_presort_docs, **xml_updated_docs}


combo_xml_all.keys()


# In[26]:


# No need to pickle sub-directories of unsorted XML files
with open(pickle_path + "/xml_corpora.pkl", "wb") as f:
    pickle.dump((ind_man_docs, hyd_man_docs, trans_man_docs,                combo_xml_final, combo_xml_all), f)


# #### Markdown Stage
# Transcribed docs, yet to be ported over to XML

# In[27]:


markdown_files = glob.glob(markdown_path + r'/*.xml')

markdown_docs = {}
for longname in markdown_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    markdown_docs[short[0]] = txt

markdown_docs.keys()
#Will be empty if everything was recently parsed and transfered, per workflow


# #### Markdown, old system

# In[28]:


markdown_old_files = glob.glob(md_backlog_path + r'//**/*.txt', recursive=True)

markdown_old_docs = {}
for longname in markdown_old_files:
    f = open(longname)
    txt = f.read()
    f.close()
    start = os.path.basename(longname)
    short = os.path.splitext(start)
    markdown_old_docs[short[0]] = txt

markdown_old_docs.keys()


# ## Persian Literature Digital Corpus
# Massive corpus of Persian literature, pulled from Ganjur (http://ganjoor.net/) by Roshan (https://persdigumd.github.io/PDL/)
# 
# *Corpus pre-cleaned, tokenized, and pickled from a separate script. (Cleaning takes a long time; and this corpus doesn't change very often, and so does not need to be re-run.)*

# In[29]:


f = open(perslit_path + '/persian_lit_toks.pkl', 'rb') 

pers_lit_toks = pickle.load(f)
f.close()


# In[30]:


#pers_lit_toks.keys()
#pers_lit_toks["hafez.masnavi"][:50]
#pers_lit_toks['ferdowsi.shahnameh']

#type (pers_lit_toks['ferdowsi.shahnameh'][5])


# ### Cleaning edited texts and notes

# In[31]:


# possible to do this once by iterating over the following? crashed computer last time...

# indo_corpus, trans_corpus, khiva_corpus
# indo_man, trans_man
# trans_man_docs, hyd_man_docs, ind_man_docs
# xml_presort_docs, xml_oldsys_docs, markdown_docs, markdown_old_docs


clean_indo = {fn: ac.clean_document(doc) for fn, doc in indo_corpus.items()}
clean_trans = {fn: ac.clean_document(doc) for fn, doc in trans_corpus.items()}
clean_khiva = {fn: ac.clean_document(doc) for fn, doc in khiva_corpus.items()}

clean_indo_man = {fn: ac.clean_document(doc) for fn, doc in indo_man.items()}
clean_trans_man = {fn: ac.clean_document(doc) for fn, doc in trans_man.items()}

clean_trans_man_docs = {fn: ac.clean_document(doc) for fn, doc in trans_man_docs.items()}
clean_hyd_man_docs = {fn: ac.clean_document(doc) for fn, doc in hyd_man_docs.items()}
clean_ind_man_docs = {fn: ac.clean_document(doc) for fn, doc in ind_man_docs.items()}

clean_xml_presort_docs = {fn: ac.clean_document(doc) for fn, doc in xml_presort_docs.items()}
clean_xml_oldsys_docs = {fn: ac.clean_document(doc) for fn, doc in xml_oldsys_docs.items()}
clean_markdown_docs = {fn: ac.clean_document(doc) for fn, doc in markdown_docs.items()}
clean_markdown_old_docs = {fn: ac.clean_document(doc) for fn, doc in markdown_old_docs.items()}



#clean_trans['ikromcha'][:1000]
#clean_trans['ikromcha'][:1000]


#clean_xml['ser561']

#clean_indo['mu_vol1'][:1000]


# ## Tokenizing

# In[32]:


#apparently this dependency is needed for below
nltk.download('punkt')


# In[33]:



# External Corpora Toks

indo_nar_ext_toks = {}
for (fn, txt) in clean_indo.items():
    toks = nltk.word_tokenize(txt)
    indo_nar_ext_toks[fn] = toks

trans_nar_ext_toks = {}
for (fn, txt) in clean_trans.items():
    toks = nltk.word_tokenize(txt)
    trans_nar_ext_toks[fn] = toks 
    
khiva_doc_toks = {}
for (fn, txt) in clean_khiva.items():
    toks = nltk.word_tokenize(txt)
    khiva_doc_toks[fn] = toks

    
# Manually Entered Manuscript Toks

indo_nar_toks = {}
for (fn, txt) in clean_indo_man.items():
    toks = nltk.word_tokenize(txt)
    indo_nar_toks[fn] = toks
    
trans_nar_toks = {}
for (fn, txt) in clean_trans_man.items():
    toks = nltk.word_tokenize(txt)
    trans_nar_toks[fn] = toks

# Clean XML-stage Document Toks
 
trans_xml_toks = {}
for (fn, txt) in clean_trans_man_docs.items():
    toks = nltk.word_tokenize(txt)
    trans_xml_toks[fn] = toks
    
hyd_xml_toks = {}
for (fn, txt) in clean_hyd_man_docs.items():
    toks = nltk.word_tokenize(txt)
    hyd_xml_toks[fn] = toks

indo_xml_toks = {}
for (fn, txt) in clean_ind_man_docs.items():
    toks = nltk.word_tokenize(txt)
    indo_xml_toks[fn] = toks


# Unorganized Markdown-stage Toks


presort_xml_toks = {}
for (fn, txt) in clean_xml_presort_docs.items():
    toks = nltk.word_tokenize(txt)
    presort_xml_toks[fn] = toks
    
oldsys_xml_toks = {}
for (fn, txt) in clean_xml_oldsys_docs.items():
    toks = nltk.word_tokenize(txt)
    oldsys_xml_toks[fn] = toks
    
md_stage_toks = {}
for (fn, txt) in clean_markdown_docs.items():
    toks = nltk.word_tokenize(txt)
    md_stage_toks[fn] = toks

md_oldsys_toks = {}
for (fn, txt) in clean_markdown_old_docs.items():
    toks = nltk.word_tokenize(txt)
    md_oldsys_toks[fn] = toks


# *First-stage combinations*: Collapse unsorted documents

# In[34]:


unsorted_doc_toks = {**presort_xml_toks, **oldsys_xml_toks, **md_stage_toks, **md_oldsys_toks}


# In[35]:


#unsorted_doc_toks['ser560']


# ### Pickling Corpora

# In[36]:


pickle_path = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Corpora/pickled_tokenized_cleaned_corpora"


# In[37]:


with open(pickle_path + "/corpora.pkl", "wb") as f:
    pickle.dump((unsorted_doc_toks,                indo_xml_toks, hyd_xml_toks, trans_xml_toks,                trans_nar_toks, indo_nar_toks,                trans_nar_ext_toks, indo_nar_ext_toks, khiva_doc_toks), f)


# In[ ]:





# In[38]:


#trans_nar_toks["ziyarat_bukhara_kazan_manuscript_ser492"]

df = pd.DataFrame (trans_nar_toks["ziyarat_bukhara_kazan_manuscript_ser492"], columns=['Token'])
df['Text']='title'


# ### Corpus Formation: Dataframes

# In[39]:


# External Corpora Toks

concat_indo_nar_ext_toks = sum([[("indo_nar_ext_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in indo_nar_ext_toks.items()], [])

concat_trans_nar_ext_toks = sum([[("trans_nar_ext_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in trans_nar_ext_toks.items()], [])

concat_khiva_doc_toks = sum([[("khiva_doc_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in khiva_doc_toks.items()], [])


    
# Manually Entered Manuscript Toks

concat_trans_nar = sum([[("trans_nar", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in trans_nar_toks.items()], [])

concat_indo_nar = sum([[("indo_nar", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in indo_nar_toks.items()], [])


# Clean XML-stage Document Toks
 
concat_trans_xml_toks = sum([[("trans_xml_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in trans_xml_toks.items()], [])

concat_hyd_xml_toks = sum([[("hyd_xml_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in hyd_xml_toks.items()], [])

concat_indo_xml_toks = sum([[("indo_xml_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in indo_xml_toks.items()], [])



# Unorganized Markdown-stage Toks

concat_presort_xml_toks = sum([[("presort_xml_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in presort_xml_toks.items()], [])

concat_trans_nar = sum([[("oldsys_xml_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in oldsys_xml_toks.items()], [])

concat_md_stage_toks = sum([[("md_stage_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in md_stage_toks.items()], [])

concat_md_oldsys_toks = sum([[("md_oldsys_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in md_oldsys_toks.items()], [])



# In[40]:


# Persian Lit

concat_pers_lit_toks = sum([[("pers_lit_toks", text, idx, tok) for idx, tok in enumerate(toks)] for text, toks in pers_lit_toks.items()], [])



# In[41]:


# just delete the 'test' column above; do the different categories separately, manually specifying the category
# then just concat them all together at the end.


# In[42]:


#concat_indo_nar[0:10]

concat = concat_indo_nar_ext_toks + concat_trans_nar_ext_toks + concat_khiva_doc_toks + concat_trans_nar + concat_indo_nar + concat_trans_xml_toks + concat_hyd_xml_toks + concat_indo_xml_toks +concat_presort_xml_toks + concat_trans_nar + concat_md_stage_toks + concat_md_oldsys_toks



# In[43]:


concat = concat + concat_pers_lit_toks


# In[44]:


df = pd.DataFrame(concat, columns = ["Category", "Text", "No", "Token"])


# In[51]:


df.sample()


# In[52]:


df.to_csv(os.path.join(pickle_path,r'eurasia_corpus.csv'), index=False)


# In[53]:


df[5:10]

