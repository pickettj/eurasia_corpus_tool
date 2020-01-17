
#Arabic-script cleaning routine

import re
def clean_document(doc):
    doc = re.sub(r'ي', 'ی', doc)
    doc = re.sub(r'أ', 'ا', doc)
    doc = re.sub(r'ك', 'ک', doc)
    doc = re.sub(r'ة', 'ه', doc)
    doc = re.sub(r'ۀ', 'ه', doc)
    doc = re.sub(r'مسئله', 'مساله', doc)
    doc = re.sub(r'ئ', 'ی', doc)
    doc = re.sub(r'[^آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهس ي یی ]', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc)
    doc = doc.strip()
    return doc
