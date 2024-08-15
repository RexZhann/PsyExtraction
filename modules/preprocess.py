import fitz
from pysbd import Segmenter
import re
import os
import numpy as np


segmenter = Segmenter()


# function for loading txt files (rec)
def read_txts_to_list(folder_path):
    txt_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()

                # Brush the text to remove exe's
                full_text = re.sub(r"\xa0", " ", full_text)
                full_text = re.sub(r"-\n ", "", full_text)
                full_text = re.sub(r"  ", "", full_text)
                full_text = re.sub(r" - ", "-", full_text)

                # Segement into sentences using PySBD
                # Alternatively, can use this paragraph-level segmenter
                # txt_texts.append(full_text.split('\n\n'))

                txt_texts.append(segmenter.segment(full_text.lower()))

    return txt_texts


# function for loading pdf files
def read_pdfs_to_list(folder_path):
    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            with fitz.open(file_path) as doc:
                full_text = ""
                for page in doc:
                    full_text += page.get_text()

                full_text = re.sub(r"\n", " ", full_text)
                full_text = re.sub(r"\xa0", " ", full_text)
                full_text = re.sub(r"- ", "", full_text)
                pdf_texts.append(segmenter.segment(full_text.lower()))

    return pdf_texts


# function for filtering out the sentences that includes the keyword
def find_related_sent(keyword, sentences, win_size=5):

    related_sent = []

    for sent_per_text in sentences:
        for i, sent in enumerate(sent_per_text):
            if keyword.lower() in sent:
                for j in range(np.max([0, i - win_size]), np.min([i + win_size, len(sent_per_text)])):
                    if sent_per_text[j] not in related_sent:
                        if len(sent_per_text[j]) > 1300:
                            related_sent.append(sent_per_text[j][:500])
                            related_sent.append(sent_per_text[j][501: ])
                        else:
                            related_sent.append(sent_per_text[j])

    articles = list(set(related_sent))
    articles = [phrase for phrase in articles if len(phrase) > 4]

    if len(articles) > 450:
        return articles[30:480]
    else:
        return articles


folder_path = 'D:\\RexZhann\\nlp\\papers'

txt_texts = read_txts_to_list(folder_path)
pdf_texts = read_pdfs_to_list(folder_path)
