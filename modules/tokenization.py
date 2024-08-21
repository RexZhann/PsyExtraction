import dashscope
from dashscope import Generation
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
import json


stop_words = set(stopwords.words('english'))


def exwords(keyword, stop_words=stop_words):
    return set([word for word in keyword.split() if word not in stop_words])



# summarize each entry, would be helpful for paragraphs
def para_sum(articles, keyword, dep_key, modl='qwen-max'):

    res = []
    for i, para in enumerate(articles):

        prompt = f'''complete the #OBJECTIVE# based on the #CONTEXT#, and generate the output based on #STYLE# and #RESPONSE#
        # CONTEXT #
        You are an professional and seasoned expert with 20 years experience of summarizing text based on the deep connections 
        within context and dependency of the text, including recognizing any named entities, possible phrases and chunks of 
        nouns especially prepositional phrases with objects of preposition. You are also good at recognizing the logical relation 
        between the phrases in one context. The purpose of summarization is to explore the {dep_key} of {keyword} based on the given paragraphs.
        # OBJECTIVE #
        Summarize this paragraph: {para}
        # STYLE #
        output only the summarization result, all the words chosen must be originally from the given paragraph. Skip any paragraph with no complete sentence structure and use the original para as the output
        Be extra careful when a paragraph is likely to contain the {dep_key} of {keyword}
        # RESPONSE #
        do not output anything other than the resulted summary. Do not write any explanatory contents for the summarization. Output English only
        '''
        
        rsp = Generation.call(model=modl, prompt=prompt)
        res.append(rsp.output.text.lower())
    
    return res


#tokenize the article by grammatical phrases
def tokenizer_batch(articles, keyword, dep_key, modl='qwen-max', batchsize=5):

    def batch_generator(articles, batchsize):
        for i in range(0, len(articles), batchsize):
            yield articles[i:i + batchsize]
    
    res = []
    
    with tqdm(total=len(articles) // batchsize + (len(articles) % batchsize > 0)) as pbar:
        for idx, batch in enumerate(batch_generator(articles, batchsize)):
            
            debug_info = f"tokenizing batch {idx + 1}/{len(articles) // batchsize + (len(articles) % batchsize > 0)} with {len(batch)} phrases"
            
            print(f"Debug: {debug_info}")  
            
            prompt = f'''complete the #OBJECTIVE# based on the #CONTEXT#, and generate the output based on #STYLE# and #RESPONSE#
            # CONTEXT #
            You are an professional and seasoned expert with 20 years experience of tokenizing text based on the deep connection 
            within context and dependency of the text, including extracting out any named entities, possible phrases and chunks of 
            nouns especially prepositional phrases with objects of preposition. You are also good at recognizing the logical relation 
            between the phrases in one context. The purpose of tokenization is to explore the {dep_key} of {keyword} based on the given paragraphs.
            # OBJECTIVE #
            tokenize these paragraphs by phrases and chunks of nouns:{' '.join(batch)}
            # STYLE #
            output only the result of tokenization, separated by quotation and comma, no linebreaks. Prefer chunks of nouns than 
            single nouns, try to output chunks of nouns when possible. Be extra careful when a paragraph is likely to contain the {dep_key} of {keyword}
            # RESPONSE #
            do not output anything other than the result of tokenization, do not output too many single-word tokens. Do not output 
            complete sentences.Do not output any stopwords that's not part of a phrase.
            '''
            
            rsp = Generation.call(model=modl, prompt=prompt)
            res.append(rsp.output.text.strip())
            
            pbar.update(len(batch))  
    
    
    return [re.findall(r'"(.*?)"', phrase) for phrase in res]


# remove single stopwords
def remove_sw(phrases_llm, stop_words=stop_words):
    

    p1 = r'^(f\d{2}-f\d{2})?'

    ph = [[re.sub(p1, '', phrase.lower()) for phrase in phrases 
           if len(phrase) <= 40 and len(re.sub(r'^(f\d{2}?)', '', phrase)) > 4 
           and '%' not in phrase and phrase not in stop_words] for phrases in phrases_llm]
    
    ph = [phrases for phrases in ph if len(phrases) != 0]

    return ph


# exclude the keyword from the phrases obtained
def exclude_key(phrases_llm, keyword, stop_words, ex_words=['']):
    phrase_res = []

    ex_words = set([word for word in keyword.split() if word not in stop_words])

    for sub in phrases_llm:
        sub = [phrase for phrase in sub if not any(ex in phrase for ex in ex_words)]
        phrase_res.append(sub)

    return phrase_res


# generate the corresponding dependency labels of each phrase
def dep_reco_batch(articles, phrases_llm, keyword, dep_key, batchsize=5, modl='qwen-max'):
    def batch_generator(articles, batchsize):
        for i in range(0, len(articles), batchsize):
            yield articles[i:i + batchsize]
    
    art_batch = list(batch_generator(articles, batchsize=5))

    dep_res = []
    
    with tqdm(total=len(articles) // batchsize + (len(articles) % batchsize > 0)) as pbar:
        for idx, batch in enumerate(phrases_llm):

            debug_info = f"Dependency parsing batch {idx + 1}/{len(articles) // batchsize + (len(articles) % batchsize > 0)} with {len(batch)} phrases"
            
            print(f"Progress: {debug_info}")  
            
            prompt = f'''complete the #OBJECTIVE# based on the #CONTEXT#, and generate the output based on #STYLE# and #RESPONSE#
            # CONTEXT #
            You are an professional and seasoned expert with 20 years experience of recognizing the dependency relation of phrases based on the deep connection 
            within context and dependency of the text, including recognizing any named entities, possible phrases and chunks of 
            nouns especially prepositional phrases with objects of preposition. You are also good at recognizing the logical relation 
            between the phrases in one context. The purpose of dependency recognition is to explore the {dep_key} of {keyword} based on the given paragraphs.
            # OBJECTIVE #
            the list phrases of the context are here: {batch}, and the entire context is here: {' '.join(art_batch[idx])}
            # STYLE #
            You have six choices for labels: 'sub' for the phrase '{keyword}', 'cat' for phrases in the similar categories with 'sub', 'par' for synonyms of 
            the phrase '{dep_key}', 'des' for descriptions for the 'sub' and 'par' in the same sentence, 'dob' for the objects induced by {keyword} in the 
            sentences, and 'oth' for others in the sentences. Output a list of labels corresponding to each phrase in the input list that best fits their dependency.
            # RESPONSE #
            do not output anything other than the resulted list, do not output any label that's not in the given six. Any phrases with 'disease' inside should be 
            labeled 'cat' or 'sub'. The label 'des' usually means the phrase represents the key symptoms occurs alongside with the diagnosis of '{keyword}'
            '''
            
            rsp = Generation.call(model=modl, prompt=prompt)
            dep_res.append(rsp.output.text.strip())
            
            pbar.update(len(batch))  
    dep_res = [phrase.replace("'", '"') for phrase in dep_res]
    dep_res = [json.loads(phrase) for phrase in dep_res]
    
    return dep_res


def obtain_dep_phrases(phrase_res, dep_res, labels=['dob', 'des', 'oth'], ex_words=['']):
    desc_phrases = [[phrase for phrase, label in zip(sublist_a, sublist_b) 
                     if label in labels] for sublist_a, sublist_b in zip(phrase_res, dep_res)]
    desc_phrases = [phrases for phrases in desc_phrases if len(phrases) != 0]

    desc_ph = []

    for phrases in desc_phrases:
        phr = [phrase for phrase in phrases if not any(ex in phrase for ex in ex_words)]
        desc_ph.append(phr)

    desc_ph = [phrases for phrases in desc_ph if len(phrases) != 0]

    return desc_phrases

