import dashscope
from dashscope import Generation
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords


# exclude the keyword from the phrases obtained
def exclude_key(phrases_llm, keyword, stop_words):
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

    tokenized_results = []
    
    with tqdm(total=len(articles) // batchsize + (len(articles) % batchsize > 0)) as pbar:
        for idx, batch in enumerate(phrases_llm):

            debug_info = f"Processing batch {idx + 1}/{len(art_batch) // batchsize + (len(art_batch) % batchsize > 0)} with {len(batch)} phrases"
            
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
            tokenized_results.append(rsp.output.text.strip())
            
            pbar.update(len(batch))  
    
    return tokenized_results


