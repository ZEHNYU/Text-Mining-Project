import pandas as pd
file='cleaned_dataset_part1_with_All_Sentiment_Scores.csv'
df=pd.read_csv(file)
pd.set_option('display.max_columns', None)

#Dropna, and keep negative score review
df= df[df['Sentiment_Compound_Score'] < 0]
df=df.dropna(subset='app_name')

import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from contractions import CONTRACTION_MAP
import unicodedata

nlp = spacy.load('en_core_web_sm')
tokenizer = ToktokTokenizer()
stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('no')
stopwords.remove('not')

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
df['review_text']=df['review_text'].apply(remove_accented_chars)

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
df['review_text']=df['review_text'].apply(expand_contractions)

def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    return text
df['review_text']=df['review_text'].apply(remove_special_characters)

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
df['review_text']=df['review_text'].apply(lemmatize_text())

def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

df['review_text']=df['review_text'].apply(remove_stopwords)


# print(df_nostopwords)
df.to_csv('process_cleaned.csv',index=False,header=True)