import streamlit as st
import pandas as pd
import yake

st.title('Text Analytics')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    #st.write(df)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

df_clean = df[df['text'].apply(lambda x: isinstance(x, str))]
texts = [item.replace("\t", " ") for item in df_clean['text']]


# Keywords Identification for the whole text corpus and individual documents

# YAKE Config for the entire text corpus
number_of_concepts = st.number_input('How many concepts you want?', key=int)
kw_extractor = yake.KeywordExtractor()
language = 'en'
max_ngram_size = 2
deduplication_threshold = 0.9
numOfKeywords = 50

#Keyword for the corpus a.k.a Global Concepts
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

keyword_extraction_state = st.text('Extracting Global Concepts...')
keywords = custom_kw_extractor.extract_keywords(corpus)
keyword_extraction_state.text('Loading data...done!')