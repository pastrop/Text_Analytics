import streamlit as st
import pandas as pd
import yake

#S-BERT packages for embedding calculations
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2', device = None)
model.max_seq_length = 300

# Functions to be used

#Global Extractor
@st.cache
def global_extractor(number_of_concepts,corpus):
    kw_extractor = yake.KeywordExtractor()
    language = 'en'
    max_ngram_size = 2
    deduplication_threshold = 0.9
    numOfKeywords = number_of_concepts

    #Keyword for the corpus a.k.a Global Concepts
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, 
                                                top=numOfKeywords, features=None
                                                )
    keywords = custom_kw_extractor.extract_keywords(corpus)
    return keywords

#Keyword for the individual documents
@st.cache
def doc_keys_extractor(texts):
    language ='en'
    max_ngram_size = 2
    deduplication_threshold = 0.9
    custom_kw_extractor_docs = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, 
                                                    top=10, features=None
                                                    )
    distilled_docs = []
    for item in texts:
      keywrds = custom_kw_extractor_docs.extract_keywords(item)
      tmp = [item[0] for item in keywrds]
      distilled_docs.append(tmp)
    keyword_extraction_state_docs.text('Extracting Document Keywords Done...')
    return  distilled_docs      

@st.cache
def embeds(input):
    emb = model.encode(input)
    return emb


st.title('Text Analytics')

uploaded_file = st.file_uploader("Choose a file, has to be .csv with text for analysis in a column named 'text' ")

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    df_clean = df[df['text'].apply(lambda x: isinstance(x, str))]
    texts_raw = df['text'].tolist()
    texts = [item.replace("\t", " ") for item in df_clean['text'][:1000]]
    #Create a single blob of text
    corpus = ' '.join(texts)


    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(df)



    # Keywords Identification for the whole text corpus and individual documents

    # YAKE Config for the entire text corpus
    st.title('_Keyword Extractor_')
    number_of_concepts = st.number_input('How many concepts you want?', min_value=0, max_value=100)
    if number_of_concepts > 0:
        #kw_extractor = yake.KeywordExtractor()
        #language = 'en'
        #max_ngram_size = 2
        #deduplication_threshold = 0.9
        #numOfKeywords = number_of_concepts

        #Keyword for the corpus a.k.a Global Concepts
        #custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

        keyword_extraction_state = st.text('Extracting Global Concepts...')
        keywords = global_extractor(number_of_concepts,corpus)
        keyword_extraction_state.text('Extracting...done!')
        keywords_dict = {'global concept':[item[0] for item in keywords], 'score, less the better!':[item[1] for item in keywords]}
        df_keywords = pd.DataFrame(keywords_dict)
        st.dataframe(df_keywords)


        #Keyword for the individual documents
        #Summarizing each document to a set of keywords
        keyword_extraction_state_docs = st.text('Extracting Document keywords...')
        distilled_docs = doc_keys_extractor(texts)
        if distilled_docs:  
            doc_number = st.number_input('Which Document you want to see?', min_value=-1, max_value=len(distilled_docs))
            if doc_number > -1:
                st.text('Original')
                st.markdown(texts_raw[doc_number])
                st.text('Keywords')
                st.text(distilled_docs[doc_number])







