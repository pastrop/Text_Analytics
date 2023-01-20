import streamlit as st
import pandas as pd
import yake

#S-BERT packages for embedding calculations
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2', device = None)
model.max_seq_length = 300

#Visualization
from sklearn.decomposition import PCA
#import umap.umap_ as umap
#import altair as alt

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
            #getting embedding for the distilled_docs
            emb_distilled_input = [' '.join(item) for item in distilled_docs]
            emb_distilled = embeds(emb_distilled_input)
            target = st.selectbox('Select Global Concept you want to find the closest documents for (precomputed for higest rated concept)',df_keywords)
            target_emb = embeds(target)
            cosine_score = util.cos_sim(target_emb, emb_distilled)
            final_scores = list(enumerate(cosine_score.flatten().tolist()))
            res = sorted(final_scores, key = lambda x:x[1], reverse=True)
            out_distilled = []
            out_original = []
            for item in res[:5]:
              out_distilled.append(distilled_docs[item[0]])
              out_original.append(texts[item[0]])
            st.text("indexes of 5 closest to the selected concept documents & cosine scores") 
            st.text(res[:5])  
            st.text("5 Documents closest to the selected concept")  
            st.markdown(out_original)  

            #Displaying original document and its keywords
            doc_number = st.number_input('Document & Document Keywords Lookup by Index?', min_value=-1, max_value=len(distilled_docs))
            if doc_number > -1:
                st.text('Original')
                st.markdown(texts_raw[doc_number])
                st.text('Keywords')
                st.text(distilled_docs[doc_number])

            # concepts closeness
            #(1)keywords co-occuring with with the target concept in the documents closest in meaning 
            #(2)kewords from the documents that are close in meaning yet doesn't include the target concept
            closest = set()
            associated = set()
            for item in res[:10]:
              if texts[item[0]].find(target) != -1:
                closest.update(distilled_docs[item[0]]) 
              else:
                associated.update(distilled_docs[item[0]]) 
            closest.remove(target)

            st.text('Keywords co-occuring with with the target concept in the 10 documents closest in meaning ') 
            st.text(set(closest))
            st.text('Keywords from the 10 documents closest in meaning to the target concept in which the target is not present')
            st.text(set(associated))   

            #Vizualization
            pca = PCA(n_components=2)
            #umap_embeds = reducer.fit_transform(emb_texts)
            principal_comp = pca.fit_transform(emb_distilled)
            distilled_texts = [' '.join(item) for item in distilled_docs]
            target_display = st.selectbox('Select Global Concept you want to dispay the closest documents for (precomputed for higest rated concept)',df_keywords)
            #creating groupings to be colored by a different color
            text_search=[True if item.find(selector.value) != -1 or item.find(target_display.lower()) != -1 else False for item in distilled_texts]







