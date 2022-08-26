import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans

col = 'Client'

st.title('Company Name Matching')

file = st.file_uploader('Upload .xlsx', accept_multiple_files=False)

@st.cache
def load_data(file):
    df = pd.read_excel(file, sheet_name='Top 100 +-')
    return df


def get_tfidf_scores(df):
    clients = list(df[col].dropna().unique())
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(clients)
    X = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())
    
    transformer = TfidfTransformer()
    d = transformer.fit_transform(X).toarray()  
    df = pd.DataFrame(data=d, index=X.index.unique(), columns=vectorizer.get_feature_names())

    return df

def knn(df):
    clients = list(df[col].dropna().unique())
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(clients)

    kmeans = KMeans(n_clusters=int(tfidf.shape[0] * 0.9)).fit(tfidf)
    preds = kmeans.predict(tfidf_vectorizer.transform(clients))
    preds = sorted(list(zip(preds, clients)), key=lambda x: x[0])
    preds_df = pd.DataFrame(preds, columns=['class', col])
    return preds_df

def add_suggestion(df):
    pass


if file is not None:
    dataset = load_data(file)
    st.write(dataset[:5])
    st.write('Starting Number of Unique Clients: ', dataset[col].nunique())
    # print length of unique clients in dataset
    
    tfidf = get_tfidf_scores(dataset)
    st.write(tfidf[:5])

    preds = knn(dataset)
    st.title('Suggested matches: ')
    st.write(preds)
    st.write('Suggested Number of Unique Clients: ', preds['class'].nunique())

