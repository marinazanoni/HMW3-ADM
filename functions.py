import pandas as pd
import json
from collections import defaultdict
from bs4 import BeautifulSoup
import requests
import os
import re
from collections import Counter
from functools import reduce
from tqdm.notebook import tqdm
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
from itertools import product
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import *


def extract_masters(this_url):
    result_url = requests.get(this_url)
    result_soup = BeautifulSoup(result_url.text, 'html.parser')
    result_links = result_soup.find_all('a', {'class': 'courseLink'})
    result_list = []
    for item in result_links:
        result_list.append(item['href'])
    return result_list



def parse_html(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    parsed_dfs = []
    # Iterate through all folders and subfolders using os.walk
    for folder_path, _, file_names in os.walk(folder_path):
        # Check if there are files in the current folder
        if file_names:
            # Iterate through each file in the current folder
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)

                # Store the information only is the dictionary is not empty (has at list a name course)
                try:
                    # Parse the file and append the result to the list
                    parsed_df = custom_functions.parser(file_path)
                    parsed_dfs.append(parsed_df)
                except Exception as e:
                    # Print the file path when an exception occurs
                    print(f"Error parsing file: {file_path}")
                    # print(f"Error details: {e}")

    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(parsed_dfs, ignore_index=True)
    return concatenated_df


# Function to remove stopwords and punctuation from a text
def clean(text):
    """
    The following function returns the filtered element for each column of a dataframe.
    Filtering operation consists in removing punctuation and removing stopwords given text with lower case
    """
    words = word_tokenize(str(text))
    # Remove punctuation using NLTK and string.punctuation
    filtered_words = [word for word in words if word not in string.punctuation + "'’...?'+,-'‘“”„•…›✓"]
    # Remove stopwords
    filtered_words = [word for word in filtered_words if word.lower() not in stop_words]
    return ' '.join(filtered_words)



# Function to convert any currency to the common currency (USD in this case)
def convert_to_common_currency(target_currency='USD',currency_symbol = '£',amount=0):
    try:
        # Map the currency symbol to the API symbol
        api_currency_symbol = currency_symbol_mapping.get(currency_symbol)

        if not api_currency_symbol:
            return None

        # Extract the exchange rate from the pre-fetched rates
        exchange_rate = exchange_rates[api_currency_symbol]

        # Remove the currency symbol and commas, then convert to float
        amount = float(amount.replace(',', ''))

        # Convert to USD using the obtained exchange rate
        amount_target_currency = amount/(exchange_rate)
        return round(amount_target_currency,2)
        #return currency_symbol

    except Exception as e:
       return None


def return_cost(text):
    """
    return the maximum fees converted to USD given a text (description column)
    """
    # Return None if the input is not a string
    if not isinstance(text, str):
        return None  

    matches = re.finditer(pattern, text)
    converted_list = []

    for match in matches:
        value = match.group('value')
        if match.group('symbol_before'):
            symbol_before = match.group('symbol_before')
            # Combine symbol_before, value, and symbol_after into a single string
            converted_list.append(functions.convert_to_common_currency(currency_symbol = symbol_before,amount=value.replace(',','')))

        elif match.group('symbol_after'):
            symbol_after = match.group('symbol_after')
            converted_list.append(functions.convert_to_common_currency(currency_symbol= symbol_after,amount=value.replace(',','')))

    # Drop None values using a list comprehension
    if len(converted_list)>=1:
        filtered_list = [value for value in converted_list if value is not None]
        if len(filtered_list)>=1:
            return(max(filtered_list))

def rank_documents(query1):
    """
    Given a query, computes the tfidf for that query and evaluate the cosine similarity con for query given
    the document extracting with engine function. Returns the top-k documents
    """
    ## COMPUTING THE IFIDF FOR THE QUERY

   # Tokenize the query and save the stammed words for computinf tfidf
    tokens = word_tokenize(query1)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    tfidf_vectorizer = TfidfVectorizer()

    # Combine the stemmed tokens considering it single string (document)
    stemmed_query = ' '.join(stemmed_tokens)

    # Fit the vectorizer on the list of queries and transform the query into a tfidf vector
    query_tfidf = tfidf_vectorizer.fit_transform([stemmed_query])

    # Convert the TF-IDF matrix to a dense pandas DataFrame
    tfidf_query = pd.DataFrame(query_tfidf.todense(), index=['query'], columns=tfidf_vectorizer.get_feature_names_out())

    ## OBTAINE THE TFIDF BEWTEEN THE QUERY AND THE DOCUMENT
    # Initialize a heap for the k-top results
    k_top_results = []

    # Iterate over the documents
    for idx, document_text in enumerate(engine(query1)['description']):

        # Tokenize and preprocess the document text
        document_tokens = [stemmer.stem(word) for word in word_tokenize(document_text)]
        document_text_processed = ' '.join(document_tokens)

        # Transform the document text using the fitted vectorizer
        tfidf_document = tfidf_vectorizer.transform([document_text_processed])

        # Compute cosine similarity between the query and the document
        similarity_score = cosine_similarity(tfidf_query, tfidf_document)[0, 0]

        # Display the cosine similarity score
        #print("Cosine Similarity Score:", similarity_score,idx)

       ## USING A HEAP TO HAVE THE TOP-K DOCUMENTS
        # Append the similarity score and document index to the k_top_results list
        k_top_results.append((round(similarity_score,5), engine(query1)['description'].index[idx]))


    # Retrieve the k-top results from the heap using nlargest
    k_top_results = heapq.nlargest(10, k_top_results, key=lambda x: x[0])
    rank_df = pd.DataFrame()

    for cossim, indexsorted in k_top_results:
        # Return the information about the document sorted by cosine similarity
            document_info = df_query.loc[[indexsorted]].copy()
#            return document_info
            document_info['Similarity'] = cossim
            rank_df = pd.concat([rank_df, document_info])

    # Eventually we could reset the index of the final DataFrame
    # rank_df = rank_df.reset_index(drop=True)

    return rank_df
