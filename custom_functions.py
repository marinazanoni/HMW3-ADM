
# Function to download HTML from a URL with prefix and save it to a file
def crawler(url, output_path):
    full_url = 'https://www.findamasters.com/' + url
    try:
        response = requests.get(full_url)
        if response.status_code == 200:
            with open(output_path, 'w', encoding='utf-8') as html_file:
                html_file.write(response.text)
        else:
            print(f"Failed to download {full_url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {full_url}: {e}")
        
        
def engine(query):
    """
    Given a query made up by multiple word it returns the documents were ALL the word are found
    """
    doc_set_indexes = []
    words_in_query = query.split()

    for word in words_in_query:
        # Stemming the word
        stemmed_word = stemmer.stem(word)

        # Check if the stemmed word exists in the 'Word' column after applying stemming
        if stemmed_word in vocabulary_reverse['Word'].apply(stemmer.stem).values:
            # Get the document set indexes for the stemmed word
            indexes_for_word = vocabulary_reverse[vocabulary_reverse['Word'].apply(lambda x: x == stemmed_word)]['reverse'].values

            # Flatten the lists in 'reverse' column
            flattened_indexes = [item for sublist in indexes_for_word for item in sublist]

            # Append the flattened document set indexes to the list
            doc_set_indexes.append(flattened_indexes)
           # print(doc_set_indexes)

        else:
            print(f"Stemmed word '{stemmed_word}' not found in vocabulary_reverse")

    # Find the intersection of all document sets
    selected_doc = list(set.intersection(*map(set, doc_set_indexes)))

    # Select rows using iloc
    selected_rows = df.iloc[selected_doc]

    return selected_rows
