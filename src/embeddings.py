import openai
from ast import literal_eval
from scipy.spatial import distance
import pandas as pd


def get_ada_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# When reading the embeddings from a file, we need to convert the column containing the embeddings to a list
def load_csv_file_with_embeddings(filename, embedding_column_name, sep='|',encoding='utf-8'):
    df = pd.read_csv(filename,sep=sep,encoding=encoding)
    df[embedding_column_name] = df[embedding_column_name].apply(literal_eval)
    return df

def get_closest_nodes(df, embedding_column_name, question_embedding, n=4):
    # Calculate the cosine distance
    df['cosine_dist'] = df[embedding_column_name].apply(lambda x: distance.cosine(x, question_embedding))
    # Sort the DataFrame based on the cosine distance and get the top n rows
    closest_nodes = df.nsmallest(n, 'cosine_dist')
    return closest_nodes
