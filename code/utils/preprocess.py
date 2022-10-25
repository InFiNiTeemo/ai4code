import ast

# fid: code
def get_code_dict(df):
    # print(df['code'][0])
    col = df['code'].apply(lambda x: ' '.join([y[2] for y in ast.literal_eval(x)]))
    return dict(zip(df['fid'], col))


# fid: gid
def get_group_dict(df):
    return dict(zip(df['fid'], df['gid']))


# to shorter the text,
import re
def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    document = document.replace('_',' ')

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()
    return document
