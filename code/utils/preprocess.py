import ast

# fid: code
def get_code_dict(df):
    # print(df['code'][0])
    col = df['code'].apply(lambda x: ' '.join([y[2] for y in ast.literal_eval(x)]))
    return dict(zip(df['fid'], col))


# fid: gid
def get_group_dict(df):
    return dict(zip(df['fid'], df['gid']))
