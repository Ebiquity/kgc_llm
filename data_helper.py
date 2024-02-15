import variables as V
import pandas as pd
from datasets import Dataset, DatasetDict

def read_data(file_name):
    '''
    To read data from pickle file
    '''
    dataset = pd.read_pickle(file_name)
    df = pd.DataFrame(dataset)
    return df

def add_s_r_o(df, args):

    '''
    To split the combined facts from "subject; relation; object" into
    3 new columns subject, relation, and object
    1 new column for human annotation
    '''
    df['subject'] = [i.split(";")[0] for i in df[0].tolist()]
    df['relation'] = [i.split(";")[1] for i in df[0].tolist()]
    df['object'] = [i.split(";")[2] for i in df[0].tolist()]
    if args.dataset_name == 'tac_2015':
        df['consistent'] = df['relation'] == df[3] # tac 2015 dataset processing have one item extra
    else:
        df['consistent'] = df['relation'] == df[2]
    return df


def add_qa_column(df):
    '''
    Combine the fact and extracted provenance into the multi choice prompt
    '''
    new_column = []

    for idx, row in df.iterrows():
        triple = f'{row["subject"]} {row["relation"]} {row["object"]}'
        text = row[1]

        filled_prompt = V.PROMPT.format(text, triple)
        new_column.append(filled_prompt)

    df['text'] = new_column
    return df

def add_label_columns(df):
    '''
    Convert the True/False into multiple choice option "(a)" and "(b)"
    '''

    labels = []
    for idx, row in df.iterrows():
        if row['consistent']:
            labels.append(V.TRUE_LABEL)
        else:
            labels.append(V.FALSE_LABEL)

    df['label'] = labels
    return df

def add_columns(df_train, df_test, df_valid, args):
    '''
    To add additional columns to the dataframe
    '''
    df_test = add_s_r_o(df_test, args)
    df_train = add_s_r_o(df_train, args)
    df_valid = add_s_r_o(df_valid, args)

    df_test = add_qa_column(df_test)
    df_train = add_qa_column(df_train)
    df_valid = add_qa_column(df_valid)

    df_test = add_label_columns(df_test)
    df_train = add_label_columns(df_train)
    df_valid = add_label_columns(df_valid)

    return df_train, df_test, df_valid



def df2dataset_dict(df_train, df_test, df_valid):
    '''
    Convert pandas dataframe to huggingface dataset
    '''
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    valid_dataset = Dataset.from_pandas(df_valid)
    dataset_dict = DatasetDict({"train":train_dataset,"test":test_dataset, "valid":valid_dataset})
    return dataset_dict


def remove_columns(dataset_dict):
    '''
    Remove columns that are not needed by huggingface Trainer
    '''
    dataset_dict = dataset_dict.remove_columns(['0', '1', '2', 'subject', 'relation', 'object',  'consistent'])
    return dataset_dict


def remove_columns_tokenized(tokenized_ds):
    '''
    Remove columns that are not needed by huggingface Trainer
    '''
    for s in ['train', 'valid', 'test']:
        for c in ["__index_level_0__", "text", "label"]:
            if c in tokenized_ds.column_names[s]:
                tokenized_ds[s] = tokenized_ds[s].remove_columns([c])
    return tokenized_ds

