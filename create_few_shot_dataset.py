import pandas as pd
import argparse
import data_helper as dh


def sample_examples(df, args):
    if len(df) > args.number_of_few_shots:
        df_sample = df.sample(args.number_of_few_shots, random_state=123)
    else:
        df_sample = df
    return df_sample

def create_few_shot_train_dataset(df_train, args):

    dfs = []

    unique_relations = df_train.relation.unique().tolist()
    for relation in unique_relations:
        dff = df_train[df_train.relation == relation]
        dff_true = dff[dff.consistent==True]
        dff_false = dff[dff.consistent==False]

        df_true_sample = sample_examples(dff_true, args)
        df_false_sample = sample_examples(dff_false, args)

        df_final = pd.concat([df_true_sample, df_false_sample])
        dfs.append(df_final)

    return pd.concat(dfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='create_few_shot_dataset')

    parser.add_argument('--dataset_name', type=str, default='tac_2017')
    parser.add_argument('--number_of_few_shots', type=int, default=5)

    args = parser.parse_args()


    DF_TRAIN = pd.read_pickle(f'train_{args.dataset_name}.pickle')
    DF_TEST = pd.read_pickle(f'test_{args.dataset_name}.pickle')
    DF_VAL = pd.read_pickle(f'valid_{args.dataset_name}.pickle')

    DF_TRAIN = pd.DataFrame(DF_TRAIN)
    DF_TEST = pd.DataFrame(DF_TEST)
    DF_VAL = pd.DataFrame(DF_VAL)

    DF_TRAIN, DF_TEST, DF_VAL = dh.add_columns(DF_TRAIN, DF_TEST, DF_VAL, args)

    print(DF_TRAIN.consistent.value_counts())

    DF_TRAIN_FEW_SHOT = create_few_shot_train_dataset(DF_TRAIN, args)

    print(DF_TRAIN_FEW_SHOT.shape
          )
    print(DF_TRAIN_FEW_SHOT.consistent.value_counts())

    DF_TRAIN_FEW_SHOT.to_pickle(f'train_{args.dataset_name}_few_shot_{args.number_of_few_shots}_v2.pickle')



