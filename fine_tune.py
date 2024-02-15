from transformers import BartModel, BartTokenizer
import pickle
import argparse
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForCausalLM, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import variables as V
import utils
import data_helper as dh


MODEL = None
TOKENIZER = None
DF_TRAIN, DF_TEST, DF_VAL = None, None, None
LOAD_WITH_LOWER_PRECISION = None



def preprocess_function(examples, padding='max_length'):

    inputs = examples['text']
    model_inputs = TOKENIZER(inputs, max_length=V.MAX_INPUT_LENGTH, padding=padding, truncation=True)

    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(examples["label"], max_length=V.MAX_TARGET_LENGTH, padding=padding, truncation=True)
        if padding == "max_length":
            labels ["input_ids"] = [
                [(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels['input_ids']
            ]

    model_inputs['labels'] = labels["input_ids"]
    return model_inputs


def get_config_for_lower_precision(args):

    target_modules = V.TARGET_SEQ_2_SEQ_MODULES[args.model_id]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    return bnb_config, config


def init_model_and_tokenizer(args):
    global MODEL, TOKENIZER

    if LOAD_WITH_LOWER_PRECISION:
        bnb_config, config = get_config_for_lower_precision(args)
        MODEL = AutoModelForSeq2SeqLM.from_pretrained(args.model_id,
                                                      quantization_config=bnb_config,
                                                      device_map={"": 2})

        MODEL.gradient_checkpointing_enable()
        MODEL = prepare_model_for_kbit_training(MODEL)
        MODEL = get_peft_model(MODEL, config)
        utils.print_trainable_parameters(MODEL)
    else:
        MODEL = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    TOKENIZER_CLS_NAME = BartTokenizer if 'facebook/bart' in args.model_id else AutoTokenizer
    TOKENIZER = TOKENIZER_CLS_NAME.from_pretrained(args.model_id)


def init_data(args):
    '''
    Initialize the datasets
    '''
    global DF_TRAIN, DF_TEST, DF_VAL

    DF_TRAIN = dh.read_data(V.TRAIN_FILE_LOCATION.format(dataset_name=args.dataset_name, nb_of_few_shot=args.nb_of_few_shot))
    DF_TEST = dh.read_data(V.TEST_FILE_LOCATION.format(dataset_name=args.dataset_name))
    DF_VAL = dh.read_data(V.VALIDATION_FILE_LOCATION.format(dataset_name=args.dataset_name))

    if args.trim:
        DF_TRAIN = DF_TRAIN.sample(200) if len(DF_TRAIN) > 200 else DF_TRAIN
        DF_TEST = DF_TEST.sample(50) if len(DF_TEST) > 500 else DF_TEST
        DF_VAL = DF_VAL.sample(50) if len(DF_VAL) > 500 else DF_VAL

    DF_TRAIN, DF_TEST, DF_VAL = dh.add_columns(DF_TRAIN, DF_TEST, DF_VAL, args)

    assert len(DF_TRAIN.consistent.unique().tolist()) == 2


def compute_metrics(eval_preds):
    '''
    compute metrics
    '''
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    decoded_labels = [1 if l == '(a)' else 0 for l in decoded_labels ]
    decoded_preds = [1 if l == '(a)' else 0 for l in decoded_preds ]

    result = {'precision' : precision_score(decoded_labels, decoded_preds)}
    result['recall'] = recall_score(decoded_labels, decoded_preds)
    result['f1'] = f1_score(decoded_labels, decoded_preds)

    return result


def decode_predictions(eval_preds):
    '''
    Decode predictions and compute score
    '''
    preds, labels, metric = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    decoded_labels_bool = [1 if l == '(a)' else 0 for l in decoded_labels]
    decoded_preds_bool = [1 if l == '(a)' else 0 for l in decoded_preds]

    return {'decoded_labels_bool': decoded_labels_bool,
            'decoded_preds_bool': decoded_preds_bool,
            'decoded_labels': decoded_labels,
            'decoded_preds': decoded_preds,
            'metric': metric
            }


def get_training_arguments(args, stnd_name):
    return Seq2SeqTrainingArguments(
        f'./saved_model/{stnd_name}-finetuned-kgcleaner-v2',
        evaluation_strategy="steps",
        eval_steps=250,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.00,
        save_total_limit=3,
        num_train_epochs=1 if args.trim else 3,
        predict_with_generate=True,
        fp16=True if LOAD_WITH_LOWER_PRECISION else False,
        optim="paged_adamw_8bit" if LOAD_WITH_LOWER_PRECISION else "adamw_torch"
    )


def train(args):

    dataset_dict = dh.df2dataset_dict(DF_TRAIN, DF_TEST, DF_VAL)
    tokenized_ds = dataset_dict.map(preprocess_function, batched=True)
    tokenized_ds = dh.remove_columns_tokenized(tokenized_ds)

    #Seq2SeqTrainingArguments
    stnd_name = utils.get_standard_name(args)
    training_args = get_training_arguments(args, stnd_name)

    data_collator = DataCollatorForSeq2Seq(TOKENIZER, model=MODEL)
    trainer = Seq2SeqTrainer(
        MODEL,
        training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['valid'],
        data_collator=data_collator,
        tokenizer=TOKENIZER,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    predictions = trainer.predict(test_dataset=tokenized_ds['test'])
    predictions = decode_predictions(predictions)
    return predictions


def set_load_with_precision_flag(args):
    global LOAD_WITH_LOWER_PRECISION

    LOAD_WITH_LOWER_PRECISION = True if args.model_id in V.TARGET_SEQ_2_SEQ_MODULES else False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='kg_llm_fine_tune')

    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--dataset_name', type=str, default='none')
    parser.add_argument('--trim', type=utils.str2bool, default=False)
    parser.add_argument('--nb_of_few_shot', type=int, default=5)
    parser.add_argument('--log', type=utils.str2bool, default=True)

    args = parser.parse_args()
    model_name = utils.get_model_name(args)
    stnd_name = utils.get_standard_name(args)

    # CREATING LOG FILE TO LOG PROGRESS
    print('writing in ', f'{stnd_name}_v2_no_lora.txt')
    if args.log:
        sys.stdout = sys.stderr = open(f'{stnd_name}_v2_no_lora.txt', 'w')
    print(args)

    '''
    Determine if model can/cannot fit in the GPU memory
    Initilize the Train, Validation and Test dataset
    '''

    set_load_with_precision_flag(args)
    init_data(args)
    sys.stdout.flush()

    '''
    Initilize model and tokenizer to train the model 
    '''
    init_model_and_tokenizer(args)
    predictions = train(args)
    pickle.dump(predictions,
                open(V.PREDICTION_FILE_LOCATION.format(stnd_name=stnd_name), 'wb'))
    sys.stdout.flush()



