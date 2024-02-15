import argparse


def str2bool(v):
    '''
    To convert string to boolean value
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_model_name(args):
    '''
    Get model name without puntuations
    '''
    return args.model_id.replace("/","_").replace("-","_")

def get_standard_name(args):
    '''
    Get standard name for consistent naming
    '''
    return f'{get_model_name(args)}_{args.dataset_name}_few_shot_{args.nb_of_few_shot}'