import json


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params} ")
    return trainable_model_params


def read_json_file(path) -> json:
    with open(path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    return json_data
