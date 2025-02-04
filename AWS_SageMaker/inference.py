"""
In this file I will write the code that will be used to conduct inference on new data

The file will contain the following:
- Importing necessary libraries/modules
- Re-creating the model class
- Defining a function to initialize a new model, and loading the trained weights into it
- Defining a function to transform the new input data into a format that the model accepts
- Defining a function that does inference on the new data (after we transform it with the step above)
- Defining a function that transforms back the output of the inference into a format that is ready for the end-user
"""
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing libraries & modules
import os
import torch
import json

from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer

MAX_LEN = 512

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Re-create the model class - FT_DistilBERT() (FT stands for Fine-Tuned)
class FT_DistilBERT(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.block_1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.layer_2 = nn.Linear(in_features=768,
                                 out_features=768)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.classifier_layer = nn.Linear(in_features=768,
                                          out_features=num_classes)


    def forward(self, input_ids, mask_ids):
        # 1. Send through the DistilBERT pre-trained model
        output = self.block_1(input_ids = input_ids,
                              attention_mask = mask_ids)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        
        # 2. Send through the linear layer - this serves to increase the representational capacity of our model
        output = self.layer_2(pooler)
        # 3. Send through a non-linear activation function
        output = self.activation(output)
        # 4. Apply dropout to fight over-fitting
        output = self.dropout(output)
        # 5. Get the classification prediction (in logits)
        output = self.classifier_layer(output)

        return output



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define a function to initialize a new model from the FT_DistilBERT class, and load the fine-tuned weights into it
def model_fn(model_dir):

    # 1. Initialize a new model
    model = FT_DistilBERT(num_classes=4)

    # 2. Load in the fine-tuned model weights (the model that we saved into S3 after training)
    state_dict_location = os.path.join(model_dir, 'pytorch_distilbert_model_news.bin')
    model_state_dict = torch.load(state_dict_location, map_location= torch.device('cpu'))

    # 3. Apply the trained state_dict() to our newly initialized model
    model.load_state_dict(model_state_dict)

    return model



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define a function to transform raw inputs into a format that our model accepts
def input_fn(request_body, request_content_type):

    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        headline = input_data['inputs']
        return headline
    else:
        raise ValueError(f"Model does not support the specified content type: {request_content_type}")



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define a function to conduct inference

def predict_fn(input_data, model):

    # 0. Setting up some device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # 1. Tokenize the data
    data = tokenizer(text=input_data, return_tensors='pt').to(device) 

    ids = data['input_ids'].to(device)
    mask = data['attention_mask'].to(device)

    # 2. Run the model with the data
    model.eval()

    with torch.no_grad():
        logits = model(ids, mask)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        class_names = ['BUSINESS', 'ENTERTAINMENT', 'HEALTH', 'SCIENCE'] # checked the correct order in 3.Script.ipynb
        pred_class = probabilities.argmax(axis=1)[0].item()
        pred_label = class_names[pred_class]

        probabilities_dict = {class_names[i]: float(probabilities[0, i]) for i in range(len(class_names))}

    return {'predicted_label': pred_label}, probabilities_dict



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define a function to convert outputs into a json format
def output_fn(prediction_output, accept):
    if accept == "application/json":
        return json.dumps(prediction_output)
    else:
        raise ValueError(f"Unsupported content type: {accept}")