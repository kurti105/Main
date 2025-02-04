# In this `script.py` file we will do the following:
# 1. Import necessary libraries & get data
# 2. Encode the categories
# 3. Prepare the Tokenization for the input text (i.e., the news headlines in our case)
# 4. Create the PyTorch Dataset class - this is where we will tokenize the text data
# 5. Split the data into training & testing sets
# 6. Create the model class
# 7. Create training loop
# 8. Set up the `main()` function which will start everything this will contain the following:
    # 8.1 Setup device-agnostic code
    # 8.2 Setup argument parser through which we pass our hyperparameters
    # 8.3 Create the training/testing PyTorch DataLoaders
    # 8.4 Initialize model
    # 8.5 Select loss function, optimizer, accuracy metrics
    # 8.6 Start training
    # 8.7 Creating the output directory and saving the training results there (i.e., model weights & vocabulary)





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Install & import libraries, modules & get data
# In a Python script, the packages are installed by looking into the requirements.txt file - I have added s3fs and torchinfo there

# Import libraries and modules
import torch
import transformers
import pandas as pd
import numpy as np
import os
import argparse
import torchmetrics
import s3fs
import sklearn


from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, Precision, Recall

# Initialize the S3 filesystem
fs = s3fs.S3FileSystem(anon=False)  # Use anon=True for public buckets

# Let's get data
s3_path = 's3://tk5-huggingface-multiclass-textclassification-bucket/training_data/newsCorpora.csv'

# Read the CSV using s3fs
with fs.open(s3_path) as f:
    df = pd.read_csv(f, 
                     sep='\t', 
                     names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])



# Let's update the CATEGORY variable as we did before
# Create the mapping through which we will update the CATEGORY variable
my_dict = {
    'b': 'BUSINESS',
    't': 'SCIENCE',
    'e': 'ENTERTAINMENT',
    'm': 'HEALTH'
}

# Create helper function
def update_category(x, dictionary: dict):
    return dictionary.get(x)

# Update the CATEGORY variable & keep only the necessary columns
df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_category(x, dictionary=my_dict))
df = df[['TITLE', 'CATEGORY']]



# We will initially train the model on a small fraction of the data to make sure everything is running as intended - I don't want to figure out something is wrong after running training on a GPU instance
# for many hours 
df = df.sample(frac=0.10, random_state=1) # selecting only 10% of the data
df = df.reset_index(drop=True)
print(f"Count by category: {df.groupby(['CATEGORY']).count()}")





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2: Encode the Categorical output variable
df['ENCODE_CAT'] = df.groupby(by=['CATEGORY']).ngroup() # the .ngroup() assigns a number to each unique category

# To get the categories and category_to_id dictionary:
small_sample_df = df.drop_duplicates(subset=['CATEGORY', 'ENCODE_CAT'])
small_sample_df = small_sample_df.sort_values(by=['ENCODE_CAT']).reset_index(drop=True)
categories = list(small_sample_df['CATEGORY'])
encoded_cats = list(small_sample_df['ENCODE_CAT'])

class_to_idx = {}
i = 0
for category in categories:
    class_to_idx[f'{category}'] = encoded_cats[i]
    i += 1





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 3: Prepare the text tokenization
from transformers import DistilBertTokenizer

# Get the tokenizer of choice
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 4: Create the PyTorch Dataset class
class NewsDataset(Dataset):

    # Define the __init__() method:
    def __init__(self, data, tokenizer, max_length):
        super().__init__()
        
        # 1. Initialize the data, tokenizer & allocated maximum length for the model inputs (remember, our task deals with news headlines, so will choose max_length accordingly)
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.len = len(data)

    # Define the __getitem__ method:
    def __getitem__(self, index):
        # 1. Get headline from the source dataframe
        # headline = self.data.TITLE[index]
        headline = str(self.data.iloc[index, 0]) # more efficient than the line above
        headline = " ".join(headline.split())

        # 2. Tokenize the headline
        headline_tokenized = self.tokenizer.encode_plus(
            
            headline, # input text
            add_special_tokens = True,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_token_type_ids = True,
        
        )

        ids = headline_tokenized['input_ids']
        mask = headline_tokenized['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index, 2], dtype=torch.long)
        }

    # Define the __len__() method:
    def __len__(self):
        return self.len





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 5: Split data into training/testing sets
import sklearn
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df['CATEGORY'])
print(f"Training data has shape: {train_df.shape}")
print(f"Testing data has shape: {test_df.shape}")





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 6: Create the model class
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





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 7: Create functions for the Training Loop

# 1. Train step
def train_step(model: torch.nn.Module, 
               train_dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim, 
               metrics: Dict[str, torchmetrics.Metric], 
               device: torch.device):
    """
    This function conducts one training loop across all batches in a DataLoader

    Parameters:
        - model: a PyTorch model architecture that will be trained
        - train_dataloader: a PyTorch DataLoader that contains the training batches
        - loss_fn: a PyTorch loss function through which we measure how much the model errors
        - optimizer: a PyTorch optimizer that determines how we adjust the weights of the model
        - metrics: a list of PyTorch (torchmetrics) metrics that we want to review as the model progresses through training
        - device: the target device in which we will train the model (e.g., CPU, GPU)
    """
    # 0. Set up variables that will contain the training loss, and accuracy metric
    total_train_loss = 0
    metric_totals = {name: 0.0 for name in metrics.keys()}
    
    ## I also want to keep a record of how the loss & accuracy metrics develop per each batch in an epoch - in case we want to plot the progress
    results_dict = {
        'batch_train_loss': [],
        'batch_train_accuracy': [],
        'batch_train_precision': [],
        'batch_train_recall': []
    } 
    
    
    # 1. Set the model in training mode
    model.train()

    # 2. Send model & metrics to target device, and reset them
    model.to(device)
    for metric in metrics.values():
        metric.to(device)
        metric.reset()

    # 3. Start the training step
    for idx, data in enumerate(train_dataloader):
        
        # 3.1 Get data into target device
        inputs = data['ids'].to(device)
        mask = data['mask'].to(device)
        targets = data['targets'].to(device)

        # 3.2 Run model on data
        outputs = model(input_ids=inputs,
                        mask_ids=mask)
        probabilities = torch.softmax(outputs,
                                      dim=1)
        predictions = torch.argmax(probabilities,
                                   dim=1)

        # 3.3 Calculate the loss & accuracy metrics
        ## Loss
        loss = loss_fn(outputs, targets)
        total_train_loss += loss
        results_dict['batch_train_loss'].append(loss.item()) 

        ## Metrics
        for name, metric in metrics.items():
            value = metric(predictions, targets)
            metric_totals[name] += value.item()
            results_dict[f'batch_train_{name}'].append(value.item())
            

        
        # 3.4 Optimizer zero grad
        optimizer.zero_grad()

        # 3.5 Do backpropagation
        loss.backward()

        # 3.6 Apply optimizer step
        optimizer.step()


        # Print out what's happening every 100th batch - we get the loss & metrics results per batch by dividing by the number of batches:
        if idx % 100 == 0:
            progress_metrics = {
                name: total / (idx + 1) 
                for name, total in metric_totals.items()
            }
            current_loss = total_train_loss / (idx + 1)
            
            print(f"Batch {idx}")
            print(f"Training loss: {current_loss:.3f}")
            for name, value in progress_metrics.items():
                print(f"Train {name}: {value:.3f}")
            print()

    # 4. Get the final train loss/accuracy/precision/recall per batch
    num_batches = len(train_dataloader)
    final_metrics = {
        name: total / num_batches 
        for name, total in metric_totals.items()
    }
    final_loss = total_train_loss / num_batches
    
    return final_loss, final_metrics, results_dict



# 2. Test step
def test_step(model, 
              test_dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              metrics: Dict[str, torchmetrics.Metric], 
              device: torch.device):
    """
    This function conducts one validation loop across all batches in a DataLoader

    Parameters:
        - model: a PyTorch model architecture that will be trained
        - test_dataloader: a PyTorch DataLoader that contains the training batches
        - loss_fn: a PyTorch loss function through which we measure how much the model errors
        - metrics: a list of PyTorch (torchmetrics) metrics that we want to review as the model progresses through training
        - device: the target device in which we will train the model (e.g., CPU, GPU)
    """
    # 0. Set up variables that will contain the training loss, and accuracy metric
    total_test_loss = 0.0
    metric_totals = {name: 0.0 for name in metrics.keys()}
    
    ## To keep a record of how the loss & accuracy metrics develop per each batch in an epoch
    results_dict = {
        'batch_test_loss': [],
        'batch_test_accuracy': [],
        'batch_test_precision': [],
        'batch_test_recall': []
    } 
    
    # 1. Set the model in evaluation mode
    model.eval()

    # 2. Send model & metrics to target device (and reset the metrics)
    model.to(device)
    for metric in metrics.values():
        metric.to(device)
        metric.reset()
    
    # 3. Start validation loop
    
    ## 3.1 Set model in inference mode
    with torch.inference_mode():

        for idx, data in enumerate(test_dataloader):
            
            ## 3.2 Get data into target device
            inputs = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['targets'].to(device)

            ## 3.3 Get predictions
            outputs = model(input_ids=inputs,
                            mask_ids=mask)
            probabilities = torch.softmax(outputs,
                                          dim=1)
            predictions = torch.argmax(probabilities,
                                       dim=1)

            ## 3.4 Estimate loss & metrics
            ### Loss
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            results_dict['batch_test_loss'].append(loss.item()) 
    
            ### Metrics
            for name, metric in metrics.items():
                value = metric(predictions, targets)
                metric_totals[name] += value.item()
                results_dict[f'batch_test_{name}'].append(value.item())

            # Print out what's happening every 100th batch - we get the loss & metrics results per batch by dividing by the number of batches:
            if idx % 100 == 0:
                progress_metrics = {
                    name: total / (idx + 1) 
                    for name, total in metric_totals.items()
                }
                current_loss = total_test_loss / (idx + 1)
                
                print(f"Batch {idx}")
                print(f"Test loss: {current_loss:.3f}")
                for name, value in progress_metrics.items():
                    if name == 'acc':
                        print(f"Test accuracy: {value*100:.2f}%")
                    else:
                        print(f"Test {name}: {value:.3f}")
                print()

    # 4. Get final test loss/accuracy/precision/recall per batch
    num_batches = len(test_dataloader)
    final_metrics = {
        name: total / num_batches 
        for name, total in metric_totals.items()
    }
    final_loss = total_test_loss / num_batches

    return final_loss, final_metrics, results_dict



# 3. Consolidated Training Loop
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          epochs: int, 
          loss_fn: torch.nn.Module, 
          optimizer: torch.optim, 
          metrics: Dict[str, torchmetrics.Metric],
          device: torch.device):
    """
    This function combines the train_step() and test_step() functions we created above, to provide a consolidated training loop that includes both training & validation for a 
    specified number of epochs.

    Arguments:
        - model: a PyTorch model architecture that will be trained
        - train_dataloader: a PyTorch DataLoader that contains the training batches
        - test_dataloader: a PyTorch DataLoader that contains the testing batches
        - epochs: the number of epochs for which we will train the model (i.e., how many full iterations through the train & test DataLoaders)
        - loss_fn: a PyTorch loss function through which we measure how much the model errors
        - optimizer: a PyTorch optimizer that determines how we adjust the weights of the model
        - metrics: a list of PyTorch (torchmetrics) metrics that we want to review as the model progresses through training
        - device: the target device in which we will train the model (e.g., CPU, GPU)
    """
    consolidated_results_dict = {
        'training_history': [],
        'testing_history': [],
        'batch_history': {}
    }

    # Get start time for model - want to check the time it takes to train, although SageMaker measures it as well (just for comparison)
    train_start_time = timer()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training loop
        train_loss, train_metrics, train_batch_results = train_step(model=model,
                                                                    train_dataloader=train_dataloader,
                                                                    loss_fn=loss_fn,
                                                                    optimizer=optimizer,
                                                                    metrics=metrics,
                                                                    device=device)


        # Testing loop
        test_loss, test_metrics, test_batch_results = test_step(model=model,
                                                                test_dataloader=test_dataloader,
                                                                loss_fn=loss_fn,
                                                                metrics=metrics,
                                                                device=device)
        
        # Print out some results
        print(f"Epoch {epoch+1} ends - here are the results:")
        print(f"Train Loss: {train_loss:.3f}\nTrain Metrics: {train_metrics}")
        print(f"Test Loss: {test_loss:.3f}\nTest Metrics: {test_metrics}")
        print(f"-"*100)

    # Get end time for model
    train_end_time = timer()

    print(f"Time to train model: {train_end_time - train_start_time:.2f}/60 minutes")





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 8: Setup the argument parser
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Add arguments that match your estimator hyperparameters
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    
    args, _ = parser.parse_known_args()
    return args





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Step 9: Create the main() function in which we prepare the training/testing dataloaders, initialize the model, select loss function, optimizer, and accuracy metrics, define the output directory
# and save model outputs
def main():
    print("Start Training")

    # 1. Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    # 2. Create argument parser that will provide the hyperparameters to the model
    args = parse_args()
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Length: {args.max_length}")
    print(f"Train Batch Size: {args.train_batch_size}")
    print(f"Test Batch Size: {args.test_batch_size}")

    
    MAX_LENGTH = args.max_length
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    
    # 3. Get tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    
    # 4. Get train/test Datasets and DataLoaders
    train_dataset = NewsDataset(data=train_df, tokenizer=tokenizer, max_length=MAX_LENGTH)
    test_dataset = NewsDataset(data=test_df, tokenizer=tokenizer, max_length=MAX_LENGTH)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)

    
    # 5. Initialize a model instance
    print("Initializing model...")
    model = FT_DistilBERT(num_classes=4)
    model.to(device)

    
    # 6. Choose loss function, optimizer & accuracy metrics
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=LEARNING_RATE)
    metrics = {
    'accuracy': Accuracy(task='multiclass', num_classes=10),
    'precision': Precision(task='multiclass', num_classes=10),
    'recall': Recall(task='multiclass', num_classes=10)
    }

    
    # 7. Train model
    print("Starting training...")
    train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          epochs=NUM_EPOCHS,
          loss_fn=loss_fn,
          optimizer=optimizer,
          metrics=metrics,
          device=device)

    
    # 7. Specify output directory
    output_dir = os.environ['SM_MODEL_DIR']
    print(f"Output directory: {output_dir}")

    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_model_news.bin')
    output_vocab_file = os.path.join(output_dir, 'vocab_distilbert_news.bin')

    
    # 8. Save model weights & vocabulary
    torch.save(obj=model.state_dict(),
               f=output_model_file)
    tokenizer.save_vocabulary(save_directory=output_vocab_file)



if __name__ == '__main__':
    main()