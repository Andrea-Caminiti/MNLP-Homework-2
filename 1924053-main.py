import torch
import torch.nn as nn
import numpy as np
import transformers 
import pandas as pd
from typing import Dict
import torch
from datasets import load_dataset, load_metric, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding
)
import argparse

def compute_metrics(eval_pred):
   '''Function to computer accuracy, f1-score, precision, and recall'''
   l_accuracy = load_metric('accuracy', trust_remote_code=True)
   l_f1 = load_metric('f1',trust_remote_code=True)
   l_precision = load_metric('precision', trust_remote_code=True)
   l_recall = load_metric('recall', trust_remote_code=True)


   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = l_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = l_f1.compute(predictions=predictions, references=labels, average='weighted')["f1"]
   precision = l_precision.compute(predictions=predictions, references=labels, average='weighted')["precision"]
   recall = l_recall.compute(predictions=predictions, references=labels, average='weighted')["recall"]

   return {"accuracy": accuracy, "f1": f1, 'precision': precision, 'recall': recall}

def tokenize_function(examples):
    str_to_int = {'ENTAILMENT': 0, 'NEUTRAL': 1, 'CONTRADICTION': 2}
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    texts = [premise + ' ' + tokenizer.sep_token + ' ' + hypothesis for premise, hypothesis in zip(examples["premise"],examples['hypothesis'])]
    tokens = tokenizer(texts, padding=True, truncation=True)
    tokens['label'] = [str_to_int[label] for label in examples["label"]]
    return tokens


def main(args):

    # Training Argurments
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0.001 
    epochs = 1
    device = "cuda" #if torch.cuda.is_available() else "cpu"
    set_seed(3233)
    
    # Initialize the model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.mode == 'train':
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                ignore_mismatched_sizes=True,
                                output_attentions=False, 
                                output_hidden_states=False,
                                num_labels=3)
        model.to(device)
        #Load dataset
        NLI_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")
        tokenized_datasets = NLI_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
                                        output_dir="training_dir",                    
                                        num_train_epochs=epochs,                      
                                        per_device_train_batch_size=batch_size,       
                                        warmup_steps=500,                             
                                        weight_decay=weight_decay,
                                        eval_strategy='epoch',                    
                                        save_strategy="epoch",
                                        save_total_limit=5,
                                        load_best_model_at_end=True,
                                        learning_rate=learning_rate,
                                        logging_strategy='epoch'               
                                    )
        
        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=tokenized_datasets["train"],
                            eval_dataset=tokenized_datasets["validation"],
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            compute_metrics=compute_metrics,
                            )
        
        training_out = trainer.train()
        evaluation_out = trainer.evaluate()

        with open(r'./training.txt', 'a') as f:
            f.write('\n ' + '#'*25 + ' TRAINING ' + '#'*25 )
            f.write(training_out)
            f.write('\n ' + '#'*25 + ' EVALUATION ' + '#'*25 )
            f.write(evaluation_out)
        
    elif args.mode == 'test' and args.data == 'original':
        #Load dataset
        NLI_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")
        #Launch test
        
    else:
        #Load dataset 
        NLI_dataset= load_dataset("iperbole/adversarial_fever_nli")
        #Launch test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train, test, and evaluate the model')
    # Add the positional argument 'mode' which can be either 'train' or 'test'
    parser.add_argument('mode', choices=['train', 'test'], help="The mode to 'train' or 'test' the model")

    # Add the argument '--data' which can be either 'original' or 'adversarial'
    parser.add_argument('--data', 
                        choices=['original', 'adversarial'], 
                        required=False, 
                        help="The type of test set to use: 'original'  or 'adversarial', for the original test set and the adversarial one respectly.")

    args = parser.parse_args()
    print(args)
    main(args)