from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

class auto_nlp_modeling(self):
    
    def __init__(self, data, nb_labels = 88, tuning = False):
        self.model = "camembert-base"
        self.nb_labels = nb_labels
        self.data = data
        self.tuning = tuning 

    def train_model(self, train, test, epochs = 5, batch_size = 16, weight_decay = 0.01):
        """ train model with fixed hyperparameters """

        clf_model = AutoModelForSequenceClassification.from_pretrained(self.model, num_labels=self.nb_labels)

        # Config the Trainer
        training_args = TrainingArguments(
        output_dir="./output_model",     # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=20,                # number of steps before to store training metrics
        evaluation_strategy="steps",     # strategy to compute the training metrics
        save_strategy="steps",           # should be the same as evaluation_strategy
        load_best_model_at_end=True,     # load the best model at the end of the training
        report_to="none",                # useful if used with mlflow for training reporting
        run_name="none",                 # name of the run to report to mlflow
        )

        # trainng setting
        trainer = Trainer(
        model=clf_model,                  # the instantiated 🤗 Transformers model to be trained
        args=training_args,               # training arguments, defined above
        train_dataset=train_dataset,      # training dataset
        eval_dataset=val_dataset,         # evaluation dataset
        #compute_metrics=compute_metrics,  # function to compute the metrics during the training
        )

        trainer.train()
        trainer.save_model(output_dir="./output_model")

        return trainer

    
    def train_val_split(self):
        """Train validation split on dataset"""

        texts = self.data["ACTIVITE"].tolist()
        labels = self.data["encoded_label"].tolist()

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=.4, random_state=17, stratify=labels
        )
    
        return train_texts, val_texts, train_labels, val_labels


    def fit(self):
        # tokenize
        tokenizer = AutoTokenizer.from_pretrained(self.model)

        # Split data into training and validation sets
        texts = self.data["ACTIVITE"].tolist()
        labels = self.data["encoded_label"].tolist()
        train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=.1, random_state=17, stratify=labels)

        # Prepare train and val sets for the training
        train_encodings = tokenizer(train_texts, truncation=True, max_length=300,
                                padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, max_length=300,
                            padding=True)

        # tensor transformation
        train_dataset = ClassificationDataset(train_encodings, train_labels)
        val_dataset = ClassificationDataset(val_encodings, val_labels)

        # hyperparameter tuning and training
        if self.tuning = True:
            epochs, batch_size, warmup_steps, weight_decay = self.hyperparameter_tuning(train=train_dataset, test=val_dataset, epochs, batch_size, weight_decay)
            model = self.train_model(train=train_dataset, test=val_dataset, epochs, batch_size, weight_decay)
        else: 
            model = self.train_model(train=train_dataset, test=val_dataset, epochs, batch_size, weight_decay)

        return model


""" TO DO """

    def hyperparameter_tuning(self, train, test, epochs, batch_size, weight_decay):
        """ call the train_model for each combination of parameters and return the best params """

        return epochs, batch_size, warmup_steps, weight_decay

    def predict(self, test):
        """ prediction of data"""
        pred = model.predict(test_dataset=test, metric_key_prefix="val")


    def evaluation(self, model):
        """ model evaluation, should be used in hyperparameter tuning"""





class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)