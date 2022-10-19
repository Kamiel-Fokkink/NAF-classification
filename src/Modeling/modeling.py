from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

class auto_nlp_modeling():
    
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
        model=clf_model,                  # the instantiated :hugging_face: Transformers model to be trained
        args=training_args,               # training arguments, defined above
        train_dataset=train,      # training dataset
        eval_dataset=test,         # evaluation dataset
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
            texts, labels, test_size=.4, random_state=17
        )
    
        return train_texts, val_texts, train_labels, val_labels


    def fit(self):
        # tokenize
        tokenizer = AutoTokenizer.from_pretrained(self.model)

        # Split data into training and validation sets
        texts = self.data["ACTIVITE"].tolist()
        labels = self.data["encoded_label"].tolist()
        train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=.1, random_state=17)

        # Prepare train and val sets for the training
        train_encodings = tokenizer(train_texts, truncation=True, max_length=300,
                                padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, max_length=300,
                            padding=True)

        # tensor transformation
        train_dataset = ClassificationDataset(train_encodings, train_labels)
        val_dataset = ClassificationDataset(val_encodings, val_labels)

        # hyperparameter tuning and training
        if self.tuning == True:
            epochs, batch_size, warmup_steps, weight_decay = self.hyperparameter_tuning(epochs, batch_size, weight_decay, train=train_dataset, test=val_dataset)
            model = self.train_model(train=train_dataset, test=val_dataset)
        else: 
            model = self.train_model(train=train_dataset, test=val_dataset)

        return model


    def hyperparameter_tuning(self, train, test, epochs, batch_size, weight_decay):
        return epochs, batch_size, warmup_steps, weight_decay

    def predict(self, trained_model, df_test):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        # Split data into training and validation sets
        test_texts = df_test["text"].tolist()
        test_encodings = tokenizer(test_texts, truncation=True, max_length=300,
                                padding=True)
        # tensor transformation
        test_dataset = ClassificationTestDataset(test_encodings, len(test_texts))
        outputs = trained_model.predict(test_dataset, metric_key_prefix = "test")
        return outputs

    def evaluation(self, model):
        return


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


class ClassificationTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, length):
        self.encodings = encodings
        self.length = length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.length