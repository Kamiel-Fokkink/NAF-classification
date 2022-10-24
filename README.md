# NAF-classification
Classify the NAF (Nomenclature d’activités française) label of a company based on newspaper articles and company descriptions. Project made as part of a course for X-HEC Data Science for Business, in collaboration with Quinten. 7 collaborators contributed to completion of this project. The repository is organised as follows:

### Inputs
Contains raw data used to train the model with. The main data file, NAF activite, has not been included, as it contains 1.2 million records, and is known to evaluators of the project. Another external data file that was collected during the process is included here. It includes for every NAF category and subcategory, a set of keywords related to activities in that sector, which provides an additional source of data that enhances model performance.

### Outputs
Contains all final results produced by this project. One component is preprocessed data, obtained by running several preprocessing and augmentation steps on raw data. These files are not included, as the results can be obtained by locally executing the preprocessing script. Another component is model, which includes the trained model used for the final predictions. This file has been included in the repository.

### Params
Contains several parameters related to optimising the workflow. One is the config file, which contains several global constants, such as file names and internal paths. Another is the log file, which contains a history of all runs of the project, useful for debugging and seeing what goes on in the code.

### Src
All code produced by various collaborators is included in this directory. Organised into several subdirectories, each of which is imported as python modules. Preprocessing: cleaning useless data records, transforming data format into one that is suitable for model training, combining data from different sources. Collection: collecting data from an external pdf source, preprocessing it into the desired format. Augmentation: performs data augmentation techniques, to balance all classes and increase the number of records in undersampled classes. Employs backtranslation and random swaps. Modelling: trains a transfer model (Camembert) on this specific classification problem, uses the new model to make predictions.

All these components of the project are tied together in the main.py file in the src directory. It can be run from the command line, with arguments to indicate which parts of the process to execute (e.g. only data collection and preprocessing). Requirements.txt includes all packages that are required for executing the code.