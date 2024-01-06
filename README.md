# 02476 Hatespeech Classification

### Group 31
Group members:
Simon Daniel Eiriksson, s180722
Magnus Harder, s204117
Mattias Olofsson, s232407 
Amalie Roark, s184227 

## Project Discription
A Course in ML-Ops

### Overall goal
The overall goal of the project is to create a hate-speech classifier with a command line interface. The classifier will be based on a two stage deep learning architecture, where the first stage embeds raw text using some pre-trained text embedding model, and the second stage is a classification layer trained using hate-speech text data, that determines if the text is hatespeech. Ultimately, an end user should be able to provide text as a prompt as well as provide the classification results through the command line. Furthermore, the user should be enabled to alter between different underlying text embedding models along with the type of classification layer. 


### Frameworks
The models are build upon pretrained transformers existing in the huggingface framework, while pytorch-lightning enables training automatization through model deffinition.

- Transformers
- Pytorch_lightning


### Data
The Data consists of senteces 440907 senteces labled by wheater they are hate speech.
“A Curated Hate Speech Dataset”, Mendeley Data, V1 (https://data.mendeley.com/datasets/9sxpkmm8xn/1)

Devansh Mody, YiDong Huang, Thiago Eustaquio Alves de Oliveira,
A curated dataset for hate speech detection on social media text,
Data in Brief, Volume 46, 2023, 108832, ISSN 2352-3409,
https://www.sciencedirect.com/science/article/pii/S2352340922010356

### Models
- Sentence embedding models from hugginface
- Classificaation Layer
- Adaptivity of underlying LLM's



## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── 02476 Hatespeech Classification  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
