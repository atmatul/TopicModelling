# Topic Modelling

## Introduction
A basic implementation of LDA based
[Topic Modelling](https://ai.stanford.edu/~ang/papers/nips01-lda.pdf)
on the scientific publication data.

This project processes the __abstract__ field of the papers and
identifies the most used topics within them. The steps to build the
model is described in topicmodel.ipynb.

##Data
- Name: bibliography.bib
- Description: A collection of `1485` bibliography entries that describe the publication.

## Project Structure
- `data` data file

- `Plots` folder to save plots

- `LDA_Visualization.html` An interactive html file
                          that provides visualization to
                          LDA model.

- `topicmodel.ipynb` A jupyter notebook that describes the process
                    of building a topic model and visualize
                    the result.

- `utils.py` A python files that implements the utility function
             such as loading, parsing, cleaning, building bi-gram
             & tri-gram analysis and build word-cloud over the
             corpus.

- `requirements.txt` required packages

## How to run
Open Terminal and Navigate to Project Home Directory
1. `pip install -r requirements.txt`
2. `jupyter notebook`
3. Run `topicmodel.ipynb`

## How to change data
To run the model and prepare visualization on a separate data,
place the filename.bib file in the data folder and run the jupyter
notebook.

