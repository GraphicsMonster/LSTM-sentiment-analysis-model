# Sentiment analysis with LSTM architecture - Pytorch

This project aims to build a Sentiment analysis model using the LSTM(Long-Short term memory) architecture.

## Project Structure

The project has the following structure:

- `Dataset`: This directory contains the dataset files used for training and evaluation.
- `model.py`: This file contains the relevant piece of code required to run the model for inference after training.
- `train.py`: You train the modle by running this script. If you make any hyperparam changes in the model.py file make sure to make those changes here as well.
- `requirements.txt`: requirements file to automate the process of installing the required dependencies.
- `model_test.py`: This is the script you'll run to test the model on your own text data.

## Dependencies

The project requires the following dependencies:

- Python 3.9 or higher
- numpy
- pandas
- scikit-learn
- tensorflow
- keras
- torch
- torchtext
- tweet-preprocessor
- pickle

Ensure that you have the necessary dependencies installed before running the project.

You may install the above dependencies simply by using:

    pip install -r requirements.txt

## Installation

- Open the terminal in your code editor and type this in

    `git clone https://github.com/GraphicsMonster/LSTM-sentiment-analysis-model`

- To install the required dependencies, type this in

    `pip install -r requirements.txt`

- Once the dependencies are installed you are ready to train the model and evaluate its performance. If you have your own data to train the model on, you can update the code in the model.py to refer to the location of your dataset on your local machine. Be sure to update the preprocessing steps accordingly!!

- Train the model run this command in the terminal

    `python train.py`

- Once you've successfully trained the model, it will automatically be saved in the same directory with the name `model.pt`

- Test the model on your own text data

    `python model_test.py`

## Contributing

Contributions to this project are heavily encouraged! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Any kind of contribution will be appreciated.

## License

This project is licensed under the [MIT License](LICENSE).
