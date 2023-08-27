# Sentiment analysis with LSTM architecture - Pytorch

This project aims to build a Sentiment analysis model using the LSTM(Long-Short term memory) architecture.

## Project Structure

The project has the following structure:

- `Dataset`: This directory contains the dataset files used for training and evaluation.
- `model.py`: Every relevant piece of code is in this one file and to see the model in action you will run this python script.
- `requirements.txt`: requirements file to automate the process of installing the required dependencies.

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

Ensure that you have the necessary dependencies installed before running the project.

You may install the above dependencies simply by using:

    pip install -r requirements.txt

## Installation

- Open the terminal in your code editor and type this in

    `git clone https://github.com/GraphicsMonster/  LSTM-sentiment-analysis-model`

- To install the required dependencies, type this in

    `pip install -r requirements.txt`

- Once the dependencies are installed you are ready to train the model and evaluate its performance. If you have your own data to train the model on, you can update the code in the model.py to refer to the location of your dataset on your local machine. Be sure to update the preprocessing steps accordingly!!

- Run the model through the terminal

    `python model.py`


## Contributing

Contributions to this project are heavily encouraged! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Any kind of contribution will be appreciated.

## License

This project is licensed under the [MIT License](LICENSE).
