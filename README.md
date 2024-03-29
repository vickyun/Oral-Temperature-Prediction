### Input Data
The input data is taken from Irvine Machine Learning Repository - [Infrared Thermography Temperature](https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset).

This dataset contains temperatures read from various locations of \
infrared face images of different patients, with the addition of \
oral temperatures measured for each individual. It contains 33 features \
such as gender, age, ethnicity, ambient temperature, humidity, distance, \
and other temperature readings from the thermal images.

To know about data acquisition see the [original paper](https://www.semanticscholar.org/paper/Infrared-Thermography-for-Measuring-Elevated-Body-Wang-Zhou/443b9932d295ca3a014e7d874b4bd77a33a276bd).

### Requirements
- python: 3.8
- pandas:  2.1.4
- mlflow: 2.10.2
 - ucimlrepo: 0.0.3
 - matplotlib: 3.8.3

Install dependencies: `pip install requirements.txt`

### Run the training script
Run train script: `python main.py`

### Parameters
Use utils.constant module to modify training parameters such as:
* feature set
* learning rate
* train/test split
* run name
* number of iterations
* random seed


### Exploratory Data Analysis 
EDA is available in Jupyter Notebook: /notebooks/Exploratory_Data_Analysis.ipynb

### Output 
