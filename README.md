# Fake News Detection

This project aims to detect fake news using Big Data technologies. We use various machine learning models and natural language processing techniques to analyze and classify news articles.

## Table of Contents

- [Fake News Detection](#fake-news-detection)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Technologies Used](#technologies-used)
  - [Data Collection](#data-collection)
  - [Data Processing](#data-processing)
  - [Model Training](#model-training)
  - [Results](#results)
  - [Conclusion](#conclusion)
- [Running the Code](#running-the-code)

## Introduction

Fake news is a significant issue in today's digital age. This project aims to tackle this problem using Big Data technologies and machine learning.

## Technologies Used

We use a combination of Python, Apache Kafka, and PySpark for data processing and model training.

## Data Collection

We collected our data from various online sources. The data consists of news articles labeled as 'true' or 'fake'. You can find the datasets in the [datasets](BigData-FakeNewDT/datasets) directory.

## Data Processing

We use Apache Kafka for real-time data streaming and PySpark for data processing. You can find the code for this in the [kafka](BigData-FakeNewDT/kafka) directory.

## Model Training

We trained our model using PyTorch. The trained model can be found in the [model](BigData-FakeNewDT/model) directory.

## Results

Our model achieved an accuracy of 91.78% on the test dataset.

## Conclusion

This project demonstrates the potential of using Big Data technologies in tackling the issue of fake news. Future work could involve improving the model's accuracy and using more diverse data sources.

# Running the Code

Follow these steps to run the code:

1. **Navigate to the project directory:**

```sh
cd BigData-FakeNewDT
```

2. **Activate the virtual environment:**
```sh
source venv/bin/activate
```
3. **Install the required dependencies:**
```sh
pip install -r requirements.txt
```
4. **Run the application:**
```sh
python application.py
```