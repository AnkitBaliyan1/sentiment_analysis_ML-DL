# IMDb Sentiment Analysis Project

## Overview

This project focuses on performing sentiment analysis on a dataset of 50,000 movie reviews from IMDb. The objective is to classify each review as either "positive" or "negative" based on its content. Various classification algorithms and deep learning models were employed to achieve this task, and their performance was evaluated using precision and recall metrics.

## Dataset

The dataset used for this project consists of two columns:

1. **Review**: This column contains the text comments provided by users, representing movie reviews.
2. **Sentiment**: This column indicates whether each review is categorized as "positive" or "negative."

## Project Structure

The project is structured as follows:

1. **Data Preprocessing**: The text data undergoes cleaning and preprocessing, including tasks such as stopwords removal, tokenization, and text vectorization.

2. **Exploratory Data Analysis (EDA)**: EDA is conducted to gain insights into the dataset's characteristics. This step helps in understanding the distribution of positive and negative reviews.

3. **Model Selection**: Various machine learning and deep learning models are explored for sentiment classification. Some of the models included Logistic Regression, Random Forest, K-Nearest Neighbors, Multinomial Naive Bayes, and Convolutional Neural Networks (CNN).

4. **Model Training and Evaluation**: The dataset is split into training and testing sets, and the selected models are trained. Their performance is assessed using precision and recall metrics.

## Performance Summary

The performance summary table for the models used in this project is as follows:

| Model                   | Precision | Recall   |
|-------------------------|-----------|----------|
| Logistic Regression     | 0.870     | 0.879    |
| Random Forest           | 0.845     | 0.841    |
| K-Nearest Neighbors     | 0.681     | 0.551    |
| Multinomial Naive Bayes | 0.863     | 0.838    |
| CNN                     | 0.865     | 0.862    |

## Getting Started

To reproduce the results of this project or use the trained model for your own sentiment analysis tasks, follow these steps:

1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Ensure you have all the necessary libraries and packages installed by using the `requirements.txt` file.

3. **Explore the Notebooks**: Detailed explanations of data preprocessing, EDA, model training, and evaluation steps are provided in the Jupyter notebooks.

4. **Utilize the Trained Model**: To use the trained model for sentiment analysis, consult the provided code examples or scripts in the "Usage" section of the repository.

## Future Improvements

Here are potential areas for future improvement:

- **Hyperparameter Tuning**: Fine-tuning model hyperparameters may enhance performance further.
- **Data Augmentation**: Expanding the dataset or using data augmentation techniques could improve model robustness.
- **Advanced Deep Learning Architectures**: Experimenting with more complex neural network architectures like LSTM or Transformer models may lead to better results.
- **Sentiment Analysis for Specific Genres**: Customizing the model for specific movie genres could increase sentiment analysis accuracy.

## Conclusion

This IMDb sentiment analysis project demonstrates the application of various machine learning and deep learning techniques to classify movie reviews as positive or negative. The performance summary table provides insights into the effectiveness of different models, and the code and instructions allow for easy replication and usage of the project.

If you have any questions or suggestions for improvement, please don't hesitate to reach out to me at [a.baliyan008@gmail.com](a.baliyan008@gmail.com).
