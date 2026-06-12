## Project Overview

This project implements a machine learning solution for automatically detecting sentiment in tweets directed at brands and products. The system classifies tweets as Positive, Negative, or Neutral to enable real-time brand monitoring and competitive intelligence.

## Business Problem

Companies need automated systems to monitor social media sentiment toward their products and competitors in real-time. This project builds a proof-of-concept classifier to determine whether a tweet expresses positive, negative, or neutral sentiment toward a brand or product.

## Business Value

Real-time brand monitoring: Automatically flag negative mentions as they occur

Competitive intelligence: Track sentiment for competitor products

Crisis detection: Early warning system for potential PR issues

Campaign measurement: Quantify sentiment impact of marketing efforts

## Dataset

The dataset contains approximately 9,000 tweets from SXSW 2011, labeled by human raters as:

Positive emotion (toward a brand/product) - 2,968 tweets

Negative emotion (toward a brand/product) - 569 tweets

No emotion (or neutral) - 5,372 tweets

## Approach

The project follows a structured approach:

Binary classification (Positive vs. Negative) as primary focus

Multiclass classification (Positive/Negative/Neutral) for comprehensive analysis

## Methodology

## Data Preprocessing

The following cleaning steps were applied to raw tweet text:

Conversion to lowercase

Removal of URLs, @mentions, and hashtag symbols

Removal of special characters, numbers, and punctuation

Tokenization and lemmatization

Stop word removal

Filtering words shorter than 3 characters

Feature Engineering

Two types of features were extracted:

TF-IDF Features (3000 features)

N-gram range: (1,2)

Minimum document frequency: 2

Maximum document frequency: 0.95

Sublinear TF scaling enabled

Handcrafted Features (8 features)

Tweet length

Word count

Positive word count (using sentiment lexicon)

Negative word count (using sentiment lexicon)

Exclamation mark count

Question mark count

Is retweet (binary)

Has hashtag (binary)

Total feature space: 3,008 features

### Models Evaluated

| Model | Binary Accuracy | Multiclass Accuracy |
| :--- | :---: | :---: |
| Random Forest | 0.8785 | 0.6906 |
| Multinomial Naive Bayes | 0.8644 | 0.7104 |
| Linear SVM | 0.8630 | 0.6773 |
| Bernoulli Naive Bayes | 0.8616 | N/A |
| Complement NB | 0.8602 | 0.6514 |
| Logistic Regression | 0.8573 | 0.7082 |

## Key Findings

Binary Classification (Positive vs. Negative)

Best Model: Random Forest (87.85% accuracy)

Runner-up: Multinomial Naive Bayes (86.44% accuracy)

Fastest: Multinomial Naive Bayes

Multiclass Classification (Positive/Negative/Neutral)

Best Model: Multinomial Naive Bayes (71.04% accuracy)

Runner-up: Logistic Regression (70.82% accuracy)

## Feature Importance Insights

The most predictive features for sentiment classification include:

Sentiment-bearing words (e.g., "love", "great", "awesome" for positive; "hate", "bad", "fail" for negative)

Tweet structure features (exclamation marks, question marks)

Product/brand mentions (iPad, Apple, iPhone, Google)

## Recommendations for Deployment

Primary Model: Use Multinomial Naive Bayes for highest overall accuracy across both binary and multiclass tasks

Fallback Model: Use Random Forest when maximum recall for negative sentiment detection is critical

Confidence Thresholding: Implement a confidence threshold (e.g., >0.7) for uncertain predictions to improve reliability

## Technical Requirements
Python 3.11+

Required libraries: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud

## Key Code Components

Data Loading & Cleaning (judge-1377884607_tweet_product_company.csv)

Text Preprocessing: Cleaning, tokenization, lemmatization

Feature Engineering: TF-IDF + handcrafted features

Model Training: 6 different classifiers with hyperparameter tuning

Evaluation: Accuracy, precision, recall, F1-score, confusion matrices

Visualization: Word clouds, feature importance, model comparison charts

## Results Visualization

The notebook includes:

Sentiment distribution charts (bar and pie)

Word clouds for each sentiment type

Product/brand mention analysis

Model accuracy comparison charts

Confusion matrices for top models

Feature importance bar charts for Logistic Regression

## Future Improvements

Test deep learning approaches (LSTM, BERT)

Implement ensemble methods combining multiple models

Add real-time streaming capabilities

Expand sentiment lexicon with domain-specific terms

Implement active learning for continuous model improvement
