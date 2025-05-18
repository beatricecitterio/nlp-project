# NLP Political Tweet Analysis

This repository provides a comprehensive pipeline for the analysis, classification, and generation of political tweets in Italian. It combines data collection, preprocessing, topic modeling, sentiment analysis, and fine-tuning of language models to study political discourse on social media.

This was the project for the course "Language Technology" at Bocconi University by:
- Beatrice Citterio
- Filippo Focaccia
- Giulio Pirotta
- Martina Serandrei
- Tommaso Vezzoli


## Project Overview

The project is organized around the following core tasks:

- **Data Collection:** Scraping and curating political tweets from Italian politicians and parties.
- **Preprocessing:** Cleaning and standardizing the tweet dataset for downstream tasks.
- **Topic Modeling & Classification:** Assigning macro and micro topics to tweets using LDA and supervised models.
- **Active Learning:** Iteratively labeling data to improve model performance with minimal annotation effort.
- **Sentiment Analysis:** Applying multilingual and Italian-specific transformer models for sentiment and emotion detection.
- **Dataset Creation:** Building and exporting a dataset for fine-tuning transformer models.
- **Model Fine-Tuning:** Training and evaluating transformer-based models (Minerva models) for generation tasks.
- **Evaluation:** Assessing the fine-tuned model performance and compare it to the baseline.


## Repository Structure
    .
    ├── files/                              # Data files (baselines, finetuned, train)
    ├── fine_tuning/                        # Fine-tuning scripts and configs
    │ ├── minerva-350M/
    │ └── minerva-1B/
    ├── gen/                                # Generated datasets
    ├── politicians_data/                   # Raw and processed data about politicians
    ├── results/                            # Topic modeling results
    ├── sentiment_analysis/                 # Sentiment analysis notebooks and scripts
    │ ├── active_learning_sentiment.ipynb
    │ ├── sentiment_base.ipynb
    │ ├── clean_data.ipynb
    │ ├── results_sentiment/                # Sentiment analysis results on generated data
    │ └── model_sentiment/                  # Custom sentiment model files
    ├── gen_util.ipynb
    ├── politics.ipynb
    ├── topic_classifier.ipynb
    ├── topics.ipynb
    ├── tweet_classifier.ipynb
    ├── scraper.py
    ├── requirements.txt
    ├── report.pdf
    └── README.md

Please read the included report for all the details about the project and the results obtained.