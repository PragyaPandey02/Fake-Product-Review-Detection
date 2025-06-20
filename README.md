# Fake-Product-Review-Detection
The project aims to detect fake product reviews using various supervised machine learning algorithms. The  methodology involves preprocessing review text using NLP techniques like tokenization, lemmatization, and TF IDF vectorization, followed by training models such as SVM, Random Forest, and XGBoost for classification. 

# Fake Product Review Detection Using Supervised Machine Learning Techniques

This project focuses on detecting fake product reviews from e-commerce platforms using various supervised machine learning algorithms. By leveraging Natural Language Processing (NLP) techniques and powerful classifiers, the model helps identify deceptive reviews and improve trust in online marketplaces.

## Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Overview
Online product reviews have a significant impact on customer purchasing behavior. However, fake or deceptive reviews can mislead users and harm brand reputations. This project implements and compares multiple machine learning models to identify such reviews effectively.

## Methodology
1. Data Collection: Labeled dataset containing genuine and fake reviews was used.
2. Preprocessing:
   - Tokenization
   - Lowercasing
   - Stopword removal
   - Lemmatization
   - TF-IDF vectorization
3. Model Training & Evaluation: 
   - Algorithms used: SVM, Random Forest, Naive Bayes, KNN, Decision Tree, XGBoost, SGD
   - Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

## Technologies Used
- Python
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Matplotlib / Seaborn (for visualizations)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-review-detection.git
   cd fake-review-detection

Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Run the preprocessing script to clean and vectorize the dataset.

Execute the model training script to evaluate different classifiers.

Use the test script to classify new review inputs as genuine or fake.

Results
Best-performing model: Support Vector Machine (SVM)

Achieved Accuracy: 85.57%

Balanced performance across Precision, Recall, and F1-score

Robust against textual data complexity and imbalances

Contributors
Kumari Sakshi - sakshi01dolly@gmail.com

Pragya Pandey - pragyaa1857@gmail.com

