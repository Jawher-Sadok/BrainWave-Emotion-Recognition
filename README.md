# Emotion Classification with Explainable AI (XAI) - Function Documentation

## Overview
This MATLAB-based emotion classification system implements a complete machine learning pipeline with integrated Explainable AI components. The system processes EEG-derived features to classify emotional states (Relaxed vs. Funny/Happy) while providing transparent insights into model decisions.

## Core Functions

### 1. Data Loading and Preprocessing

#### `load_and_preprocess_data(filename)`
- Loads emotion data from a CSV file
- Extracts and validates feature names while excluding label columns
- Removes specified irrelevant features (MeanDelta10sec, BetaAveraged, SumDelta10sec, DeltaAveraged)
- Creates binary labels (1 for Relaxed, 2 for Funny/Happy)
- Handles samples with multiple or missing labels
- Removes rows with NaN values
- Normalizes features using z-score standardization
- Visualizes feature distributions by class
- Returns processed features, labels, and feature names

#### `visualize_feature_distributions(features, labels, featureNames)`
- Creates histogram visualizations for each feature
- Displays distributions separated by emotion class
- Uses blue for Relaxed and red for Funny/Happy samples
- Provides a comprehensive view of feature characteristics

### 2. Feature Analysis

#### `analyze_features(features, labels, featureNames)`
- Performs statistical analysis of features by class
- Calculates correlation matrix between features
- Computes t-tests to identify significant differences between classes
- Displays formatted results with means and p-values
- Calls visualization function for comprehensive analysis

#### `visualize_feature_analysis(features, labels, featureNames, corrMatrix, p_values)`
- Creates four-panel visualization:
  - Correlation heatmap between features
  - PCA projection of samples colored by class
  - Feature means comparison by class
  - Statistical significance (-log10 of p-values)
- Provides multidimensional insight into feature relationships and discriminative power

### 3. Model Training with XAI

#### `train_emotion_classifier(features, labels, featureNames)`
- Main training function with integrated XAI components
- Splits data into training and testing sets (70/30)
- Handles class imbalance using SMOTE oversampling
- Performs hyperparameter optimization via grid search
- Trains ensemble classifier with bagged decision trees
- Performs cross-validation to estimate performance
- Calculates feature importance
- Applies feature selection based on importance thresholds
- Generates XAI explanations and visualizations
- Returns trained model, accuracy metrics, feature importance, and XAI results

#### `smote_oversample(X, y)`
- Implements simplified SMOTE (Synthetic Minority Over-sampling Technique)
- Identifies minority class and replicates samples
- Balances class distribution while maintaining data characteristics
- Shuffles the augmented dataset to maintain randomness

### 4. Explainable AI Components

#### `generate_xai_explanations(model, X_test, y_test, featureNames, featureImportance)`
- Coordinates various XAI techniques to explain model behavior
- Computes partial dependence plots to show feature effects
- Generates local explanations for individual predictions
- Extracts interpretable decision rules from the model
- Analyzes confidence calibration metrics
- Returns comprehensive XAI results structure

#### `compute_partial_dependence(model, X, featureNames, importance, numTopFeatures)`
- Calculates how predictions change with feature values
- Creates grid points for top important features
- Computes average predictions across feature value ranges
- Helps understand global feature behavior

#### `compute_local_explanations(model, X, y, featureNames, numSamples)`
- Generates explanations for individual predictions
- Selects representative samples from each class
- Calculates feature contributions using perturbation analysis
- Provides insight into why specific predictions were made

#### `extract_decision_rules(model, featureNames, numRules)`
- Creates interpretable rules based on feature importance
- Translates complex model into human-understandable logic
- Identifies the most influential decision patterns

#### `analyze_confidence_calibration(model, X_test, y_test)`
- Evaluates how well prediction confidence matches accuracy
- Calculates calibration error between average confidence and accuracy
- Assesses model reliability for trustworthy deployment

### 5. Prediction and Explanation

#### `predict_emotion(model, new_data, featureNames, xaiResults)`
- Makes predictions on new data samples
- Validates input dimensions against expected features
- Normalizes new data using same criteria as training
- Returns predictions, confidence scores, and class probabilities
- Displays XAI insights when provided
- Visualizes predictions for single samples

#### `display_xai_insights(xaiResults, sampleData, featureNames, prediction, confidence)`
- Shows detailed explanations for specific predictions
- Identifies top influential features and their values
- Provides confidence assessment and calibration context
- Explains key decision factors for the prediction
- Highlights relevant decision patterns

#### `visualize_prediction(scores, confidence, prediction)`
- Creates two-panel visualization for single predictions:
  - Bar chart showing class probability scores
  - Polar plot showing confidence level as gauge
- Uses color coding for intuitive interpretation

### 6. Dashboard and Interactive Features

#### `display_xai_dashboard(xaiResults, featureNames)`
- Presents comprehensive model transparency report
- Shows top important features with scores
- Displays confidence calibration metrics
- Lists key decision patterns learned by the model
- Provides model trustworthiness assessment
- Offers specific recommendations for improvement

#### `interactive_xai_exploration(model, features, labels, featureNames, xaiResults)`
- Provides menu-driven interface for exploring model behavior
- Allows users to choose different exploration modes
- Coordinates between various exploration functions

#### `explore_feature_explanations(xaiResults, featureNames)`
- Details how each feature influences predictions
- Shows importance scores and average effects on outcomes
- Helps understand feature contribution mechanisms

#### `test_what_if_scenarios(model, features, labels, featureNames, xaiResults)`
- Allows users to explore counterfactual scenarios
- Shows how changing feature values affects predictions
- Identifies most impactful features for manipulation

#### `analyze_specific_samples(model, features, labels, featureNames, xaiResults)`
- Provides detailed analysis of selected samples
- Shows true labels, predictions, and confidence levels
- Highlights correct/incorrect predictions

### 7. Main Execution

#### `main_emotion_predictions()`
- Coordinates the entire workflow
- Calls functions in proper sequence:
  - Data loading and preprocessing
  - Feature analysis
  - Model training with XAI
  - Demonstration predictions
  - XAI dashboard display
  - Interactive exploration
- Handles errors and provides user feedback
- Displays comprehensive summary of results

## Supporting Utilities

#### `visualize_training_results(model, X_test, y_test, featureImportance, featureNames, xaiResults)`
- Creates performance visualization dashboard:
  - Confusion matrix
  - Feature importance horizontal bar chart
  - ROC curve with AUC score
- Provides at-a-glance model assessment

#### `ifelse(condition, trueVal, falseVal)`
- Ternary operator replacement for MATLAB
- Improves code readability in conditional display statements

## Workflow Summary
The system implements a comprehensive pipeline:

1. **Data Preparation**: Loading, cleaning, and normalizing emotion data
2. **Exploratory Analysis**: Understanding feature distributions and relationships
3. **Model Training**: Building an optimized classifier with ensemble methods
4. **XAI Integration**: Generating explanations for model transparency
5. **Prediction**: Making classifications with confidence estimates
6. **Visualization**: Creating intuitive displays of results and explanations
7. **Interaction**: Allowing users to explore and understand model behavior

This integrated approach ensures both high performance and interpretability, making the system suitable for research and potential clinical applications where understanding model decisions is crucial.