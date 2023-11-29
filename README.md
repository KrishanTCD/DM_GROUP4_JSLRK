# DM_GROUP4_JSLRK
Disclaimer:
The information, code, and data provided in this project are intended solely for the purpose of submission and analysis. The team, including its members and the owner, disclaims all responsibility, whether collective or individual, for any reference, inference, analysis mismatch, result mismatch, false findings, or any related legal consequences that may arise from the use of the code, information, file, or data provided.

By accessing and utilizing the code or information from this project, you agree that the team and the owner shall not be held responsible in any capacity for the accuracy, completeness, or reliability of the code, information, or results. The user assumes full responsibility for any consequences, legal or otherwise, that may arise from the use of the provided code or information.

The team and the owner make no warranties or representations regarding the suitability, reliability, or accuracy of the code, information, or results. The user acknowledges that they are using the provided materials at their own risk and discretion.

Analysis of the project:

The main purpose of the project is to analyze card fraud for a finanical institution specifically banks. 

The following variables are provided along with description:

# Numeric
distance_from_home: The distance from customers home to location of last transaction

distance_from_last_transaction: Distance from last transaction to the last transaction
These check are case the card is stolen and is being used in some other state.

ratio_to_median_purchase_price: Ratio of purchased price transaction to median purchase price.
Check in case user is suddenly doing a higher purchase than normal trend of the user.

repeat_retailer: If the fraud occurred is it from same origin of purchase or seller.

# Binary
used_chip: Chip implies if it was a credit card transaction.
used_pin_number: Checking if the user pin was to see if the user pin was compromised.
online_order: Major frauds occur online and as a result online fraud check is there
fraud: Flag for fraudulent model yes or no

This is a case of classical binary classification problem.

The analysis has been done in 4 major parts:

1. Initial explaratory analysis: The analysis revealed the range of variables and their deviation, additionally it also highlighted the imbalance in the dataset.
2. Pre-Processing: During the initial explaratory period due to limited computational power the sample set was reduced from 1 million observations to 100k observations using random sampling. Further stratified sampling was used to maintain class observations for the binary class variable Fraud.
3. Running baseline models: To set a standard baseline for models being run on imbalanced dataset to allow for comparison when a balanced dataset is fed.
4. Running SMOTE balanced datasets: Using SMOTE, Logistic regression, CART and Ensemble Random forest and GBM were used.

From the final remarks it can be concluded that Decision Tree on imbalanced dataset produced a good tradeoff for high accuracy and computational power. If the reader wants to pursue a higher accuracy ensemble Random Forest works very well with 99.99% accuracy.
