Projekat 2: Data Analysis and Prediction using WEKA

Overview

This project focuses on a comprehensive analysis of a banking dataset to predict whether a client will purchase a Personal Equity Plan (PEP).
Dataset

- Source: [Bank Data](http://facweb.cs.depaul.edu/mobasher/classes/ect584/WEKA/data/bank-data.csv)
- Attributes include ID, age, sex, region, income, marital status, etc.

![attributs](https://github.com/AsinaMilic/Skladistenje-podataka-i-otkrivanje-znanja/assets/54029561/df579330-f4ba-4522-aa0a-0efc92d6b581)


Methodology

- **Data Preprocessing**: Removal of unnecessary attributes like ID, data discretization, and normalization of numerical values.

![image](https://github.com/AsinaMilic/Skladistenje-podataka-i-otkrivanje-znanja/assets/54029561/1e6a8923-855c-4306-ae57-dcc1366c9e73)

- **Analysis**: Implementation of various algorithms like NaiveBayes, ZeroR, OneR, K Nearest Neighbors (kNN), J48, and Random Forest.
- **Performance Evaluation**: Using metrics like accuracy, sensitivity, specificity, precision, recall, and F-measure.

![image](https://github.com/AsinaMilic/Skladistenje-podataka-i-otkrivanje-znanja/assets/54029561/70953b96-7c18-4f17-aea6-ce7bf0ec5b56)


Key Findings

- The analysis included both unfiltered and discretized datasets, with NaiveBayes serving as a baseline.
- The use of ensemble learning techniques, particularly Bagging, yielded the best results.
- Detailed analysis of the algorithms' performance using confusion matrix and ROC curves.

Conclusion

The project demonstrates the effectiveness of various machine learning algorithms in predicting client behavior based on banking data, with a particular emphasis on ensemble techniques for enhanced accuracy.

