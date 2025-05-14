# Linear Regression Data Science Project on Cancer Severity Scores Predictions
## Case Description
This data science case study focuses on a group of cancer patients with their cancer related data and the patients' cancer severity scores. The goal would be to find patterns between features and cancer severity scores, and build three machine learning models to predict the score given the info.

## Data Description
There are in total 15 columns and 50000 rows.

![image](https://github.com/user-attachments/assets/28eb9fed-52f8-4369-8c9a-c00e26523999)


## Table of Content
* Part 1: DEA & Visualization
* Part 2.1: Machine Learning Model: Lasso Regression
* Part 2.2: Machine Learning Model: Ridge Regression
* Part 2.3: Machine Learning Model: Deep Learning
* Part 3: Machine Learning Model: KMeans Clustering
* Part 4: Conclusions & Recommendations

## Part 1: DEA & Visualization
* Most of the features are uniformly distributed in two tiers. One group would have lower level of data points than the other group. But both groups are uniformly distributed on its own. Target Severity Scores have almost the perfect normal distribution. with a mean of around 5.

![image](https://github.com/user-attachments/assets/09fc9afc-f3f1-4287-932a-6aa01e8016f7)

* For the categorical features such as gender, country region, cancer type, and cancer stage, each category from those features would have equal amont of patients. The population of colon cancer is slightly more than other type of cancers.
* **Insights**: Gender, county region do't play a role on severity of cancer. And each type of cancer has a similar percentage of chance to be had. 

![image](https://github.com/user-attachments/assets/afd65609-d140-4fa2-afda-79f7f9b3bfad)

* The following scatter plot has showed that there are strong correlations between target severity scores and genetic risk, air pollution, alcohol use, smoking, obesity level, treatment cost. And the following correlation analysis had confirmed the findings from scatterplot.

* **Insights**: Genetics, air pollutions, alcohol use, smoking, and obesity level would increase the severity of cancer. Have a healthies life style, clean air to breath in and a healthy weight would reduce cancer severity. Cancer treatment is very ecpensive, but they are effective to reduce the cancer. The effective cancer treatment is always expensive. The two most relevant features are genetic risks, smoking, and treatment cost. 

![image](https://github.com/user-attachments/assets/f99872a5-3b6c-4b5c-8a89-9d7430e3b723)

![image](https://github.com/user-attachments/assets/5ae938f5-8eb3-4e9b-b2e2-222477441aec)

* Each gender have similar use of alcohol and smoking level.
* Each country have a similar level of air pollution, obesity level.
* Each cancer type has similar distribution of genetic risks and treatment cost, similar treatment cost.
* Each cancer stage has similar obesity distribution and treatment cost distributions.

![image](https://github.com/user-attachments/assets/e9d32a0b-90a5-4cc5-a100-85b3d7b0aae2)

## Part 2.1 Lasso Regression

* Applied Lasso CV for a series of alphas. The best alpha is 0.001.
* The training r2 Score is 0.9999, and the testing r2 Score is 0.99998.
* The adjusted training r2 Score is 0.9999, and the adjusted testing r2 Score is 0.9999.
* The testing Mean Squared Error is 1.42e-05, and The testing Root Mean Squared Error is 0.00377, and the testing Mean Absolute Error is 0.003086.
* This is a very accurate model since it has the mos relevant features such as pollution level, obesity level, genetics, and treatment cost. All the prediction values and observed values are able to form a straight line along diagnal.

![image](https://github.com/user-attachments/assets/9f434333-ca08-42f2-9d35-11a466ff74ed)

## Part 2.2 Ridge Regression

* Applied Ridge CV. The best alpha is 0.001.
* The training r2 Score is 0.9999, and the testing r2 Score is 0.9999.
* The training adjusted r2 Score is 0.9999, and the testing adjusted r2 Score is 0.9999.
* The testing Mean Squared Error is 8.3295e-06, and The testing Root Mean Squared Error is 0.002886, and the testing Mean Absolute Error is 0.0024959.
* Once again, this is a very accurate model. All the prediction values and observed values are able to form a straight line along diagnal.

![image](https://github.com/user-attachments/assets/e61aa286-fd93-4a5f-b700-43bfd21f7309)

## Part 2.3 Deep Learning 
