# Machine learning based feature(gene) selection for better predictive model of Bipolar Disorder #

Bipolar disorder (BD) is a severe, highly heritable, recurrent mood disorder, associated with a significant morbidity and mortality. In the current clinical practice, the diagnosis of BD is made by history taking, interview and behavioural observations, thereby lacking an objective, biological validation. 

Next-generation sequencing (NGS) is a massively parallel sequencing technology that offers ultra-high throughput, scalability, and speed. One of the most widely used NGS is RNA-seq, which has the ability to measure the expression level of all gene in one experiment. Hence it's a good data source for developing predictive model of Bipolar Disorder.


### Whole blood RNA-seq data ###

The whole blood RNA-sequencing data comes from 240 bipolar disorder patients and 240 healthy controls. After quality control, there are 13k genes left.


<p align="center">
<img src="/imgs/RNA_seq_raw.png">
<br>
<em>Bitcoin</em></p>

Here shows the valcano plot of the result of Differential Expressed Genes analysis:

<p align="center">
<img src="/imgs/valcano.png">
<br>
<em>Bitcoin</em></p>

### Feature selection using mRMR "minimum Redundancy - Maximum Relevance" ###
y is the target variable, is the variable you want to predict. The rest of the variables are the features we want to select from. K is the number of features will be selected.

```python
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=10)

selected_features
['TAGLN2P1',
 'NOLC1P1',
 'UBL5P2',
 'tobacco',
 'RN7SKP70',
 'VTRNA1-1',
 'age',
 'TBC1D22B',
 'MIR23A',
 'SLPI']
```

### Different maching learning algorithm to search for the best model ###

I tested Random Forest, SVM, XGB/Gradient Boosting, LightGBM, Catboost using grid search to compare their best performance.

<p align="center">
<img src="/imgs/Roc.png">
<br>
<em>RocAUC of the best model from the grid search of different algorithms</em></p>

### References ###
[1] Molly Howland, M.D. Alex El Sehamy, M.D. What Are Bipolar Disorders?. BMC Med Genomics 13, 122 (2020).

[2] Chris Ding, Hanchuan Peng. Minimum redundancy feature selection from microarray gene expression data. J Bioinform Comput Biol. 2005 Apr;3(2):185-205.

