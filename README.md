# Semantic textual similarity in Slovak

This GitHub contains source code used to produce data for a paper not yet published. Link to the paper will be added here once it is published. Note that this project is supposed to work with Slovak language texts.

In case of any questions, do not hesitate to contact the authors:
- Lukáš Radoský - lukas.radosky59@gmail.com, lukas.radosky@kinit.sk, lukas.radosky@fmph.uniba.sk
- Miroslav Blšták - miroslav.blstak@kinit.sk

The GitHub consists of two parts:
- **Optimizing** - contains the source code used to produce data for the paper, including dataset preprocessing, unsupervised method calculations, artificial beehive optimization of machine learning models using unsupervised methods as features and statistical calculations of results.
- **Ready-to-use** - contains modified subpart of the above described source code. It offers pretrained models ready for use in your project. The root of this folder demonstrates example usage. Model names specify optimization run (*1st* vs *2nd*), dataset version (*raw* vs *lemmatized*) and dataset name. We recommend using models from *2nd* optimization run. Unless you provide lemmatized input, we suggest using *raw* models. Models trained on *sick* or *all* dataset should cover your needs the best, as they were trained on the largest sample of data. You may also try to compare all existing models to find the best one for you.

Note that these projects depend on various libraries:

    pip install textdistance
    pip install nltk
    pip install scipy
    pip install scikit-learn==0.24.2

Below, we display tables describing all provided models and their configurations. The table was unnecessarily large for the paper. Hyperparameter names and values respect Python notation of [sklearn library v0.24.2]([https://scikit-learn.org/](https://scikit-learn.org/stable/whats_new/v0.24.html)).

## First optimization run

| Dataset      | Version | ModelType                    | Validation Pearson | Feature Count | Features             | Hyperparameters        |
| ------------ | ------- | ---------------------------- | ------------------ | ------------- | -------------------- | ---------------------- |
| semeval-2012 | raw     | Gradient Boosting Regression | 0.557              | 16            | cosine               | loss:huber             |
|              |         |                              |                    |               | damerau\_levenshtein | max\_depth:12          |
|              |         |                              |                    |               | hamming              | max\_features:log2     |
|              |         |                              |                    |               | jaccard              | max\_leaf\_nodes:22    |
|              |         |                              |                    |               | jaro                 | min\_samples\_leaf:3   |
|              |         |                              |                    |               | jaro\_winkler        | min\_samples\_split:6  |
|              |         |                              |                    |               | lcsseq               | n\_estimators:300      |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              | lemma   | Random Forest Regression     | 0.560              | 16            | cosine               | max\_depth:13          |
|              |         |                              |                    |               | damerau\_levenshtein | max\_features:auto     |
|              |         |                              |                    |               | euclidean            | max\_leaf\_nodes:22    |
|              |         |                              |                    |               | jaccard              | min\_samples\_leaf:1   |
|              |         |                              |                    |               | jaro                 | min\_samples\_split:15 |
|              |         |                              |                    |               | jaro\_winkler        | n\_estimators:50       |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
| semeval-2013 | raw     | Random Forest Regression     | 0.591              | 17            | cosine               | max\_depth:17          |
|              |         |                              |                    |               | damerau\_levenshtein | max\_features:log2     |
|              |         |                              |                    |               | hamming              | max\_leaf\_nodes:14    |
|              |         |                              |                    |               | jaccard              | min\_samples\_leaf:11  |
|              |         |                              |                    |               | jaro                 | min\_samples\_split:33 |
|              |         |                              |                    |               | jaro\_winkler        | n\_estimators:200      |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
|              | lemma   | Linear Regression            | 0.575              | 18            | cosine               | fit\_intercept:True    |
|              |         |                              |                    |               | cosine\_vector       | normalize:True         |
|              |         |                              |                    |               | damerau\_levenshtein |                        |
|              |         |                              |                    |               | hamming              |                        |
|              |         |                              |                    |               | jaccard              |                        |
|              |         |                              |                    |               | jaro                 |                        |
|              |         |                              |                    |               | jaro\_winkler        |                        |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
| semeval-2014 | raw     | Random Forest Regression     | 0.571              | 19            | cosine               | max\_depth:6           |
|              |         |                              |                    |               | cosine\_vector       | max\_features:auto     |
|              |         |                              |                    |               | damerau\_levenshtein | max\_leaf\_nodes:26    |
|              |         |                              |                    |               | euclidean            | min\_samples\_leaf:24  |
|              |         |                              |                    |               | jaccard              | min\_samples\_split:34 |
|              |         |                              |                    |               | jaro                 | n\_estimators:200      |
|              |         |                              |                    |               | jaro\_winkler        |                        |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
|              | lemma   | Random Forest Regression     | 0.599              | 15            | cosine               | max\_depth:10          |
|              |         |                              |                    |               | damerau\_levenshtein | max\_features:sqrt     |
|              |         |                              |                    |               | hamming              | max\_leaf\_nodes:22    |
|              |         |                              |                    |               | jaccard              | min\_samples\_leaf:2   |
|              |         |                              |                    |               | jaro                 | min\_samples\_split:28 |
|              |         |                              |                    |               | jaro\_winkler        | n\_estimators:50       |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
| semeval-2015 | raw     | Random Forest Regression     | 0.442              | 17            | cosine               | max\_depth:11          |
|              |         |                              |                    |               | damerau\_levenshtein | max\_features:auto     |
|              |         |                              |                    |               | euclidean            | max\_leaf\_nodes:18    |
|              |         |                              |                    |               | hamming              | min\_samples\_leaf:25  |
|              |         |                              |                    |               | jaccard              | min\_samples\_split:2  |
|              |         |                              |                    |               | jaro                 | n\_estimators:100      |
|              |         |                              |                    |               | jaro\_winkler        |                        |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
|              | lemma   | Linear Regression            | 0.461              | 15            | cosine               | fit\_intercept:True    |
|              |         |                              |                    |               | damerau\_levenshtein | normalize:True         |
|              |         |                              |                    |               | euclidean            |                        |
|              |         |                              |                    |               | jaccard              |                        |
|              |         |                              |                    |               | jaro                 |                        |
|              |         |                              |                    |               | jaro\_winkler        |                        |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
| semeval-2016 | raw     | Random Forest Regression     | 0.184              | 18            | cosine               | max\_depth:7           |
|              |         |                              |                    |               | cosine\_vector       | max\_features:sqrt     |
|              |         |                              |                    |               | damerau\_levenshtein | max\_leaf\_nodes:29    |
|              |         |                              |                    |               | hamming              | min\_samples\_leaf:4   |
|              |         |                              |                    |               | jaro                 | min\_samples\_split:36 |
|              |         |                              |                    |               | jaro\_winkler        | n\_estimators:100      |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
|              | lemma   | Support Vector Regression    | 0.166              | 15            | cosine               | C:0.2                  |
|              |         |                              |                    |               | cosine\_vector       | coef0:0                |
|              |         |                              |                    |               | damerau\_levenshtein | degree:5               |
|              |         |                              |                    |               | euclidean            | epsilon:0.01           |
|              |         |                              |                    |               | hamming              | gamma:scale            |
|              |         |                              |                    |               | jaccard              | kernel:rbf             |
|              |         |                              |                    |               | jaro                 | max\_iter:176          |
|              |         |                              |                    |               | jaro\_winkler        | shrinking:True         |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
| semeval-all  | raw     | Gradient Boosting Regression | 0.404              | 19            | cosine               | loss:ls                |
|              |         |                              |                    |               | damerau\_levenshtein | max\_depth:14          |
|              |         |                              |                    |               | euclidean            | max\_features:log2     |
|              |         |                              |                    |               | hamming              | max\_leaf\_nodes:39    |
|              |         |                              |                    |               | jaccard              | min\_samples\_leaf:11  |
|              |         |                              |                    |               | jaro                 | min\_samples\_split:9  |
|              |         |                              |                    |               | jaro\_winkler        | n\_estimators:200      |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
|              | lemma   | Linear Regression            | 0.409              | 17            | cosine               | fit\_intercept:True    |
|              |         |                              |                    |               | euclidean            | normalize:True         |
|              |         |                              |                    |               | jaccard              |                        |
|              |         |                              |                    |               | jaro                 |                        |
|              |         |                              |                    |               | jaro\_winkler        |                        |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
| sick         | raw     | Gradient Boosting Regression | 0.697              | 20            | cosine               | loss:ls                |
|              |         |                              |                    |               | damerau\_levenshtein | max\_depth:14          |
|              |         |                              |                    |               | euclidean            | max\_features:auto     |
|              |         |                              |                    |               | hamming              | max\_leaf\_nodes:41    |
|              |         |                              |                    |               | jaccard              | min\_samples\_leaf:15  |
|              |         |                              |                    |               | jaro                 | min\_samples\_split:34 |
|              |         |                              |                    |               | jaro\_winkler        | n\_estimators:200      |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | leacock\_chodorow    |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
|              | lemma   | Gradient Boosting Regression | 0.708              | 18            | cosine               | loss:huber             |
|              |         |                              |                    |               | cosine\_vector       | max\_depth:11          |
|              |         |                              |                    |               | damerau\_levenshtein | max\_features:sqrt     |
|              |         |                              |                    |               | euclidean            | max\_leaf\_nodes:10    |
|              |         |                              |                    |               | hamming              | min\_samples\_leaf:20  |
|              |         |                              |                    |               | jaro                 | min\_samples\_split:38 |
|              |         |                              |                    |               | jaro\_winkler        | n\_estimators:300      |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
| all          | raw     | Gradient Boosting Regression | 0.459              | 18            | cosine               | loss:lad               |
|              |         |                              |                    |               | cosine\_vector       | max\_depth:21          |
|              |         |                              |                    |               | damerau\_levenshtein | max\_features:log2     |
|              |         |                              |                    |               | hamming              | max\_leaf\_nodes:39    |
|              |         |                              |                    |               | jaro                 | min\_samples\_leaf:9   |
|              |         |                              |                    |               | jaro\_winkler        | min\_samples\_split:34 |
|              |         |                              |                    |               | lcsseq               | n\_estimators:200      |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | minkowski            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | path                 |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |
|              | lemma   | Gradient Boosting Regression | 0.468              | 18            | cosine               | loss:ls                |
|              |         |                              |                    |               | cosine\_vector       | max\_depth:14          |
|              |         |                              |                    |               | damerau\_levenshtein | max\_features:log2     |
|              |         |                              |                    |               | euclidean            | max\_leaf\_nodes:12    |
|              |         |                              |                    |               | hamming              | min\_samples\_leaf:16  |
|              |         |                              |                    |               | jaccard              | min\_samples\_split:20 |
|              |         |                              |                    |               | jaro                 | n\_estimators:300      |
|              |         |                              |                    |               | jaro\_winkler        |                        |
|              |         |                              |                    |               | lcsseq               |                        |
|              |         |                              |                    |               | lcsstr               |                        |
|              |         |                              |                    |               | levenshtein          |                        |
|              |         |                              |                    |               | manhattan            |                        |
|              |         |                              |                    |               | needleman\_wunsch    |                        |
|              |         |                              |                    |               | ochiai               |                        |
|              |         |                              |                    |               | overlap              |                        |
|              |         |                              |                    |               | smith\_waterman      |                        |
|              |         |                              |                    |               | sorensen\_dice       |                        |
|              |         |                              |                    |               | wu\_palmer           |                        |

## Second optimization run

| Dataset      | Version | ModelType                    | Validation Pearson | Feature Count | Features             | Hyperparameters                 |
| ------------ | ------- | ---------------------------- | ------------------ | ------------- | -------------------- | ------------------------------- |
| semeval-2012 | raw     | Gradient Boosting Regression | 0.549              | 7             | cosine\_vector       | loss:lad                        |
|              |         |                              |                    |               | euclidean            | max\_depth:8                    |
|              |         |                              |                    |               | minkowski            | max\_features:auto              |
|              |         |                              |                    |               | path                 | max\_leaf\_nodes:39             |
|              |         |                              |                    |               | wu\_palmer           | min\_samples\_leaf:5            |
|              |         |                              |                    |               | sorensen\_dice       | min\_samples\_split:19          |
|              |         |                              |                    |               | lcsseq               | n\_estimators:200               |
|              | lemma   | Random Forest Regression     | 0.551              | 6             | euclidean            | max\_depth:19                   |
|              |         |                              |                    |               | minkowski            | max\_features:sqrt              |
|              |         |                              |                    |               | path                 | max\_leaf\_nodes:9              |
|              |         |                              |                    |               | levenshtein          | min\_samples\_leaf:1            |
|              |         |                              |                    |               | cosine               | min\_samples\_split:40          |
|              |         |                              |                    |               | needleman\_wunsch    | n\_estimators:300               |
| semeval-2013 | raw     | Bayesan Ridge Regression     | 0.587              | 7             | cosine\_vector       | alpha\_1:2e-06                  |
|              |         |                              |                    |               | euclidean            | alpha\_2:1e-06                  |
|              |         |                              |                    |               | leacock\_chodorow    | alpha\_init:0.001               |
|              |         |                              |                    |               | manhattan            | lambda\_1:0.0002                |
|              |         |                              |                    |               | lcsseq               | lambda\_2:0.0005                |
|              |         |                              |                    |               | overlap              | n\_iter:250                     |
|              |         |                              |                    |               | overlap              |                                 |
|              | lemma   | Linear Regression            | 0.576              | 10            | cosine\_vector       | fit\_intercept:True             |
|              |         |                              |                    |               | euclidean            | normalize:True                  |
|              |         |                              |                    |               | leacock\_chodorow    |                                 |
|              |         |                              |                    |               | manhattan            |                                 |
|              |         |                              |                    |               | minkowski            |                                 |
|              |         |                              |                    |               | path                 |                                 |
|              |         |                              |                    |               | wu\_palmer           |                                 |
|              |         |                              |                    |               | sorensen\_dice       |                                 |
|              |         |                              |                    |               | ochiai               |                                 |
|              |         |                              |                    |               | cosine               |                                 |
| semeval-2014 | raw     | Bayesan Ridge Regression     | 0.592              | 8             | cosine\_vector       | alpha\_1:0.0005                 |
|              |         |                              |                    |               | euclidean            | alpha\_2:4.9999999999999996e-06 |
|              |         |                              |                    |               | minkowski            | alpha\_init:None                |
|              |         |                              |                    |               | path                 | lambda\_1:1e-06                 |
|              |         |                              |                    |               | wu\_palmer           | lambda\_2:2e-05                 |
|              |         |                              |                    |               | overlap              | n\_iter:200                     |
|              |         |                              |                    |               | needleman\_wunsch    |                                 |
|              |         |                              |                    |               | lcsseq               |                                 |
|              | lemma   | Linear Regression            | 0.581              | 7             | leacock\_chodorow    | fit\_intercept:True             |
|              |         |                              |                    |               | manhattan            | normalize:True                  |
|              |         |                              |                    |               | path                 |                                 |
|              |         |                              |                    |               | wu\_palmer           |                                 |
|              |         |                              |                    |               | overlap              |                                 |
|              |         |                              |                    |               | overlap              |                                 |
|              |         |                              |                    |               | lcsseq               |                                 |
| semeval-2015 | raw     | Random Forest Regression     | 0.446              | 8             | cosine\_vector       | max\_depth:8                    |
|              |         |                              |                    |               | leacock\_chodorow    | max\_features:log2              |
|              |         |                              |                    |               | manhattan            | max\_leaf\_nodes:8              |
|              |         |                              |                    |               | minkowski            | min\_samples\_leaf:22           |
|              |         |                              |                    |               | path                 | min\_samples\_split:16          |
|              |         |                              |                    |               | wu\_palmer           | n\_estimators:200               |
|              |         |                              |                    |               | lcsstr               |                                 |
|              |         |                              |                    |               | jaccard              |                                 |
|              | lemma   | Bayesan Ridge Regression     | 0.470              | 8             | cosine\_vector       | alpha\_1:0.0002                 |
|              |         |                              |                    |               | euclidean            | alpha\_2:2e-06                  |
|              |         |                              |                    |               | leacock\_chodorow    | alpha\_init:0.001               |
|              |         |                              |                    |               | manhattan            | lambda\_1:1e-06                 |
|              |         |                              |                    |               | minkowski            | lambda\_2:0.0001                |
|              |         |                              |                    |               | lcsseq               | n\_iter:200                     |
|              |         |                              |                    |               | needleman\_wunsch    |                                 |
|              |         |                              |                    |               | sorensen\_dice       |                                 |
| semeval-2016 | raw     | Random Forest Regression     | 0.165              | 7             | cosine\_vector       | max\_depth:11                   |
|              |         |                              |                    |               | leacock\_chodorow    | max\_features:sqrt              |
|              |         |                              |                    |               | manhattan            | max\_leaf\_nodes:8              |
|              |         |                              |                    |               | minkowski            | min\_samples\_leaf:2            |
|              |         |                              |                    |               | path                 | min\_samples\_split:10          |
|              |         |                              |                    |               | damerau\_levenshtein | n\_estimators:300               |
|              |         |                              |                    |               | sorensen\_dice       |                                 |
|              | lemma   | Linear Regression            | 0.153              | 9             | euclidean            | fit\_intercept:True             |
|              |         |                              |                    |               | leacock\_chodorow    | normalize:True                  |
|              |         |                              |                    |               | manhattan            |                                 |
|              |         |                              |                    |               | minkowski            |                                 |
|              |         |                              |                    |               | path                 |                                 |
|              |         |                              |                    |               | wu\_palmer           |                                 |
|              |         |                              |                    |               | needleman\_wunsch    |                                 |
|              |         |                              |                    |               | jaro                 |                                 |
|              |         |                              |                    |               | lcsseq               |                                 |
| semeval-all  | raw     | Gradient Boosting Regression | 0.416              | 7             | cosine\_vector       | loss:ls                         |
|              |         |                              |                    |               | leacock\_chodorow    | max\_depth:13                   |
|              |         |                              |                    |               | minkowski            | max\_features:sqrt              |
|              |         |                              |                    |               | wu\_palmer           | max\_leaf\_nodes:7              |
|              |         |                              |                    |               | lcsstr               | min\_samples\_leaf:16           |
|              |         |                              |                    |               | overlap              | min\_samples\_split:2           |
|              |         |                              |                    |               | lcsseq               | n\_estimators:200               |
|              | lemma   | Gradient Boosting Regression | 0.405              | 6             | cosine\_vector       | loss:ls                         |
|              |         |                              |                    |               | leacock\_chodorow    | max\_depth:12                   |
|              |         |                              |                    |               | manhattan            | max\_features:auto              |
|              |         |                              |                    |               | path                 | max\_leaf\_nodes:4              |
|              |         |                              |                    |               | ochiai               | min\_samples\_leaf:13           |
|              |         |                              |                    |               | ochiai               | min\_samples\_split:49          |
|              |         |                              |                    |               |                      | n\_estimators:200               |
| sick         | raw     | Random Forest Regression     | 0.678              | 8             | cosine\_vector       | max\_depth:12                   |
|              |         |                              |                    |               | euclidean            | max\_features:log2              |
|              |         |                              |                    |               | manhattan            | max\_leaf\_nodes:8              |
|              |         |                              |                    |               | minkowski            | min\_samples\_leaf:20           |
|              |         |                              |                    |               | path                 | min\_samples\_split:11          |
|              |         |                              |                    |               | wu\_palmer           | n\_estimators:100               |
|              |         |                              |                    |               | cosine               |                                 |
|              |         |                              |                    |               | lcsstr               |                                 |
|              | lemma   | Random Forest Regression     | 0.683              | 7             | euclidean            | max\_depth:15                   |
|              |         |                              |                    |               | leacock\_chodorow    | max\_features:sqrt              |
|              |         |                              |                    |               | minkowski            | max\_leaf\_nodes:9              |
|              |         |                              |                    |               | wu\_palmer           | min\_samples\_leaf:7            |
|              |         |                              |                    |               | overlap              | min\_samples\_split:35          |
|              |         |                              |                    |               | overlap              | n\_estimators:100               |
|              |         |                              |                    |               | lcsstr               |                                 |
| all          | raw     | Gradient Boosting Regression | 0.461              | 9             | cosine\_vector       | loss:ls                         |
|              |         |                              |                    |               | euclidean            | max\_depth:12                   |
|              |         |                              |                    |               | leacock\_chodorow    | max\_features:log2              |
|              |         |                              |                    |               | minkowski            | max\_leaf\_nodes:17             |
|              |         |                              |                    |               | path                 | min\_samples\_leaf:21           |
|              |         |                              |                    |               | wu\_palmer           | min\_samples\_split:9           |
|              |         |                              |                    |               | overlap              | n\_estimators:200               |
|              |         |                              |                    |               | lcsseq               |                                 |
|              |         |                              |                    |               | lcsstr               |                                 |
|              | lemma   | Gradient Boosting Regression | 0.464              | 8             | cosine\_vector       | loss:lad                        |
|              |         |                              |                    |               | manhattan            | max\_depth:17                   |
|              |         |                              |                    |               | minkowski            | max\_features:sqrt              |
|              |         |                              |                    |               | path                 | max\_leaf\_nodes:27             |
|              |         |                              |                    |               | wu\_palmer           | min\_samples\_leaf:5            |
|              |         |                              |                    |               | cosine               | min\_samples\_split:17          |
|              |         |                              |                    |               | lcsstr               | n\_estimators:300               |
|              |         |                              |                    |               | overlap              |                                 |
