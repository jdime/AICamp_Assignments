##################################################
## ML From Theory To Modeling Milestone Project ##
##################################################
##Analysis of two datasets: Breast Cancer and Facebook Metrics
##Three steps:
##Step 1) Implement two tasks:
##        1.1) Read the data files in each dataset and prepare them as dataframes
##        1.2) Implement three supervised machine learning methods on each dataset including: Regression, RF and k-NN
##Step 2) Implement two tasks:
##        2.1) Implement dimensionality reduction methods on each dataset
##             a) PCA, b) t-SNE, c) UMAP
##        2.2) Implement PCA and then implement three supervised machine learning methods on each dataset including
##             a) Regression, b) Random forest, c) k-NN
##Step 3) Implement clustering:
##        3.1) Read the data files in each dataset and prepare them as data frames
##        3.2) Implement different clustering on each dataset including
##################################################

############################
##Dependencies
##Tested in Python v3.8
############################
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os as os
import requests
import io
from os.path import expanduser
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import manifold
from sklearn import cluster
import umap

############################
##Set working directory
############################
home = expanduser("~")
OutDir = home + "/MilestoneProject"
if not os.path.exists(OutDir):
    os.makedirs(OutDir)
os.chdir(OutDir)

##################################################
## STEP 1.1 - Read the data files and prepare dataframes
##################################################

############################
##Breast Cancer Dataset
############################
DatasetName = "Breast_cancer"

##Get dataset using sklearn
bcancer = load_breast_cancer()

######################
##Create pandas data frame
bcancer_data_df = pd.DataFrame(bcancer.data, columns=[bcancer.feature_names])
bcancer_target_df = pd.DataFrame(bcancer.target)

### For some reason here ( 0 = malignant and 1 = benign, but other references to this dataset have it the other way around
### https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python
bcancer_target_names_df = bcancer_target_df.copy()
bcancer_target_names_df[0].replace(0, 'Benign',inplace=True)
bcancer_target_names_df[0].replace(1, 'Malignant',inplace=True)

######################
##Exploratory Data Analysis
figure = plt.gcf()
figure.set_size_inches(8, 7)
sn.heatmap(bcancer_data_df.corr(), annot=False, xticklabels=1, yticklabels=1)
plt.tight_layout()
plt.savefig(DatasetName + '_correl_heatmap.png', dpi=100)
plt.close()

######################
##Removed 7/30 features showing Pearson > 0.95 to at least another feature in the dataset
bcancer_data_dep95_df = bcancer_data_df.iloc[:, [0,1,4,5,6,7,8,9,10,11,14,15,16,17,18,19,21,24,25,26,27,28,29]]

print("Breast Cancer 'all' dataset :", bcancer_data_df.shape)
print("Breast Cancer 'dep95' dataset :", bcancer_data_df.shape)

figure = plt.gcf()
figure.set_size_inches(8, 7)
sn.heatmap(bcancer_data_dep95_df.corr(), annot=False, xticklabels=1, yticklabels=1)
plt.tight_layout()
plt.savefig(DatasetName + '_dep95_correl_heatmap.png', dpi=100)
plt.close()

############################
##Facebook metrics dataset
############################
DatasetName = "Facebook_metrics"

FacebookURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00488/Live.csv"
response = requests.get(FacebookURL).content
facebook_df = pd.read_csv(io.StringIO(response.decode('utf-8')))
# facebook_df = pd.read_csv("~/MilestoneProject/Live.csv") ### Alternatively, to load local file
facebook_df.drop(facebook_df.iloc[:, 12:], inplace=True, axis=1)
facebook_data_df = facebook_df.iloc[:, 3:12]
facebook_target_df = facebook_df.iloc[:,1:2]

## As per assignment instructions, using only the firt 1000 points
facebook_data_df = facebook_data_df.iloc[0:1000, :]
facebook_target_df = facebook_target_df.iloc[0:1000, :]

facebook_target_names_df = facebook_target_df.copy()

print("Facebook dataset :", facebook_data_df.shape)

######################
##Exploratory Data Analysis
sn.heatmap(facebook_data_df.corr(), annot=False, xticklabels=1, yticklabels=1)
plt.tight_layout()
plt.savefig(DatasetName + '_correl_heatmap.png', dpi=100)
plt.close()

sn.scatterplot(x=facebook_data_df["num_reactions"],y=facebook_data_df["num_likes"])
plt.savefig(DatasetName + '_num_reactions_V_num_likes_scatter.png', dpi=100)
plt.close()

##################################################
## STEP 1.2 - kNN, Log Reg, NB, RF
##################################################

############################
##Built-in functions
############################

##To print variable names
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

##To plot confusion matrices
def plot_confusion_matrix(y_true, y_pred, classes,
                          title="None",
                          cmap=plt.cm.Blues):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # To show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

######################
##Run classifications on each data subsample
######################
## bcancer_data_df : contains all features of the Breast Cancer dataset
## bcancer_data_dep95_df : contains features depurated by Pearson > 0.95 of the Breast Cancer dataset
## facebook_data_df : contains all features of the Facebook metrics dataset
##  note the facebook_data_df wasn't depurated. Two features (num_reactions and num_likes) showed R = 0.995
##  but their relationship in the scatter plot doesn't look like they are just duplicated, thus kept all features

### Passing (df_data, df_target_numeric, df_targer_label, n_for_kNN, n_components_for_PCA)
### n_for_kNN and n_components_for_PCA were determined empirically
SamplesToProcess_dic = {}
SamplesToProcess_dic[0] = (bcancer_data_df, bcancer_target_df, bcancer_target_names_df, 10, 6)
SamplesToProcess_dic[1] = (bcancer_data_dep95_df, bcancer_target_df, bcancer_target_names_df, 3, 5)
SamplesToProcess_dic[2] = (facebook_data_df, facebook_target_df, facebook_target_names_df, 6, 7)

########
for i in SamplesToProcess_dic:
    sample_data_df = SamplesToProcess_dic[i][0]
    sample_target_df = SamplesToProcess_dic[i][1]
    sample_target_names_df = SamplesToProcess_dic[i][2]
    n_kNN_manual = SamplesToProcess_dic[i][3]
    n_components_manual = SamplesToProcess_dic[i][4]
    sample_name = namestr(sample_data_df, globals())[0]

    print("\n**** STEP 1: ", sample_name, " ****\n")

    ##Subset bcancer.data for test and train subsets
    X = sample_data_df
    y = sample_target_df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)
    print(f'train: {X_train.size}')
    print(f'test: {X_test.size}')

    ######################
    ##K nearest neighbour
    ######################
    print("\n*** K nearest neighbour: ", sample_name, " ***\n")

    ##Initialize classifier, weight("uniform")
    knn = KNeighborsClassifier(n_neighbors=2, weights='distance')

    ##Fitting the model with the data
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    ##Performance measures
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Balanced_accuracy: ", metrics.balanced_accuracy_score(y_test, y_pred))
    print("F1_score:", metrics.f1_score(y_test, y_pred, average=None))
    print("Matthews_CC:", metrics.matthews_corrcoef(y_test, y_pred))

    ##Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=y.unique(),
                          title= (sample_name + '\nkNN Confusion Matrix'))
    OutFileName = (sample_name + '_kNN_confusion_mat.png')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ##Plot accuracy as a function of k of kNN
    k_range = list(range(1, 20))
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(k_range, scores)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Accuracy Score')
    plt.title(sample_name + '\nAccuracy Scores')
    plt.axvline(n_kNN_manual, linestyle='--')
    OutFileName = (sample_name + '_kNN_vs_Accuracy.png')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##Logistic Regression
    ######################
    print("\n*** Logistic Regression: ", sample_name, " ***\n")

    # Initialize classifier
    # QUESTION: why do we need to do:
    # `from sklearn.linear_model import LogisticRegression as LR`
    # and then `logreg = LR()`
    # Instead of `from sklearn.linear_model import LogisticRegression as logreg` directly
    # Doing it directly calls and error:
    # `logreg.fit(X_train, y_train)`
    # TypeError: fit() missing 1 required positional argument: 'y'
    logreg = LR()

    ##Fitting the model with the data
    logreg.fit(X_train, y_train)

    ##prediction in test set
    y_pred = logreg.predict(X_test)

    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Balanced_accuracy: ", metrics.balanced_accuracy_score(y_test, y_pred))
    print("F1_score:", metrics.f1_score(y_test, y_pred, average=None))
    print("Matthews_CC:", metrics.matthews_corrcoef(y_test, y_pred))

    #Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=y.unique(),
                          title= (sample_name + '\nLogR Confusion Matrix'))
    OutFileName = (sample_name + '_LogR_confusion_mat.png')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##Naive Bayes
    ######################
    print("\n*** Naive Bayes: ", sample_name, " ***\n")

    # Initialize classifier
    gnb = GaussianNB()

    # Train our classifier
    model = gnb.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Balanced_accuracy: ", metrics.balanced_accuracy_score(y_test, y_pred))
    print("F1_score:", metrics.f1_score(y_test, y_pred, average=None))
    print("Matthews_CC:", metrics.matthews_corrcoef(y_test, y_pred))

    #Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=y.unique(),
                          title= (sample_name + '\nNaive Bayes Confusion Matrix'))
    OutFileName = (sample_name + '_NB_confusion_mat.png')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##Random Forest
    ######################
    print("\n*** Random Forest: ", sample_name, " ***\n")

    # Initialize classifier
    RFclf = RandomForestClassifier(max_depth=2, random_state=0)

    RFclf.fit(X_train, y_train)
    y_pred = RFclf.predict(X_test)

    # QUESTION: Can we know what do these features represent?
    # Seemingly yes: https://stackoverflow.com/questions/41900387/mapping-column-names-to-random-forest-feature-importances

    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Balanced_accuracy: ", metrics.balanced_accuracy_score(y_test, y_pred))
    print("F1_score:", metrics.f1_score(y_test, y_pred, average=None))
    print("Matthews_CC:", metrics.matthews_corrcoef(y_test, y_pred))

    #Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=y.unique(),
                          title= (sample_name + '\nRandom Forest Confusion Matrix'))
    OutFileName = (sample_name + '_RF_confusion_mat.png')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

##################################################
## STEP 2.1 - PCA, t-SNE, UMAP
##################################################

def embedding_plot_string(df,X,y,labels):
    sn.scatterplot(data=df, x=X, y=y, hue=labels)

for i in SamplesToProcess_dic:
    sample_data_df = SamplesToProcess_dic[i][0]
    sample_target_df = SamplesToProcess_dic[i][1]
    sample_target_names_df = SamplesToProcess_dic[i][2]
    n_kNN_manual = SamplesToProcess_dic[i][3]
    n_components_manual = SamplesToProcess_dic[i][4]
    sample_name = namestr(sample_data_df, globals())[0]
    print("\n**** STEP 2.1: ", sample_name, " ****\n")

    ######################
    ##Scale data
    ######################
    print("\n*** Scale data: ", sample_name, " ***\n")
    X = sample_data_df
    X_norm = pd.DataFrame(preprocessing.scale(X), columns=X.columns)

    X_norm.hist(bins=50, figsize=(12,10))
    OutFileName = (sample_name + '_Norm_histograms.png')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##PCA
    ######################
    print("\n*** PCA: ", sample_name, " ***\n")

    Xnorm_pca = decomposition.PCA(n_components=X.shape[1]).fit_transform(X_norm)

    ###QUESTION: are the components sorted here by influence?
    ###or we need to sort them before to plot the top-2 components
    dfToPlot = pd.DataFrame(Xnorm_pca).iloc[:, 0:2]
    dfToPlot['Label'] = sample_target_names_df.iloc[:, 0]
    dfToPlot.columns = ('First Component', 'Second Component', 'Label')

    OutFileName = (sample_name + '_PCA_dim_red_plot.png')
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    embedding_plot_string(dfToPlot,'First Component', 'Second Component', 'Label')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##Number of components vs. variance
    ######################
    print("\n*** Number of components vs. variance: ", sample_name, " ***\n")

    pca = decomposition.PCA(n_components=Xnorm_pca.shape[1])
    pca.fit_transform(Xnorm_pca)

    plt.scatter(range(0, Xnorm_pca.shape[1]), np.cumsum(pca.explained_variance_ratio_))
    plt.gca().set_facecolor((1, 1, 1))
    plt.xlabel('components')
    plt.ylabel('variance explained')
    plt.axvline(n_components_manual, linestyle='--')
    OutFileName = (sample_name + '_PCA_knee_plot.png')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##t-SNE
    ######################
    print("\n*** t-SNE: ", sample_name, " ***\n")

    ###QUESTION: Here using n_components=2
    ###because using >4 returns error:
    ###ValueError: 'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.
    ###Should we use the number of components inferred from the PCA knee-plot?
    Xnorm_tsne = manifold.TSNE(n_components=2, init='pca',
                               perplexity=30, learning_rate=200, n_iter=500,
                               random_state=2).fit_transform(X_norm)

    dfToPlot = pd.DataFrame(Xnorm_tsne).iloc[:, 0:2]
    dfToPlot['Label'] = sample_target_names_df.iloc[:, 0]
    dfToPlot.columns = ('t-SNE 1', 't-SNE 2', 'Label')

    OutFileName = (sample_name + '_tSNE_dim_red_plot.png')
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    embedding_plot_string(dfToPlot,'t-SNE 1', 't-SNE 2', 'Label')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##UMAP
    ######################
    print("\n*** UMAP: ", sample_name, " ***\n")

    Xnorm_umap =  umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=2, random_state=2).fit_transform(X_norm)

    dfToPlot = pd.DataFrame(Xnorm_umap).iloc[:, 0:2]
    dfToPlot['Label'] = sample_target_names_df.iloc[:, 0]
    dfToPlot.columns = ('UMAP 1', 'UMAP 2', 'Label')

    OutFileName = (sample_name + '_UMAP_dim_red_plot.png')
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    embedding_plot_string(dfToPlot,'UMAP 1', 'UMAP 2', 'Label')
    plt.savefig(OutFileName, dpi=100)
    plt.close()

##################################################
## STEP 2.2 - PCA and then Regression, RF and k-NN
##################################################

### QUESTION:
### How do we take results from PCA and then implement a) Regression, b) Random forest, c) k-NN?

##################################################
## STEP 3 - UMAP and map clustering results
##################################################

########
for i in SamplesToProcess_dic:
    sample_data_df = SamplesToProcess_dic[i][0]
    sample_target_df = SamplesToProcess_dic[i][1]
    sample_target_names_df = SamplesToProcess_dic[i][2]
    n_kNN_manual = SamplesToProcess_dic[i][3]
    n_components_manual = SamplesToProcess_dic[i][4]
    sample_name = namestr(sample_data_df, globals())[0]

    print("\n**** STEP 3: ", sample_name, " ****\n")

    ######################
    ##Scale data
    ######################
    print("\n*** Scale data: ", sample_name, " ***\n")
    X = sample_data_df
    X_norm = pd.DataFrame(preprocessing.scale(X), columns=X.columns)

    ######################
    ##UMAP for clustering
    ######################
    print("\n*** UMAP for clustering: ", sample_name, " ***\n")

    Xnorm_umap_df =  pd.DataFrame(umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=2, random_state=2).fit_transform(X_norm))

    ######################
    ##k-means clustering
    ######################
    print("\n*** k-means clustering: ", sample_name, " ***\n")

    HP1 = (2,4,8,10) #Hyper parameter 1 (k)

    KmeansResults_dic = {}
    for hp1 in HP1:
        KmeansResults_dic[hp1] = cluster.KMeans(n_clusters=hp1, max_iter=300).fit(X_norm)

    OutFileName = (sample_name + '_UMAP_kMeans.png')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    fig = plt.gcf()
    fig.set_size_inches(16, 4)
    ax1.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=KmeansResults_dic[HP1[0]].labels_)
    subplot_title = ("k = ", HP1[0])
    ax1.set_title(subplot_title)
    ax2.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=KmeansResults_dic[HP1[1]].labels_)
    subplot_title = ('k = ', HP1[1])
    ax2.set_title(subplot_title)
    ax3.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=KmeansResults_dic[HP1[2]].labels_)
    subplot_title = ('k = ', HP1[2])
    ax3.set_title(subplot_title)
    ax4.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=KmeansResults_dic[HP1[3]].labels_)
    subplot_title = ('k = ', HP1[3])
    ax4.set_title(subplot_title)
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##Agglomerative clustering
    ######################
    print("\n*** Agglomerative clustering: ", sample_name, " ***\n")

    HP1 = (2, 4, 8, 10)  # Hyper parameter 1 (k)

    AgglomerativeResults_dic = {}
    for hp1 in HP1:
        AgglomerativeResults_dic[hp1] = cluster.AgglomerativeClustering(n_clusters=hp1,
                                                                        affinity='euclidean',
                                                                        linkage='complete').fit(X_norm)

    OutFileName = (sample_name + '_UMAP_Agglomerative.png')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    fig = plt.gcf()
    fig.set_size_inches(16, 4)
    ax1.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=AgglomerativeResults_dic[HP1[0]].labels_)
    subplot_title = ("k = ", HP1[0])
    ax1.set_title(subplot_title)
    ax2.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=AgglomerativeResults_dic[HP1[1]].labels_)
    subplot_title = ('k = ', HP1[1])
    ax2.set_title(subplot_title)
    ax3.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=AgglomerativeResults_dic[HP1[2]].labels_)
    subplot_title = ('k = ', HP1[2])
    ax3.set_title(subplot_title)
    ax4.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=AgglomerativeResults_dic[HP1[3]].labels_)
    subplot_title = ('k = ', HP1[3])
    ax4.set_title(subplot_title)
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##Affinity Propagation clustering
    ######################
    print("\n*** Affinity Propagation clustering: ", sample_name, " ***\n")

    HP1 = (100, 200, 300, 400)  # Hyper parameter 1 max_iter

    AffPropagationResults_dic = {}
    for hp1 in HP1:
        AffPropagationResults_dic[hp1] = cluster.AffinityPropagation(max_iter=hp1, convergence_iter=15).fit(X_norm)

    OutFileName = (sample_name + '_UMAP_AffPropagation.png')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    fig = plt.gcf()
    fig.set_size_inches(16, 4)
    ax1.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=AffPropagationResults_dic[HP1[0]].labels_)
    subplot_title = ("max_iter = ", HP1[0])
    ax1.set_title(subplot_title)
    ax2.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=AffPropagationResults_dic[HP1[1]].labels_)
    subplot_title = ('max_iter = ', HP1[1])
    ax2.set_title(subplot_title)
    ax3.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=AffPropagationResults_dic[HP1[2]].labels_)
    subplot_title = ('max_iter = ', HP1[2])
    ax3.set_title(subplot_title)
    ax4.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=AffPropagationResults_dic[HP1[3]].labels_)
    subplot_title = ('max_iter = ', HP1[3])
    ax4.set_title(subplot_title)
    plt.savefig(OutFileName, dpi=100)
    plt.close()

    ######################
    ##DBSCAN clustering
    ######################
    print("\n*** DBSCAN clustering: ", sample_name, " ***\n")

    HP1 = (3, 5, 7, 10)  # Hyper parameter 1 min_samples

    DBSCANResults_dic = {}
    for hp1 in HP1:
        DBSCANResults_dic[hp1] = cluster.DBSCAN(eps=0.5, min_samples=hp1).fit(X_norm)

    OutFileName = (sample_name + '_UMAP_DBSCAN.png')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    fig = plt.gcf()
    fig.set_size_inches(16, 4)
    ax1.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=DBSCANResults_dic[HP1[0]].labels_)
    subplot_title = ("min_samples = ", HP1[0])
    ax1.set_title(subplot_title)
    ax2.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=DBSCANResults_dic[HP1[1]].labels_)
    subplot_title = ('min_samples = ', HP1[1])
    ax2.set_title(subplot_title)
    ax3.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=DBSCANResults_dic[HP1[2]].labels_)
    subplot_title = ('min_samples = ', HP1[2])
    ax3.set_title(subplot_title)
    ax4.scatter(Xnorm_umap_df.iloc[:, 0], Xnorm_umap_df.iloc[:, 1], c=DBSCANResults_dic[HP1[3]].labels_)
    subplot_title = ('min_samples = ', HP1[3])
    ax4.set_title(subplot_title)
    plt.savefig(OutFileName, dpi=100)
    plt.close()
