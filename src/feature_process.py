from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns
from .preprocess import preprocess_data
from sklearn.model_selection import train_test_split
import numpy as np


def feature_process():
    preprocessed_data = preprocess_data()

    X=preprocessed_data.drop('SepsisLabel', axis=1)
    y=preprocessed_data['SepsisLabel']

    train_set, valid_set, train_labels, valid_labels = train_test_split(X, y, test_size=0.33, random_state=42)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(train_set, train_labels)

    predictions = rf_clf.predict(valid_set)

#print("Train set score: ", rf_clf.score(train_set, train_labels))
#print("Test set score: ", rf_clf.score(valid_set, valid_labels))

#print("Confusion matrix: ")
#cm = confusion_matrix(valid_labels, predictions,  labels = [1, 0])
#print(cm)

    feature_impt = pd.DataFrame(rf_clf.feature_importances_, index=train_set.columns).sort_values(by=0, 
                                                                                              ascending=False)

    impt_features = feature_impt.loc[feature_impt[0] > 0.005]
    impt_features_list = list(impt_features.index.values)

    non_impt_features = feature_impt.loc[feature_impt[0] <= 0.005]
    non_impt_features_list = list(non_impt_features.index.values)


    #Create arrays from feature importance and feature names
    feature_importance = np.array(rf_clf.feature_importances_)
    feature_names = np.array(train_set.columns)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    # important features
    fi_df = fi_df.loc[fi_df['feature_importance'] > 0.005]

    #Define size of bar plot
    #plt.figure(figsize=(15,10))
    #Plot Searborn bar chart
    #sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    #plt.title(model_type + ' FEATURE IMPORTANCE')
    #plt.xlabel('FEATURE IMPORTANCE')
    #plt.ylabel('FEATURE NAMES')

#plot_feature_importance(rf_clf.feature_importances_,train_set.columns,'RANDOM FOREST')

    train_set = train_set.drop(non_impt_features_list)
    valid_set = valid_set.drop(non_impt_features_list)

    return train_set, valid_set, train_labels, valid_labels

if __name__ == "__main__":
    imp_feaures_data = feature_process()