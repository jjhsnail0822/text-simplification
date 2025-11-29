#%%
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
LEVEL_ORDER = {
            "en": {
                "A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5
            },
            "ja": {
                "N5": 0, "N4": 1, "N3": 2, "N2": 3, "N1": 4
            },
            "ko": {
                "TOPIK1": 0, "TOPIK2": 1, "TOPIK3": 2, "TOPIK4": 3, "TOPIK5": 4, "TOPIK6": 5
            },
            "zh": {
                "HSK1": 0, "HSK2": 1, "HSK3": 2, "HSK4": 3, "HSK5": 4, "HSK6": 5, "HSK7-9": 6
            }
        }
HALF_LEVEL_ORDER = {
            "en": {
                "A":0,"B":1,"C":2
            },
            "ja": {
                "N54": 0, "N32": 1, "N1": 2
            },
            "ko": {
                "TOPIK12": 0, "TOPIK34": 1, "TOPIK56": 2
            },
            "zh": {
                "HSK12": 0, "HSK34": 1, "HSK56": 2, "HSK7-9": 3
            }
        }
def num_to_level(lang,num):
    for key in LEVEL_ORDER[lang].keys():
        if LEVEL_ORDER[lang][key] == num: return key
def num_to_half_level(lang,num):
    for key in HALF_LEVEL_ORDER[lang].keys():
        if HALF_LEVEL_ORDER[lang][key] == num: return key
def sorted_keys(lang):
    return sorted(list(LEVEL_ORDER[lang].keys()),key=lambda x: LEVEL_ORDER[lang][x])
def half_sorted_keys(lang):
    return sorted(list(HALF_LEVEL_ORDER[lang].keys()),key=lambda x: HALF_LEVEL_ORDER[lang][x])
def create_sklearn_datasets():
    with open('data/frequencies/frequency_data.json') as f:
        raw_data = json.load(f)
    data= dict()
    for lang in LEVEL_ORDER.keys():
        data[lang]={'X':[],'y':[]}
        for item in raw_data[lang]:
            data[lang]['X'].append((item['frequency'],))
            data[lang]['y'].append(LEVEL_ORDER[lang][item['level']])
    return data
def fit(lang,data):
    X_train,X_test,y_train,y_test = train_test_split(np.array(data['X']),np.array(data['y']),test_size=0.1,random_state=42)
    X_train_visual = (x[0] for x in X_train)
    y_train_visual = (num_to_level(lang,y) for y in y_train)
    plot = sns.displot(x=X_train_visual,hue=y_train_visual,log_scale=True,kind='ecdf',hue_order=sorted(list(LEVEL_ORDER[lang].keys())))
    plot.figure.savefig(f"data/frequencies/histogram_{lang}.png")
    transformer = FunctionTransformer(np.log1p)
    scaler = StandardScaler()
    X_train = transformer.transform(X_train)
    scaler.fit(X_train)
    predictor = LogisticRegression(random_state=42,max_iter=10000,class_weight='balanced').fit(X_train,y_train)

    print(f'{lang} accuracy: '+ str(predictor.score(X_train,y_train)))
    predictions = predictor.predict(X_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train,predictions),display_labels=sorted_keys(lang))
    disp.plot()
    savefig(f'data/frequencies/confusion_matrix_{lang}.png')
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train,predictions,normalize='true'),display_labels=sorted_keys(lang))
    disp.plot()
    savefig(f'data/frequencies/normalized_confusion_matrix_{lang}.png')
    plt.close()
def fit_half_labels(lang,data):
    X_train,X_test,y_train,y_test = train_test_split(np.array(data['X']),np.array(data['y']),test_size=0.1,random_state=42)
    y_train //= 2
    X_train_visual = (x[0] for x in X_train)
    y_train_visual = (num_to_half_level(lang,y) for y in y_train)
    plot = sns.displot(x=X_train_visual,hue=y_train_visual,log_scale=True,kind='ecdf',hue_order=sorted(list(HALF_LEVEL_ORDER[lang].keys())))
    plot.figure.savefig(f"data/frequencies/half_label_histogram_{lang}.png")
    transformer = FunctionTransformer(np.log1p)
    scaler = StandardScaler()
    X_train = transformer.transform(X_train)
    scaler.fit(X_train)
    predictor = LogisticRegression(random_state=42,max_iter=10000,class_weight='balanced').fit(X_train,y_train)

    print(f'half label {lang} accuracy: ',str(predictor.score(X_train,y_train)))
    predictions = predictor.predict(X_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train,predictions),display_labels=half_sorted_keys(lang))
    disp.plot()
    savefig(f'data/frequencies/half_label_confusion_matrix_{lang}.png')
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train,predictions,normalize='true'),display_labels=half_sorted_keys(lang))
    disp.plot()
    savefig(f'data/frequencies/half_label_normalized_confusion_matrix_{lang}.png')
    plt.close()
if __name__ == '__main__':
    for lang in ['en','ja','ko','zh']:
        fit(lang,create_sklearn_datasets()[lang])
        fit_half_labels(lang,create_sklearn_datasets()[lang])
# %%
