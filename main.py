import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df_chuncks = pd.read_csv('2015.csv', chunksize=10000)
for chunck in df_chuncks:
    df = pd.concat([chunck])
df = df.select_dtypes('number')
print(df.memory_usage())
#df = df.loc[df['_RFHLTH'].isin([1, 2])].copy()
df = df.rename(columns={'_RFHLTH': 'label'})
df = df.drop(columns=['POORHLTH', 'PHYSHLTH', 'GENHLTH', 'PAINACT2',
                        'QLMENTL2', 'QLSTRES2', 'QLHLTH2', 'HLTHPLN1', 'MENTHLTH'])

labels = np.array((df.pop('label')))


train, test, train_labels, test_labels = train_test_split(df, labels,
                                                          stratify=labels,
                                                          test_size=0.3)
train = train.fillna(0)
test = test.fillna(0)



# Features for feature importances
features = list(train.columns)

model = RandomForestClassifier(max_features='sqrt', n_jobs=-1, verbose=1)
model.fit(train, train_labels)

max_depths = []
n_nodes = []
for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

# Actual class predictions
rf_predictions = model.predict(test)
# Probabilities for each class
rf_probs = model.predict_proba(test)[:, 1]


accuracy = (rf_predictions == test_labels).sum()/len(test_labels)
print("Accuracy: {}".format(accuracy))

# Calculate roc auc
roc_value = roc_auc_score(test_labels, rf_probs)

fi_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi_model.head(10)
