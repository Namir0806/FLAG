import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
def get_label(x):
    '''
    if x>= 0 and x<0.56:
        #small positive
        return 2
    elif x>= 0.56:
        #big positive
        return 3
    elif x<0 and x>= -0.56:
        #small negative
        return 1
    elif x<-0.56:
        #big negative
        return 0
    '''
    if x>=0:
        return 1
    else:
        return 0
'''
#no-hist
sec = sys.argv[1]

df = pd.read_csv('../new-all-results.csv')

bv = sys.argv[2]

hist = sys.argv[3]


train_index, rem_index, train_labels, rem_labels = train_test_split(list(range(len(df[bv]))),
    df[bv],
    random_state=42,
    train_size=0.8)
valid_index, test_index, valid_labels, test_labels = train_test_split(rem_index, 
    rem_labels, 
    shuffle=False,
    train_size=0.5)
test_preds = np.loadtxt('new-GATv2-MSE-results/test_preds_GATv2_4_layers_4_heads_call_daily_price_change_eps5_lr=1.0e-05_plan-C_no-hist.csv', delimiter=",", dtype=float)
test_labels = np.asarray(test_labels)
y_pred = []
y_true = []
print(type(test_preds))
print(type(test_labels))
print(len(test_preds))
print(len(test_labels))
print(np.std(test_preds))
print(np.std(test_labels))
for i in range(len(test_preds)):
    y_pred.append(get_label(test_preds[i]))
    y_true.append(get_label(test_labels[i]))
print(type(test_preds))
print(type(test_labels))
print(accuracy_score(y_true,y_pred))

#with-hist

sec = sys.argv[1]

df = pd.read_csv('../new-all-results.csv')

bv = sys.argv[2]

hist = sys.argv[3]

train_index, rem_index, train_labels, rem_labels ,train_hist, rem_hist= train_test_split(list(range(len(df['post_'+bv]))),
    df['post_'+bv],
    df['prev_'+bv],
    random_state=42,
    train_size=0.8)
valid_index, test_index, valid_labels, test_labels, valid_hist, test_hist = train_test_split(rem_index, 
    rem_labels, 
    rem_hist,
    shuffle=False,
    train_size=0.5)

test_preds = np.loadtxt('new-GATv2-MSE-results/test_preds_GATv2_4_layers_4_heads_call_day_price_eps5_lr=1.0e-03_plan-C_with-hist.csv', delimiter=",", dtype=float)
test_labels = np.asarray(test_labels)
test_hist = np.asarray(test_hist)
test_pred_change = []
test_true_change = []
for i in range(len(test_preds)):
    test_pred_change.append((test_preds[i]-test_hist[i])/test_hist[i])
    test_true_change.append((test_labels[i]-test_hist[i])/test_hist[i])
y_pred = []
y_true = []
print(type(test_preds))
print(type(test_labels))
print(len(test_preds))
print(len(test_labels))
print(np.std(test_pred_change))
print(np.std(test_true_change))
for i in range(len(test_preds)):
    y_pred.append(get_label(test_pred_change[i]))
    y_true.append(get_label(test_true_change[i]))
print(type(test_preds))
print(type(test_labels))
print(accuracy_score(y_true,y_pred))


#non-random-train-test-split-new-GATv2
#with hist

sec = sys.argv[1]

df = pd.read_csv('../new-2019-result.csv')

bv = sys.argv[2]

hist = sys.argv[3]


test_preds = np.loadtxt('non-random-train-test-split-new-GATv2/new-GATv2-MSE-results/test_preds_GATv2_4_layers_4_heads_call_day_price_eps5_lr=1.0e-04_plan-C_with-hist.csv', delimiter=",", dtype=float)
test_labels = np.asarray(df['post_'+bv])
test_hist = np.asarray(df['prev_'+bv])
test_pred_change = []
test_true_change = []
for i in range(len(test_preds)):
    test_pred_change.append((test_preds[i]-test_hist[i])/test_hist[i])
    test_true_change.append((test_labels[i]-test_hist[i])/test_hist[i])
y_pred = []
y_true = []
print(type(test_preds))
print(type(test_labels))
print(len(test_preds))
print(len(test_labels))
print(np.std(test_pred_change))
print(np.std(test_true_change))
for i in range(len(test_preds)):
    y_pred.append(get_label(test_pred_change[i]))
    y_true.append(get_label(test_true_change[i]))
print(type(test_preds))
print(type(test_labels))
print(accuracy_score(y_true,y_pred))




'''



#non-random-train-test-split-new-GATv2
#no hist


sec = sys.argv[1]

df = pd.read_csv('../../new-tech-2019-result.csv')

bv = sys.argv[2]

hist = sys.argv[3]


test_preds = np.loadtxt('mse-results/test_preds_GATv2_4_layers_12_heads_tech-earnings-calls_daily_price_change_eps10_lr=9.0e-05_plan-E_no-hist.csv', delimiter=",", dtype=float)
test_labels = np.asarray(df[bv])
y_pred = []
y_true = []
print(type(test_preds))
print(type(test_labels))
print(len(test_preds))
print(len(test_labels))
print(np.std(test_preds))
print(np.std(test_labels))
for i in range(len(test_preds)):
    y_pred.append(get_label(test_preds[i]))
    y_true.append(get_label(test_labels[i]))
print(type(test_preds))
print(type(test_labels))
print(accuracy_score(y_true,y_pred))
print(classification_report(y_true,y_pred, digits=8))
