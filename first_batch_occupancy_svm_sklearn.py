#%%
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import time
custom_classifier1 = SGDClassifier(learning_rate='constant', eta0=0.01)
custom_classifier2 = PassiveAggressiveClassifier()
custom_classifier3 = Perceptron()
clf_dict =  {'svm':MultiOutputClassifier(custom_classifier1),
             'passive_aggressive': MultiOutputClassifier(custom_classifier2),
             'perceptron': MultiOutputClassifier(custom_classifier3)
                }
#%%
training_accuracy = {'svm':[],
                     'passive_aggressive': [],
                     'perceptron': []
                    }
for i, batch in enumerate(train_loader):
    #TODO: when have CUDA:
    #x_batch = batch['imagePower'].to(device)
    #y_batch = batch['label'].to(device)
    
    x_batch = batch['imagePower'].detach().cpu().numpy()
    x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1]*x_batch.shape[2]*x_batch.shape[3])
    y_batch = batch['label'].detach().cpu().numpy()
    for clf in clf_dict:
        clf_dict[clf].partial_fit(x_batch, y_batch, classes=np.array([[0, 1]] * int(y_batch.shape[1])))
        y_pred = clf_dict[clf].predict(x_batch)
        try:
            training_accuracy[clf].append(clf_dict[clf].score(x_batch, y_batch))
        except:
            training_accuracy[clf] = list()
            training_accuracy[clf].append(clf_dict[clf].score(x_batch, y_batch))
# save the model to disk
for clf in clf_dict:
    filename = f'{clf}_FirstBatch.pickle'
    pickle.dump(clf_dict[clf], open(filename, 'wb'))
print("finished training")

#%% 
from sklearn.metrics import accuracy_score
val_accuracy = {'svm':[],
                'passive_aggressive': [],
                'perceptron': []
                }
for clf in clf_dict:
    print(f'{clf}_misclassified_firstBatch.csv')
    f = open(f'{clf}_misclassified_firstBatch.csv', 'w')
    f.write(','.join(['path', 'label_seat', 'predicted_seat', 'label_type', 'predicted_type\n']))
    f.close()
for i, batch in enumerate(val_loader):
    x_batch = batch['imagePower'].detach().cpu().numpy()
    x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1]*x_batch.shape[2]*x_batch.shape[3])
    y_batch = batch['label'].detach().cpu().numpy()
    path = np.array(batch['path'])        
    for clf in clf_dict:
        loaded_model = pickle.load(open(f'{clf}_FirstBatch.pickle', 'rb'))
        misclassified_dict = dict()
        misclassified_dict = {
            'path': [],
            'label_seat': [],
            'predicted_seat':[],
            'label_type': [],
            'predicted_type': []
        }
        loaded_model.partial_fit(x_batch, y_batch, classes=np.array([[0, 1]] * int(y_batch.shape[1])))
        y_pred = loaded_model.predict(x_batch)
        misclassified_indice = np.where((y_pred!=y_batch).any(1))
        if len(misclassified_indice[0]) != 0:
            misclassified_dict['path'] = list(path[misclassified_indice])
            misclassified_dict['predicted_seat'], misclassified_dict['predicted_type'] = scenarioWiseTransformLabels(y_pred[misclassified_indice])
            misclassified_dict['label_seat'], misclassified_dict['label_type'] = scenarioWiseTransformLabels(y_batch[misclassified_indice])
            df = pd.DataFrame.from_dict(misclassified_dict)
            df.to_csv(f'{clf}_misclassified_firstBatch.csv', mode='a', header=False, index=False)
        try:
            val_accuracy[clf].append(accuracy_score(y_pred, y_batch))
        except:
            training_accuracy[clf] = list()
            val_accuracy[clf].append(accuracy_score(y_pred, y_batch))
# %%
for classifier in val_accuracy:
    acc = np.average(np.array(val_accuracy[classifier]))
    print('The {} validation accuracy is {}.'.format(classifier, acc))
for classifier in training_accuracy:
    acc = np.average(np.array(training_accuracy[classifier]))
    print('The {} training accuracy is {}.'.format(classifier, acc))
