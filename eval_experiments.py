import os
import numpy as np
import keras
from keras.datasets import cifar10, cifar100
from math import floor
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns


dataset = 'cifar10'
# Digest must be first and aux second for t-test below to work.
conditions = ['digest', 'aux', 'avg', 'skip', 'baseline']

model_dir_base = '/home/arnaldop/Work/keyboard_nn/saved_models_'
model_dir = model_dir_base + dataset

"""
Load data
"""
if dataset == 'cifar10':
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_samples = x_train.shape[0]
elif dataset == 'cifar100':
    num_classes = 100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    train_samples = x_train.shape[0]
else:
    num_classes = 1
    (x_train, y_train), (x_test, y_test) = (None, None), (None, None)
    raise ValueError("Unsupported dataset.")

# Reshape labels to one-hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Check shape is channels_last
print("Training data shape: {}".format(x_train.shape))
print("Test data shape: {}".format(x_test.shape))
# Normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("Loaded data for {}.".format(dataset.upper()))


def do_inference(model, x_test, y_test, cond):
    # Return the correct accuracy value
    if cond == 'digest':
        scores = model.evaluate(x_test, [y_test] * 4, verbose=0)
        return scores[5]
    elif cond == 'avg':
        scores = model.evaluate(x_test, [y_test] * 4, verbose=0)
        return scores[5]
    elif cond == 'aux':
        scores = model.evaluate(x_test, [y_test] * 3, verbose=0)
        return scores[4]
    elif cond == 'skip':
        scores = model.evaluate(x_test, y_test, verbose=0)
        return scores[1] # Should be
    elif cond == 'baseline':
        scores = model.evaluate(x_test, y_test, verbose=0)
        return scores[1]
    else:
        raise ValueError("Wrong condition.")

def bootstrap_inference(model, x_test, y_test, cond, n=10, split=0.8):
    accuracies = []
    n_samples = x_test.shape[0]
    all_idxs = np.linspace(0, n_samples - 1,
                           n_samples, dtype=int)
    split_int = floor(split * n_samples)
    print(("Calculating metrics for condition {} " + 
          "with bootstrap n={} and split={}").format(cond.upper(),
                                                    n,
                                                    split))
    for i in range(n):
        idxs = np.random.choice(all_idxs,
                                size=split_int,
                                replace=False)
        accuracy = do_inference(model, x_test[idxs], y_test[idxs], cond)
        accuracies.append(accuracy)
        percent = ((i + 1) / n) * 100
        print("[{}{}]\t{:.1f}%".format('==' * (i + 1) + '>',
                                       '  ' * (n - (i + 1)),
                                       percent),
               end='\r')
        if percent == 100:
            print()
    mean_acc = np.mean(accuracies)
    std_dev = np.std(accuracies)
    print("Condition {} mean accuracy: {:.4f}".format(cond.upper(),
                                                      mean_acc))
    print("Condition {} std deviation: {:.8f}".format(cond.upper(),
                                                      std_dev))
    return accuracies


"""
Inference loop --- bootstrapped
"""
all_accuracies = []
for cond in conditions:
    # Load pretrained model
    model_fname = 'model_' + cond + '.hdf5'
    m = os.path.join(model_dir, model_fname)
    model = keras.models.load_model(m)
    print("Loaded model {} to evaluate on test data.".format(model_fname))

    # Model summary
    print("Model summary:")
    model.summary()

    accuracies = bootstrap_inference(model,
                                     x_test,
                                     y_test,
                                     cond,
                                     n=50,
                                     split=0.7)
    all_accuracies.append(accuracies)  

"""
Calculate Student's t-test statistic for
DIGEST against each of the other conditions
"""
for i, cond in enumerate(conditions[1:], start=1):
    print("p-value for t-test of DIGEST vs {}".format(cond.upper()))
    _, p = ttest_ind(all_accuracies[0], all_accuracies[i])
    print("p-value: {}".format(p))

"""
Calculate Student's t-test statistic for
AUX against each of the other conditions
(except DIGEST, which is already tested above)
"""
for i, cond in enumerate(conditions[2:], start=2):
    print("p-value for t-test of AUX vs {}".format(cond.upper()))
    _, p = ttest_ind(all_accuracies[1], all_accuracies[i])
    print("p-value: {}".format(p))

"""
Plot metrics.
"""
labels = [cond.upper() for cond in conditions]
data = all_accuracies
sns.set_context('paper')
bp = sns.boxplot(x=labels, y=data)
plt.show()
