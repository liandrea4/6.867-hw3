import generate_datasets
import numpy        as np
import matplotlib.pyplot as plt
from sklearn        import neural_network


all_datasets = generate_datasets.make_all_datasets()
normalized_datasets = generate_datasets.normalize_datasets(all_datasets)


x_training, y_training = [], []
x_validate, y_validate = [], []
x_training_validate, y_training_validate = [], []
x_testing, y_testing = [], []

for key in normalized_datasets.keys():
  for datasets in normalized_datasets[key]:
    training, validation, testing = datasets[0], datasets[1], datasets[2]

    x_training.append(training)
    y_training.append(int(key))

    x_validate.append(validation)
    y_validate.append(int(key))

    x_testing.append(testing)
    y_testing.append(int(key))


for elem, key in zip(x_training, y_training):
  x_training_validate.append(elem)
  y_training_validate.append(int(key))

for elem, key in zip(x_validate, y_validate):
  x_training_validate.append(elem)
  y_training_validate.append(int(key))


architectures = [(5), (5,5), (5,5,5), (5,5,5,5), (5,5,5,5,5), (5,5,5,5,5,5), (5,5,5,5,5,5,5)]
# architectures = [(5), (100), (5,5), (100, 100), (10), (10,10)]
# architectures = [(10), (50), (100), (500), (1000)]
learning_rates = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
training_rates = {}
testing_rates = {}
num_repeats = 20
testing_rates_list = []

for learning_rate in learning_rates:
  total_training_score = 0
  total_testing_score = 0

  for i in range(num_repeats):
    classifier = neural_network.MLPClassifier((1000,1000,1000), max_iter=500, early_stopping=True, solver='sgd', alpha=0, learning_rate_init=learning_rate, validation_fraction=4./7)
    classifier.fit(x_training_validate, y_training_validate)
    total_testing_score += classifier.score(x_testing, y_testing)
    total_training_score += classifier.score(x_training_validate, y_training_validate)

  testing_rates_list.append(total_testing_score / float(num_repeats))

print "testing score: ", testing_rates_list

print "plotting..."
plt.figure()
plt.plot(learning_rates, testing_rates_list)
plt.xscale('log')
plt.title('Testing accuracy vs. learning rate, n=1000, l=3', fontsize=16)
plt.xlabel('Learning rate', fontsize=14)
plt.ylabel('Testing accuracy', fontsize=14)
plt.show()




