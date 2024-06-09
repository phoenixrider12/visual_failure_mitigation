import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import math

TIME_OF_DAY = 17
CLOUD_CONDITION = 0
RUNWAY = 'KATL'

tfms = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5],
                                                    [0.5, 0.5, 0.5])
                          ])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_weights = 'taxinet_classifier_weights.pth'
test_case = str(TIME_OF_DAY) + '_' + str(CLOUD_CONDITION) + '_' + RUNWAY

data_dir = 'dataset/' + test_case + '/'
labels_file = data_dir + 'labels.csv'

df = pd.read_csv(labels_file)
cte = df['cte']
dtp = df['dtp']
he = df['he']
values = df['values']
labels = df['labels']
states = np.transpose(np.stack((cte, dtp, he)))

model = torch.load(train_weights)
model = model.to(device)
model.eval()

targets = []
predictions = []
new_values = []
probs = []

for i in range(20000):

    if i%1000 == 0:
        print(i)
    img = str(i) + '.png'
    img = Image.open(data_dir + img).convert('RGB')
    inputs = tfms(img)
    inputs = torch.unsqueeze(inputs, dim = 0)
    inputs = inputs.to(device)
    label = labels[i]
    value = values[i]

    with torch.no_grad():
        outputs = model(inputs)
    outputs = torch.squeeze(outputs)
    softmax = (torch.exp(outputs)/torch.exp(outputs).sum()).cpu()
    prob = list(softmax.detach().numpy())
    prediction = np.argmax(prob)
    predictions.append(prediction)
    targets.append(label)

    if abs(value) < 5:
        new_values.append(value)
        probs.append(prob)

predictions = np.array(predictions)
targets = np.array(targets)

tn, fp, fn, tp = confusion_matrix(predictions, targets).ravel()
tp = tp * 100 / len(labels)
fp = fp * 100 / len(labels)
tn = tn * 100 / len(labels)
fn = fn * 100 / len(labels)

print('TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn)
print('Recall: ', recall_score(predictions, targets))
print('Accuracy: ', accuracy_score(predictions, targets))

incorrect_ctp=[]
incorrect_dtp=[]
incorrect_heading=[]
correct_ctp = []
correct_dtp = []
correct_heading = []

for i in range(20000):

    if predictions[i] != targets[i]:
        incorrect_ctp.append(states[i][0])
        incorrect_dtp.append(states[i][1])
        incorrect_heading.append(states[i][2] * 180 / math.pi)

    if predictions[i] == targets[i]:
        correct_ctp.append(states[i][0])
        correct_dtp.append(states[i][1])
        correct_heading.append(states[i][2] * 180 / math.pi)

plt.title('CTE vs Heading')
plt.xlabel('CTE')
plt.ylabel('Heading')
plt.scatter(correct_ctp, correct_heading, c = 'g')
plt.scatter(incorrect_ctp, incorrect_heading, c = 'r')

plt.show()
