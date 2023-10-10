import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torch.optim import Adam
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

image_tensor = torch.load('taxinet_dataset.pt')
labels = torch.load('taxinet_labels.pt')

print(image_tensor.shape)
print(labels.shape)

states = labels[:, :3]
values = labels[:, 3]
targets = labels[:, 4].long()

train_images, test_images, train_labels, test_labels = train_test_split(image_tensor, targets, test_size=0.2, random_state=42)

train_set = TensorDataset(train_images, train_labels)
test_set = TensorDataset(test_images, test_labels)

trainloader = DataLoader(train_set, batch_size=256, shuffle = True)
testloader = DataLoader(test_set, batch_size=256, shuffle = True)

model = torchvision.models.efficientnet_b0(pretrained = True)
for params in model.parameters():
  params.requires_grad_ = False
last_input = model.classifier[1].in_features
model.classifier[1] = Linear(last_input, 2)

optimizer = Adam(model.parameters(), lr=3e-4)
criterion = CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

batch_size = 256
n_epochs = 20

total_loss = []
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    training_loss = []

    for i, data in enumerate(trainloader):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, label)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step() 

    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
    total_loss.append(training_loss)

torch.save(model, 'taxinet_classifier_weights.pth')

batch_size = 256
train_prediction = []
target = []

for i, data in enumerate(trainloader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        output = model(inputs)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    train_prediction.append(predictions)
    target.append(labels)

train_target_list = torch.cat(target).cpu().numpy()
train_prediction_list = np.concatenate(train_prediction)

print('Train Accuracy: \t', accuracy_score(train_target_list, train_prediction_list))
print('Train Accuracy: \t', recall_score(train_target_list, train_prediction_list))

test_prediction = []
target = []

for i, data in enumerate(testloader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        output = model(inputs)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    test_prediction.append(predictions)
    target.append(labels)

test_target_list = torch.cat(target).cpu().numpy()
test_prediction_list = np.concatenate(test_prediction)

print('Train Accuracy: \t', accuracy_score(test_target_list, test_prediction_list))
print('Train Accuracy: \t', recall_score(test_target_list, test_prediction_list))