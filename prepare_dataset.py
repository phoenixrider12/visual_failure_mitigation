import torch
from dataset import taxinet_dataset, shuffle

data_dirs = ['9_0_KMWH/', '9_4_KMWH/', '21_0_KMWH/', '21_4_KMWH/', '9_0_KATL/', '9_4_KATL/', '21_0_KATL/', '21_4_KATL/', '9_0_PAEI/', '9_4_PAEI/', '21_0_PAEI/', '21_4_PAEI/']
root_dir = 'dataset/'

data_dir1 = root_dir + data_dirs[0]
data_dir2 = root_dir + data_dirs[1]

image_tensor1, labels1 = taxinet_dataset(data_dir1)
image_tensor2, labels2 = taxinet_dataset(data_dir2)

image_tensor = torch.cat((image_tensor1, image_tensor2), dim = 0)
labels = torch.cat((labels1, labels2), dim = 0)

del image_tensor1, image_tensor2, labels1, labels2

for i in range(2, len(data_dirs)):

    data_dir = root_dir + data_dirs[i]
    image_tensor3, labels3 = taxinet_dataset(data_dir)

    image_tensor = torch.cat((image_tensor, image_tensor3), dim = 0)
    labels = torch.cat((labels, labels3), dim = 0)

    del image_tensor3, labels3

image_tensor, labels = shuffle(image_tensor, labels)
print(image_tensor.shape)
print(labels.shape)

torch.save(image_tensor, 'taxinet_dataset.pt')
torch.save(labels, 'taxinet_labels.pt')