import torch
import pandas
from torchvision import transforms
from PIL import Image
import random
import numpy as np

def taxinet_dataset(data_dir):

    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224


    tfms = transforms.Compose([transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5],
                                                    [0.5, 0.5, 0.5])
                          ])

    image_list = [str(i) + '.png' for i in range(20000)]
    label_file = data_dir + 'labels.csv'
 
    labels_df = pandas.read_csv(label_file, sep=',')

    image_tensor_list = []
    target_tensor_list = []

    for i, image_name in enumerate(image_list):
        if i%5000 == 0:
            print(i)
        fname = data_dir + '/' + str(image_name)
        image = Image.open(fname).convert('RGB')
        tensor_image_example = tfms(image)

        image_tensor_list.append(tensor_image_example)

        cte = labels_df['cte'][i]
        dtp = labels_df['dtp'][i]
        he = labels_df['he'][i]
        value = labels_df['values'][i]
        label = labels_df['labels'][i]

        target_tensor_list.append([cte, dtp, he, value, label])
        
    all_image_tensor = torch.stack(image_tensor_list)
    print(all_image_tensor.shape)

    target_tensor = torch.tensor(target_tensor_list)
    print(target_tensor.shape)

    return all_image_tensor, target_tensor

def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]