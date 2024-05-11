from PIL import Image
import os
from torchvision import transforms
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


data_folder = os.path.join("Extracted Faces", "Extracted Faces")

image_data = []
for each_person in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, each_person)

    for each_img in os.listdir(folder_path):
        img_path = os.path.join(folder_path,  each_img)

        img = Image.open(img_path)

        img = transform(img)

        image_data.append([img, int(each_person)])


print("Data loaded")

train_data_size = 1000
test_data_size = 100

diff = 0
diff2 = 0
same = 0
same2 = 0
k = 0
k2 = 0
img_dim = 224
n_channels = 3

train_inputs = torch.empty(train_data_size, 2*n_channels, img_dim, img_dim)
train_labels = torch.empty(train_data_size)

test_inputs = torch.empty(test_data_size,  2*n_channels, img_dim, img_dim)
test_labels = torch.empty(test_data_size)

while same + same2 != int(train_data_size / 2 + test_data_size / 2):
    index1 = np.random.randint(len(image_data))
    gap = np.random.randint(1, 5)
    index2 = index1 + gap

    if index2 < len(image_data) and image_data[index1][1] == image_data[index2][1]:
        if same < train_data_size / 2:
            same += 1
            train_inputs[k] = torch.cat((image_data[index1][0], image_data[index2][0]))
            train_labels[k] = 1
            k += 1

        elif same2 < test_data_size / 2:
            same2 += 1
            test_inputs[k2] = torch.cat((image_data[index1][0], image_data[index2][0]))
            test_labels[k2] = 1
            k2 += 1

print("Same pairs done")

while diff + diff2 != int(train_data_size / 2 + test_data_size / 2):
    index1 = np.random.randint(len(image_data))
    index2 = np.random.randint(len(image_data))

    if image_data[index1][1] != image_data[index2][1]:
        if diff < train_data_size / 2:
            diff += 1
            train_inputs[k] = torch.cat((image_data[index1][0], image_data[index2][0]))
            train_labels[k] = 0
            k += 1

        elif diff2 < test_data_size / 2:
            diff2 += 1
            test_inputs[k2] = torch.cat((image_data[index1][0], image_data[index2][0]))
            test_labels[k2] = 0
            k2 += 1

print("Different pairs done")

train_dataset = torch.utils.data.TensorDataset(train_inputs.to(device), train_labels.to(device))
test_dataset = torch.utils.data.TensorDataset(test_inputs.to(device), test_labels.to(device))
torch.save({"train": train_dataset, "test": test_dataset}, "face_data2.pth")