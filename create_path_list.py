import os
import re
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "/home/kienvt/dataset/"
IMAGE_DIR = "/home/kienvt/dataset/images/"
LABEL_LIST_PATH = "/home/kienvt/dataset/label_list.txt"
IMAGE_LIST_PATH = "/home/kienvt/dataset/image_list.txt"

with open(IMAGE_LIST_PATH) as f:
    img_list = f.readlines()
    img_list = [DATA_DIR + s.strip() for s in img_list]

with open(LABEL_LIST_PATH, encoding="utf-8") as f:
    label_list = f.readlines()
    label_list = [s.strip() for s in label_list]

assert len(img_list) == len(label_list)

# img_list = img_list[:5000]
# label_list = label_list[:5000]

print("n_images: ", len(img_list))

random_indices = np.random.choice(len(img_list), len(img_list))
img_list = np.array(img_list)[random_indices]
label_list = np.array(label_list)[random_indices]

img_train_list, img_test_list, label_train_list, label_test_list = train_test_split(img_list, label_list, test_size=0.2, random_state=42)


def dump_list_to_file(img_list, label_list, prefix):
    print("dumping", prefix)
    len_list = [len(label) for label in label_list]
    sort_indices = np.argsort(len_list)
    label_list = label_list[sort_indices]
    img_list = img_list[sort_indices]

    with open(prefix + "ImgPathList.txt", "w") as f:
        f.write("\n".join(img_list))


    with open(prefix + "LabelList.txt", "w") as f:
        f.write("\n".join(label_list))

dump_list_to_file(img_train_list, label_train_list, "/home/kienvt/train")
dump_list_to_file(img_test_list, label_test_list, "/home/kienvt/test")

