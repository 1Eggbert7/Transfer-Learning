################################################################
# run once in the beginning to split data for training/validation
import shutil
import os


train_val_split = 0.8
""
cat_train_dir = 'data/binary_data/training/cat/'
cat_val_dir = 'data/binary_data/validation/cat/'
dog_train_dir = 'data/binary_data/training/dog/'
dog_val_dir = 'data/binary_data/validation/dog/'

def splitToBreedClasses(filename):
    breeds = [[] for i in range(37)]
    breed_labels = ["" for i in range(37)]
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            tokens = line.split()
            species = int(tokens[2])
            breed = int(tokens[3])
            if species == 2:
                breeds[breed-1].append(tokens[0])
                breed_labels[breed-1] = tokens[0][:tokens[0].rfind("_")]
            else:
                breeds[breed + 24].append(tokens[0])
                breed_labels[breed +24] = tokens[0][:tokens[0].rfind("_")]
    return breeds, breed_labels

breeds, breed_labels  = splitToBreedClasses('data/annotations/trainval.txt')

for i in range(len(breeds)):
    for j, img in enumerate(breeds[i]):
        if j < (len(breeds[i]) * train_val_split):
            if breed_labels[i][0].isupper():
                shutil.move(f'data/images/{img}.jpg', cat_train_dir)
            else:
                shutil.move(f'data/images/{img}.jpg', dog_train_dir)
        else:
            if breed_labels[i][0].isupper():
                shutil.move(f'data/images/{img}.jpg', cat_val_dir)
            else:
                shutil.move(f'data/images/{img}.jpg', dog_val_dir)
