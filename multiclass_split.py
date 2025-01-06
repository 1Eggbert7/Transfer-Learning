################################################################
# run once in the beginning to split data for training/validation
# ###############################################################

import shutil
import os


train_val_split = 0.8
train_dir = 'data/mc_data/mc_training'
val_dir = 'data/mc_data/mc_validation'
test_dir = 'data/mc_data/mc_test'

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
t_breeds, t_labels  = splitToBreedClasses('data/annotations/test.txt')

for dir in breed_labels:
    if not os.path.exists(train_dir +"/"+ dir):
        os.makedirs(train_dir +"/"+ dir)
    if not os.path.exists(val_dir +"/"+ dir):
        os.makedirs(val_dir +"/"+ dir)    
    if not os.path.exists(test_dir +"/"+ dir):
        os.makedirs(test_dir +"/"+ dir)

for i in range(len(t_breeds)):
    for j, img in enumerate(t_breeds[i]):
        shutil.move(f'data/images/{img}.jpg', test_dir + "/" + t_labels[i])

for i in range(len(breeds)):
    for j, img in enumerate(breeds[i]):
        if j < (len(breeds[i]) * train_val_split):
            shutil.move(f'data/images/{img}.jpg', train_dir+ "/" + breed_labels[i])
        else:
            shutil.move(f'data/images/{img}.jpg', val_dir+ "/" + breed_labels[i])
