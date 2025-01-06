import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import random

from tqdm import tqdm
np.random.seed(0)


class FineTunedResNet():
    def __init__(self, num_classes, dropout_rate=0, criterion=nn.CrossEntropyLoss(), optimizer=optim.Adam, lr=0.0005, pretrained_lr = 0.000005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) -> None:
        self.is_finetuned = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_classes = num_classes
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # initially, all layers are frozen
        for param in self.model.parameters():
            param.requires_grad = False
        # only the final layer is unfrozen
        for param in self.model.fc.parameters():
            param.requires_grad = True

        num_ftrs = self.model.fc.in_features

        # Replace the final layer with a Sequential containing the Dropout and Linear layers
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, self.n_classes)
        )

        self.model = self.model.to(self.device)

        self.criterion = criterion

        # define an optimizer for each layer
        self.optimizers = []
        for i in range(4):
            params = [{'params': getattr(self.model, f'layer{i+1}').parameters(), 'lr': pretrained_lr}]
            self.optimizers.append(optimizer(params, betas=betas, eps=eps, weight_decay=weight_decay))
        self.optimizers.append(optimizer([{'params': self.model.fc.parameters(), 'lr': lr}], betas=betas, eps=eps, weight_decay=weight_decay))
        
        self.schedulers = []
        # define a scheduler for each optimizer
        for opt in self.optimizers:
            self.schedulers.append(torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.8))


    def train(self, train_loader, num_epochs, save_model=True, safepath=""):
        self.is_finetuned = True
        
        self.model.train()  # Set the model to training mode

        if save_model and not safepath:
            safepath = f"ft_model_{self.n_classes}_{num_epochs}.pt"

        for epoch in range(num_epochs):
            # decide which layers to unfreeze in this epoch
            if epoch < len(self.optimizers):
                for param in self.optimizers[epoch].param_groups[0]['params']:
                    param.requires_grad = True

            r_loss = 0.0
            for i, data in tqdm(enumerate(train_loader, 0)):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                # clear and update all optimizers
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                out = self.model(x)
                loss = self.criterion(out, y)

                loss.backward()
                for optimizer in self.optimizers:
                    optimizer.step()

                r_loss += loss.item()
            
            # update all schedulers
            for scheduler in self.schedulers:
                scheduler.step()
                
            print(f"Epoch {epoch}: loss {r_loss}")

        if save_model:
            torch.save(self.model.state_dict(), safepath)


    def validate(self, val_loader):
        if not self.is_finetuned:
            raise Exception("The model has not been fine-tuned. Call model.train first to fine-tune the model.")
            
        self.model.eval()  # Set the model to evaluation mode
        
        correct = [0] * self.n_classes
        total = [0] * self.n_classes
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                c = []
                output = self.model(x)
                for out in output:
                    c.append(out.argmax())
                for i in range(len(y)):
                    label = y[i]
                    if label == c[i]:
                        correct[label] += 1
                    total[label] += 1
        return self.get_accuracy(correct, total)

    def test_single(self, test_img):
        if not self.is_finetuned:
            raise Exception("The model has not been fine-tuned. Call model.train first to fine-tune the model.")
        self.model.eval()
        img_tensor = test_img.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        out = self.model(img_tensor)
        pred_class = out.argmax().item()

        return pred_class
    
    def get_accuracy(self, correct, total):
        acc_class = np.zeros((self.n_classes, 1))
        for i in range(self.n_classes):
            acc_class[i] = correct[i]/total[i]
        acc = sum(correct)/sum(total)
        return acc_class, acc


# +
def load_data_subset(traindir, transform, subset_percent=1.0):
    # Load the dataset as usual
    full_dataset = datasets.ImageFolder(traindir, transform=transform)
    
    # Print the size of the full dataset
    print("Full dataset size: ", len(full_dataset))

    # Get the list of all targets (these are the class indices)
    targets = full_dataset.targets

    # Get the set of all unique classes
    classes = set(targets)

    subset_indices = []

    # For each class...
    for class_idx in classes:
        # Get the indices of all images of this class
        class_indices = [i for i, target in enumerate(targets) if target == class_idx]

        # Calculate the number of samples to take from this class
        subset_size = max(1, int(len(class_indices) * subset_percent))

        # Use a random sampler to get a subset of indices from this class
        class_subset_indices = np.random.choice(class_indices, subset_size, replace=False)

        # Add the selected indices to our subset_indices list
        subset_indices.extend(class_subset_indices)

    print("Subset size: ", len(subset_indices))
    
    subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)

    # Create the data loader
    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=16,
        sampler=subset_sampler,
        num_workers=2
    )

    return full_dataset, train_loader



if __name__ == "__main__":
    traindir = "data/mc_data/mc_training"
    validdir = "data/mc_data/mc_validation"
    testdir = "data/mc_data/mc_test"

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10), # Random rotation between -10 and 10 degree
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_set, train_loader = load_data_subset(traindir, data_transforms['train'], subset_percent=0.01)
    val_set = datasets.ImageFolder(validdir, transform=data_transforms['val'])

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    classes = train_set.classes

    #binary_ft_model = FineTunedResNet(2)
    binary_ft_model = FineTunedResNet(37)
    binary_ft_model.train(train_loader, 20)

    test_img_path = "data/mc_data/mc_test/pug/pug_1.jpg"
    test_img = Image.open(test_img_path)
    test_img = data_transforms['val'](test_img)  # Use the 'val' transform here
    pred = binary_ft_model.test_single(test_img)

    print(classes[pred])

# +
#binary_ft_model.validate(val_loader)
# -

# print(val_set.classes)
# print(val_set.class_to_idx)

# +
#binary_ft_model.validate(val_loader)

# +
# List of image paths
#doggo_images = [f"data/binary_data/test/dog/eurasian_OleksDoggo_{i}.jpeg" for i in range(2, 7)]


# Loop over the images
#for img_path in doggo_images:
#    test_img = Image.open(img_path)
#    test_img = data_transforms['val'](test_img)  # Use the 'val' transform here
#    pred = binary_ft_model.test_single(test_img)
#    print(f"For image {img_path}, model predicts: {classes[pred]}")


# -


