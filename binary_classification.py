import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

from tqdm import tqdm


class FineTunedResNet():
    def __init__(self, num_classes, criterion=nn.CrossEntropyLoss(), optimizer=optim.SGD, lr=0.001, pretrained_lr = 0.0001) -> None:
        self.is_finetuned = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_classes = num_classes
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        for name, param in self.model.named_parameters():
            if 'layer4' in name or 'layer3' in name:
                # unfreeze the parameters of the last layer
                param.requires_grad = True
            else:
                # freeze the parameters of all other layers
                param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.n_classes)
        self.model = self.model.to(self.device)

        self.criterion = criterion
        
        # set the learning rate for each parameter group
        params = [
#             {'params': self.model.conv1.parameters(), 'lr': pretrained_lr},
#             {'params': self.model.bn1.parameters(), 'lr': pretrained_lr},
#             {'params': self.model.layer1.parameters(), 'lr': pretrained_lr},
#             {'params': self.model.layer2.parameters(), 'lr': pretrained_lr},
            {'params': self.model.layer3.parameters(), 'lr': pretrained_lr},
            {'params': self.model.layer4.parameters(), 'lr': pretrained_lr},
            {'params': self.model.fc.parameters(), 'lr': lr},
        ]

        # define the optimizer
        self.optimizer = optimizer(params, lr=lr, momentum=0.9)

    def train(self, train_loader, num_epochs, save_model=True, safepath=""):
        self.is_finetuned = True

        if save_model and not safepath:
            safepath = f"ft_model_{self.n_classes}_{num_epochs}.pt"


        for epoch in range(num_epochs):
            r_loss = 0.0
            for i, data in tqdm(enumerate(train_loader, 0)):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                out = self.model(x)
                loss = self.criterion(out, y)

                loss.backward()
                self.optimizer.step()

                r_loss += loss.item()
            print(f"Epoch {epoch}: loss {r_loss}")
        
        if save_model:
            torch.save(self.model.state_dict(), safepath)

    def validate(self, val_loader):
        if not self.is_finetuned:
            raise Exception("The model has not been fine-tuned. Call model.train first to fine-tune the model.")
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
traindir = "data/binary_data/training"
validdir = "data/binary_data/validation"
testdir = "data/binary_data/test"

transform  = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
])

train_set = datasets.ImageFolder(traindir, transform)
val_set = datasets.ImageFolder(validdir, transform)
test_set = datasets.ImageFolder(testdir, transform)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

classes = train_set.classes

binary_model = FineTunedResNet(2)
binary_model.train(train_loader, 20)
# -

binary_model.validate(val_loader)

binary_model.validate(test_loader)
