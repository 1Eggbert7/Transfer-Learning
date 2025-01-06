import torchvision.models as models
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.optim as optim

model = models.resnet18(pretrained=True)

# Freeze all parameters
for params in model.parameters():
    params.requires_grad_ = False

# Replace the final fully connected layer
model.fc = nn.Linear(model.fc.in_features, 1)

# Define loss, optimizer and train_step
loss_fn = BCEWithLogitsLoss() # Binary Cross Entropy Loss 
optimizer = optim.Adam(model.parameters())
train_step = make_train_step(model, loss_fn, optimizer)

