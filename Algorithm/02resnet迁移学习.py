import torch
from PIL import Image
from torchvision import datasets, models, transforms,utils
import torch.nn as nn
import numpy as np
import random
import os
import torchvision
from tqdm import tqdm

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(20)

root = './'

# Hyper parameters
num_epochs = 5    
batch_size = 4 
learning_rate = 0.00005   
momentum = 0.9 
num_classes = 2 


class MyDataset(torch.utils.data.Dataset):  
    def __init__(self, datatxt, transform=None, target_transform=None):  
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  
        imgs = []  
        for line in fh:  
            line = line.rstrip()  
            words = line.split()  
            imgs.append((words[0], int(words[1]))) 
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index): 
        fn, label = self.imgs[index]  
        img = Image.open(fn).convert('RGB')  
        img = img.resize((224,224))

        if self.transform is not None:
            img = self.transform(img) 
        return img, label 

    def __len__(self): 
        return len(self.imgs)

train_data = MyDataset(datatxt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(datatxt=root + 'test.txt', transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle=False)

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True)

for i, param in enumerate(model.parameters()):
    param.requires_grad = False 

num_ftrs = model.fc.in_features 
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1)
)   

print(model)

net = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum = momentum )
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,betas=(0.9,0.999))
# train_accs = []
# train_loss = []
test_acc2 = []
test_loss2 = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = tqdm(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    net.eval()
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = net(batch_x)
            loss2 = criterion(out, batch_y)
            test_loss += loss2.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            test_acc += num_correct.item()
        test_acc2.append(test_acc/len(test_data))
        test_loss2.append(test_loss/len(test_data))
        print('Epoch :{}, Test Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, test_loss / (len(
            test_data)), test_acc / (len(test_data))))


    torch.save(net, 'model.ckpt')
print(test_acc2)
