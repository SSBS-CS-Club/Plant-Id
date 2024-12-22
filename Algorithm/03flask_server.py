import time
import os
import torch
from PIL import Image
from torchvision import transforms,models

from flask import request, Flask
import torch.nn as nn

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classes = ['当归','枸杞子']

num_classes = 2

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


path_model="./Algorithm/model.ckpt"
model=torch.load(path_model, map_location=torch.device('cpu'))
model = model.to(device)

def get_imageNdarray(imageFilePath):
    input_image = Image.open(imageFilePath).convert("RGB")
    return input_image


def process_imageNdarray(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_chw = preprocess(input_image)
    return img_chw


def predict(imageFilePath):
    input_image = get_imageNdarray(imageFilePath)  
    input_image = input_image.resize((224,224))
    img_chw = process_imageNdarray(input_image)

    if torch.cuda.is_available():
        img_chw = img_chw.view(1, 3, 224, 224).to(device)
    else:
        img_chw = img_chw.view(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        torch.no_grad()
        out = model(img_chw)

        score = torch.nn.functional.softmax(out, dim=1)[0] * 100  

        predicted = torch.max(out, 1)[1]

        score = score[predicted.item()].item()   

        if predicted <= 12 and score>50:
            result1 = str(classes[predicted.item()])
        else:
            result1 = '无法识别'
        return result1

@app.route("/", methods=['POST'])
def return_result():
    startTime = time.time()
    received_file = request.files['file']
    imageFileName = received_file.filename
    if received_file:
        received_dirPath = './resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('图片文件保存到此路径：%s' % imageFilePath)
        usedTime = time.time() - startTime
        usedTime = usedTime * 1000
        print('接收图片并保存，总共耗时%.2fms' % usedTime)
        startTime = time.time()
        print(imageFilePath)
        result = predict(imageFilePath)
        usedTime = time.time() - startTime
        print('完成对接收图片的检测，总共耗时%.2fms' % usedTime)
        print("testtest",result)
        result = result + str(' ') + str('%.2fms'%usedTime)
        return result
    else:
        return 'failed'


if __name__ == "__main__":
    app.run("127.0.0.1", port=4399)


