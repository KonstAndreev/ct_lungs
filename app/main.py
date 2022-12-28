import numpy as np
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi import Response
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms

import json

app = FastAPI()

transform_norm = transforms.Compose([
    transforms.ToTensor(),
])

def load_model():
    model = smp.Unet('resnet34', in_channels=1,
                      encoder_weights='imagenet',
                      classes=1, activation=None,
                      encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model.load_state_dict(torch.load('app/model_resnet.pth', map_location=torch.device('cpu')))
    return model

@app.on_event('startup')
async def startup():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')
    model = load_model()

    app.package = {
        'model': model,
    }

@app.get('/prediction/{phrase}')
async def predict(phrase: str, file: UploadFile = File(...)):
    img = np.array(Image.open(file.file))
    xmax, xmin = np.max(img), np.min(img)
    x = (img - xmin) / (xmax - xmin)
    x = transform_norm(x)
    x = torch.reshape(x, (1, 1, 256, 256))
    x = x.type(torch.float32)
    model = app.package['model']
    model.eval()
    with torch.no_grad():
         Y_pred = model(x).detach().to('cpu')
    a = np.array(((Y_pred.squeeze() > 0.5).int()))
    return json.dumps(a.tolist())