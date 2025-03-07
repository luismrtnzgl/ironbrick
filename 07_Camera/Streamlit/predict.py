import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def predict(image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
        predicted_class = np.argmax(probabilities)

    return predicted_class, probabilities
