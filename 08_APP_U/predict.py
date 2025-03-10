import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tta import apply_tta

def predict(image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Aplica TTA a la imagen
    images = apply_tta(image)  
    images = [transform(img).unsqueeze(0).to(device) for img in images]

    with torch.no_grad():
        outputs = [model(img) for img in images]
        probabilities = [torch.nn.functional.softmax(out, dim=1).cpu().numpy() for out in outputs]
    
    # Promedia predicciones
    avg_probabilities = np.mean(probabilities, axis=0)  
    predicted_class = np.argmax(avg_probabilities)

    return predicted_class, avg_probabilities
