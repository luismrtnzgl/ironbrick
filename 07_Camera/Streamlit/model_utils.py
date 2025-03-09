import torch
from torchvision import models

def load_model(model_path, num_classes=56):
    """Carga el modelo entrenado sin descargar pesos automáticamente."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.efficientnet_b0(weights=None)  # ❌ No intenta descargar pesos
    num_features = model.classifier[1].in_features

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.6),
        torch.nn.Linear(num_features, num_classes)
    )

    # Cargar pesos del modelo entrenado
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False) 

    model.to(device)
    model.eval()
    return model
