import torch
from torchvision import models

def load_model(model_path):
    """Carga el modelo entrenado, asegurando que el número de clases coincida."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 Cargar modelo EfficientNet-B0 sin pesos preentrenados
    model = models.efficientnet_b0(weights=None)
    
    # 🔥 Cargar los pesos del modelo para obtener el número de clases correcto
    state_dict = torch.load(model_path, map_location=device)
    num_classes = state_dict["classifier.1.weight"].shape[0]  # Detecta automáticamente el número de clases

    # 🔥 Ajustar la última capa al número correcto de clases
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.6),
        torch.nn.Linear(num_features, num_classes)  # Se adapta automáticamente
    )

    # 🔥 Cargar pesos del modelo sin errores de tamaño
    model.load_state_dict(state_dict, strict=False)  

    model.to(device)
    model.eval()
    return model
