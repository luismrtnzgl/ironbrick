import torch
from torchvision import models

def load_model(model_path, num_classes=56):  # 🔥 Asegurar que el número de clases coincide con el modelo guardado
    """Carga el modelo entrenado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features

    # 🔥 Asegurar que el modelo tenga el mismo número de clases que el archivo guardado
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.6),
        torch.nn.Linear(num_features, num_classes)  # 🔥 Cambia dinámicamente el número de clases
    )

    # 🔥 Cargar pesos ignorando los errores de tamaño
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model
