import torch
from torchvision import models

def load_model(model_path):
    """Carga el modelo entrenado, asegurando que la arquitectura coincida con los pesos guardados."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 Cargar el diccionario de pesos
    state_dict = torch.load(model_path, map_location=device)

    # 🔥 Obtener el número de clases desde los pesos guardados
    num_classes = state_dict["classifier.1.weight"].shape[0]
    print(f"📌 Número de clases detectado en los pesos guardados: {num_classes}")

    # 🔥 Cargar modelo EfficientNet-B0 sin pesos preentrenados
    model = models.efficientnet_b0(weights=None)

    # 🔥 Ajustar la última capa al número de clases detectado
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.6),
        torch.nn.Linear(num_features, num_classes)  # Se adapta automáticamente
    )

    # 🔥 Cargar los pesos y asegurar que coincidan con la arquitectura
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
