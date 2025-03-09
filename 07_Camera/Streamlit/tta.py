import torchvision.transforms as transforms

# 🔥 Definir Test Time Augmentation (TTA)
tta_transforms = [
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1))
]

def apply_tta(image, num_tta=5):
    """Aplica Test Time Augmentation (TTA) a una imagen."""
    augmented_images = [image]
    for _ in range(num_tta):
        img = image.copy()
        for transform in tta_transforms:
            img = transform(img)
        augmented_images.append(img)
    return augmented_images
