from .resnet import resnet18, ResNet50Spatial, ResNet50Triplet2D
# Import UNetHeatmap conditionally since it might be in gitignore
try:
    from .unet import UNetHeatmap
    __all__ = ["resnet18", "ResNet50Spatial", "ResNet50Triplet2D", "UNetHeatmap"]
except ImportError:
    __all__ = ["resnet18", "ResNet50Spatial", "ResNet50Triplet2D"]
