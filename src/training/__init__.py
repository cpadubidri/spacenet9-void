from .train import train
<<<<<<< HEAD
from .train_unet import train_unet
from .train_triplet import train_triplet
from .loss_fn import get_loss, CosineTripletLoss

__all = ['train','get_loss','train_unet',"train_triplet","CosineTripletLoss"]
=======
from .loss_fn import get_loss

# Import conditionally since these modules might be in gitignore
__all__ = ['train', 'get_loss']

# Try to import train_unet if available
try:
    from .train_unet import train_unet
    __all__.append('train_unet')
except ImportError:
    pass

# Try to import train_triplet if available
try:
    from .train_triplet import train_triplet
    __all__.append('train_triplet')
except ImportError:
    pass
>>>>>>> 4146e47199da6a6ccba76c2fcb8e745ff672391c
