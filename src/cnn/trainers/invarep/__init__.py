from .cvae import train_cvae
from .proxyvae import train_proxyvae
from .proxy2invarep import train_proxy2invarep
from .posthoc import train_posthoc_predictor


if __name__ == "__main__":
    train_cvae.__doc__ = "This module provides functions to train the Conditional Variational Autoencoder (CVAE) on ADNI dataset."
    train_proxyvae.__doc__ = "This module provides functions to train the Proxy Variational Autoencoder (ProxyVAE) on ADNI dataset."
    train_proxy2invarep.__doc__ = "This module provides functions to train the Proxy2InvaRep model on ADNI dataset."
    train_posthoc_predictor.__doc__ = "This module provides functions to train the post-hoc predictor on InvaRep model."
