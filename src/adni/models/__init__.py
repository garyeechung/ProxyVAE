from .vae import ConditionalVAE, InvariantVAE
from .posthocs import ProxyRep2InvarRep, VariationalPredictor


if __name__ == "__main__":
    print(ConditionalVAE.__name__)
    print(InvariantVAE.__name__)
    print(ProxyRep2InvarRep.__name__)
    print(VariationalPredictor.__name__)
