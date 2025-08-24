import matplotlib.pyplot as plt
import scienceplots as _scienceplots  # noqa: F401  required before style.use


def apply_default_style() -> None:
    plt.style.use(['science', 'nature'])


