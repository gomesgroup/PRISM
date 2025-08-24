import matplotlib.pyplot as plt
import scienceplots  # required before style.use


def apply_default_style() -> None:
    plt.style.use(['science', 'nature'])


