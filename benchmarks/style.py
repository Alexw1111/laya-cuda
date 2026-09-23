"""Shared, offline-friendly chart style for benchmark exports."""

# Teal = this library, indigo = reference backends, amber = remote services.
COLORS = ["#2B7A69", "#5B5FC7", "#C48526", "#A9ABDF"]
BACKEND_COLORS = dict(zip(("cuda", "upstream", "jev", "official"), COLORS))
INK = "#1C2024"
MUTED = "#71767D"
GRID = "#E1DFD7"
PAPER = "#FBFAF7"
# Titles use the serif display face; fall back to DejaVu where Windows fonts are absent.
TITLE = {"fontfamily": "serif"}


def apply_style(plt):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "font.sans-serif": ["Segoe UI", "Inter", "DejaVu Sans"],
        "font.serif": ["Georgia", "Palatino Linotype", "DejaVu Serif"],
        "font.monospace": ["Cascadia Mono", "Consolas", "DejaVu Sans Mono"],
        "axes.titlesize": 14, "axes.titleweight": "normal", "axes.titlecolor": INK,
        "axes.labelcolor": MUTED, "axes.labelsize": 9, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": GRID,
        "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": .7, "grid.linestyle": (0, (3, 3)),
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "svg.fonttype": "none",
        "legend.frameon": False, "legend.labelcolor": INK,
    })
