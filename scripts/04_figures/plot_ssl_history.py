#!/usr/bin/env python
"""Plot masked tabular SSL training history."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def main() -> None:
    history = pd.read_csv(TABLE_DIR / "masked_tabular_ssl_history.csv")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train total")
    ax.plot(history["epoch"], history["dev_loss"], marker="o", label="Dev total")
    ax.plot(history["epoch"], history["dev_cat_loss"], marker="s", label="Dev categorical")
    ax.plot(history["epoch"], history["dev_num_loss"], marker="s", label="Dev numeric")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked reconstruction loss")
    ax.set_title("Masked Tabular SSL Training")
    ax.set_xticks(history["epoch"])
    ax.legend(frameon=False)
    fig.tight_layout()
    output = FIGURE_DIR / "masked_tabular_ssl_training_history.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
