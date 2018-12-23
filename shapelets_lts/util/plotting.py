from __future__ import division, print_function

from random import sample

import pandas as pd
import seaborn as sns


def plot_sample_shapelets(shapelets, sample_size=1e6):
    """Plots a random sample from the passed shapelets.

    Args:
        shapelets (list): list of 1-d numpy arrays, one for each shapelet.
        sample_size: number of random shapelets to be selected for plotting.
    """
    # select maximum of len(shapelets) sample shapelets
    some_shapelets = sample(shapelets, min(sample_size, len(shapelets)))

    # put the shapelets in a format suitable for plotting with seaborn
    def _shapelet_to_df(id_, shapelet):
        return pd.DataFrame(
            dict(shapelet=id_, X=range(len(shapelet)), Y=shapelet)
        )

    shapelets_df = pd.concat(
        objs=[_shapelet_to_df(_id, s) for _id, s in enumerate(some_shapelets)],
        axis=0
    ).reset_index(drop=True)

    # plot the shapelets
    grid = sns.FacetGrid(shapelets_df, col="shapelet", col_wrap=6)
    grid.map(sns.lineplot, "X", "Y")
