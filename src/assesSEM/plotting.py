import matplotlib as mpl


def get_cmap(name="hackathon"):
    if name == "hackathon":
        cmap = (mpl.colors.ListedColormap(['white', 'white', '#000000', '#ff0000', '#00ff00', '#ffff00']))
                                                             # black,    red,      green,      yellow
    else:
        raise ValueError
    return cmap