import pyx


def convert_colour(inp):
    """
    Convert user input into a pyx colour object.
    :param inp: string corresponding to pyx's RGB or CMYK named colours, tuple of (r,g,b) values, or greyscale value between 0 and 1
    :return: pyx.color.cmyk object
    """
    if inp in cmyk_names:
        return eval("pyx.color.cmyk." + inp)
    elif inp in rgb_names:
        return _rgb_obj_to_cmyk_obj(eval("pyx.color.rgb." + inp))

    try:
        length = len(inp)
        if length == 3:
            return _rgb_to_cmyk_obj(*inp)
        elif length == 4:
            return pyx.color.cmyk(*inp)
    except TypeError:
        pass

    try:
        return _greyscale_to_cmyk_obj(inp)
    except:
        return _greyscale_to_cmyk_obj(0.5)


def categories_to_float(categories):
    """
    Converts a set of categories into a set of floats for use with colour gradients
    :param categories: An iterable of unique categories
    :return: A dictionary whose keys are the original categories and whose values are unique, evenly spaced floats between 0 and 1.
    """
    return {category: i/(len(categories)-1) for i, category in enumerate(sorted(categories))}


def _rgb_obj_to_cmyk_obj(rgb_obj):
    return _rgb_to_cmyk_obj(rgb_obj.r, rgb_obj.g, rgb_obj.b)


def _rgb_to_cmyk_obj(r, g, b):
    c = 1-r
    m = 1-g
    y = 1-b

    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    return pyx.color.cmyk(c, m, y, k)


def _greyscale_to_cmyk_obj(grey_val):
    return pyx.color.cmyk(0, 0, 0, 1 - grey_val)


rgb_names = {
    "red",
    "green",
    "blue",
    "white",
    "black"
}

cmyk_names = {
    "Apricot",
    "Aquamarine",
    "Bittersweet",
    "Black",
    "Blue",
    "BlueGreen",
    "BlueViolet",
    "BrickRed",
    "Brown",
    "BurntOrange",
    "CadetBlue",
    "CarnationPink",
    "Cerulean",
    "CornflowerBlue",
    "Cyan",
    "Dandelion",
    "DarkOrchid",
    "Emerald",
    "ForestGreen",
    "Fuchsia",
    "Goldenrod",
    "Gray",
    "Green",
    "GreenYellow",
    "JungleGreen",
    "Lavender",
    "LimeGreen",
    "Magenta",
    "Mahogany",
    "Maroon",
    "Melon",
    "MidnightBlue",
    "Mulberry",
    "NavyBlue",
    "OliveGreen",
    "Orange",
    "OrangeRed",
    "Orchid",
    "Peach",
    "Periwinkle",
    "PineGreen",
    "Plum",
    "ProcessBlue",
    "Purple",
    "RawSienna",
    "Red",
    "RedOrange",
    "RedViolet",
    "Rhodamine",
    "RoyalBlue",
    "RoyalPurple",
    "RubineRed",
    "Salmon",
    "SeaGreen",
    "Sepia",
    "SkyBlue",
    "SpringGreen",
    "Tan",
    "TealBlue",
    "Thistle",
    "Turquoise",
    "Violet",
    "VioletRed",
    "White",
    "Yellow",
    "YellowGreen",
    "YellowOrange",
}