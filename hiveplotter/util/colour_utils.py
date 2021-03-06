import pyx


def pyx2tuple(col_obj):
    return col_obj.r, col_obj.g, col_obj.b

def convert_colour2(inp):
    """
    Convert user input into a pyx colour object.
    :param inp: string corresponding to pyx's RGB or CMYK named colours, tuple of (r,g,b) values, or greyscale value between 0 and 1
    :return: pyx.color.cmyk object
    """

    if isinstance(inp, pyx.color.color):
        return inp.cmyk()

    if inp in cmyk_names:
        return eval("pyx.color.cmyk." + inp)
    elif inp in rgb_names:
        return eval("pyx.color.rgb." + inp).cmyk()

    try:
        length = len(inp)
        if length == 3:
            return pyx.color.rgb(*inp).cmyk()
        elif length == 4:
            return pyx.color.cmyk(*inp)
    except TypeError:
        pass

    try:
        return pyx.color.gray(inp).cmyk()
    except:
        return pyx.color.gray(0.5).cmyk()


def convert_colour(inp):
    """
    Convert user input into a pyx colour object.
    :param inp: string corresponding to pyx's RGB or CMYK named colours, tuple of (r,g,b) values, or greyscale value between 0 and 1
    :return: rgb tuple
    """

    if isinstance(inp, pyx.color.color):
        return pyx2tuple(inp.rgb())

    if inp in rgb_names:
        return pyx2tuple(eval("pyx.color.rgb." + inp))
    elif inp in cmyk_names:
        return pyx2tuple(eval("pyx.color.cmyk." + inp).rgb())

    try:
        length = len(inp)
        if length == 3:
            return inp
        elif length == 4:
            return pyx2tuple(pyx.color.cmyk(*inp).rgb())
    except TypeError:
        pass

    try:
        return pyx2tuple(pyx.color.gray(inp).rgb())
    except:
        return pyx2tuple(pyx.color.gray(0.5).rgb())


def hashable_colour(col):
    try:
        hash(col)
        return col
    except TypeError as e:
        if 'unhashable' in str(e):
            return tuple(col)


def categories_to_float(categories):
    """
    Converts a set of categories into a set of floats for use with colour gradients
    :param categories: An iterable of unique categories
    :return: A dictionary whose keys are the original categories and whose values are unique, evenly spaced floats between 0 and 1.
    """
    return {category: i/(len(categories)-1) for i, category in enumerate(sorted(categories))}

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