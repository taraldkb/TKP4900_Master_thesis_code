
def map_value(value, low, high):
    """
    Function for mapping a value 0-1 to a value between low and high
    :param value: value between 0-1
    :param low: the lowest value in maping set
    :param high: the highest value in maping set
    """
    return float(low + (high - low)*value)
