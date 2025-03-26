# WORK IN PROGRESS

def read_report(file_path):

    """
    Function to read report.out file and return cleaned and manipulated data

    :param file_path: report.out file path
    :return: data, array size (9, )  [concentration X8, wind_velocity]
    """

    data = []



    try:
        with open(file_path, "r") as f:
            data = None

        return data

    except:
        raise FileNotFoundError(f"Could not read file: {f}")


