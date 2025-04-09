import statistics as st


def read_report(file_path):

    """
    Function to read report.out file and return cleaned and manipulated data

    :param file_path: report.out file path
    :return data, array size (8, )  [concentration X8]
    """

    data = [[] for _ in range(8)]

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            lines = lines[3:]  # remove headers

            for line in lines:
                holder = line.strip().split()   # remove whitespace and split into a list
                holder = holder[1:]  # remove time step counter

                for i in range(len(holder)):
                    data[i].append(float(holder[i]))

                for i in range(len(data)):
                    data[i] = st.mean(data[i][-10])

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not read file: {file_path}")


def read_single_data_file(file_path):
    """
    Read .out file containing water loss data and return cleansed  and manipulated datadata
    :param file_path: file path to water loss file location str
    :return data: float variable with mean of last 10 time steps
    """

    data = []

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            lines = lines[3:]  # remove headers in file

            for line in lines:
                holder = line.strip().split()
                data.append(float(holder[-1]))

            data = st.mean(data)

            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not read file: {file_path}")