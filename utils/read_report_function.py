import statistics as st


def read_report(file_path):

    """
    Function to read report.out file and return cleaned and manipulated data

    :param file_path: report.out file path
    :return: data, array size (9, )  [concentration X8, wind_velocity]
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
        raise FileNotFoundError(f"Could not read file: {f}")


