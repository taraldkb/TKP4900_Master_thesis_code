import statistics as st
import numpy as np
import matplotlib.pyplot as plt


def read_concentrations(file_path):

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
                holder = holder[2:]  # remove time step counter

                for i in range(len(holder)):
                    data[i].append(float(holder[i]))

            for j in range(len(data)):
                data[j] = st.mean(data[j][-10:])

        return np.array(data)

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

            data = st.mean(data[-10:])

            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not read file: {file_path}")


def plot_conc(file_path, run):
    data = [[] for i in range(8)]
    time = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines = lines[3:]

        for line in lines:
            holder = line.strip().split()
            time.append(float(holder[1]))
            holder = holder[2:]

            for i in range(len(holder)):
                data[i].append(float(holder[i]))

        for i in range(len(data)):
            plt.plot(time, data[i], label=f" wall {i+1}")
        plt.title(f"concentrations run: {run}")
        plt.legend()
        plt.grid()
        plt.show()

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path} ")


def plot_water(file_path, run):
    data = []
    time = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines = lines[3:]

        for line in lines:
            holder = line.strip().split()
            data.append(round(float(holder[-1]),2))
            time.append(float(holder[1]))

        plt.plot(time, data)
        plt.title(f"water usage run: {run}")
        plt.grid()
        plt.show()

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path} ")