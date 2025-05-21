
# get case name from design parameters
def get_case(H, N):
    if N == 1:
        if H == 50:
            return "Case1"
        elif H == 75:
            return "Case2"
        elif H == 100:
            return "Case3"
    elif N == 2:
        if H == 50:
            return "Case4"
        elif H == 75:
            return "Case5"
        elif H == 100:
            return "Case6"
    else:
        raise ValueError("no case could be found")
