from rl.train import train_agent, continue_train_agent
from rl.test import test_agent
from utils.cleanup import cleanup
from utils.plot_logs_function import plot_logs
import os

choice = int(input("what to run: "))

if choice == 0:
    cleanup()
elif choice == 1:
    train_agent()
elif choice == 2:
    case = input("which case: ")
    continue_train_agent("Save/Case"+case+".pt", "logs/case"+case+".csv", 100)
elif choice == 3:
    case = input("which case: ")
    plot_logs("logs/case"+case+".csv")
elif choice == 4:
    case = input("which case: ")
    test_agent("Save/Case"+case+".pt")
else:
    print("nope")

