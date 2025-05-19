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
    continue_train_agent("Save/best_policy.pt", "logs/first_training.csv", 100)
elif choice == 3:
    plot_logs("logs/case2.csv")
elif choice == 4:
    test_agent("Save/Case2.pt")
else:
    print("nope")

