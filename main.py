from rl.train import train_agent, continue_train_agent
from utils.cleanup import cleanup
import os

choice = int(input("what to run"))

if choice == 0:
    cleanup()
elif choice == 1:
    train_agent()
elif choice == 2:
    continue_train_agent("Save/best_policy.pt", 100)
else:
    print("nope")

