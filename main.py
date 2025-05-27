from rl.train import train_agent, continue_train_agent
from rl.test import test_agent
from utils.cleanup import cleanup
from utils.plot_logs_function import plot_logs
import runpy
from rl.run_test import test_multiple



choice = int(input("what to run: "))

if choice == 0:
    cleanup()
elif choice == 1:
    case = input("which case: ")
    train_agent(case)
elif choice == 2:
    case = input("which case: ")
    ep = int(input("how many episodes: "))
    continue_train_agent("Save/Case"+case+".pt", "logs/case"+case+".csv", int(case), ep)
elif choice == 3:
    case = input("which case: ")
    plot_logs("logs/case"+case+".csv")
elif choice == 4:
    case = input("which case: ")
    test_agent(case, "Save/Case"+case+".pt")
elif choice == 5:
    runpy.run_path("optimazation/optimizer.py")
    test_agent(int(case), "Save/Case"+case+".pt")
elif choice == 6:
    test_multiple()
else:
    print("nope")

