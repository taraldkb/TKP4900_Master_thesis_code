from rl.test import test_agent



def test_multiple():
    test_cases = [3, 5, 6 ]

    for i in test_cases:
        test_agent(i, f"Save/Case{i}.pt")