from setup.setup import setup

methods = ["svc", "rf", "mlp", "lr", "mnb", "lnr"]

if __name__ == '__main__':
    for max_features in range(1000, 11000, 1000):
        for method in methods:
            tc = setup(max_features)
            print(tc.get_score("dev", method), method, max_features)
