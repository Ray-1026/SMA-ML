import os
import joblib
import numpy as np
import yaml
from pandas import DataFrame, read_csv


def read_yaml(cfgfile):
    with open(cfgfile, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg


def printDetail(name, path, X, y):
    print(f"Dataset : {name}")
    print(f"\tFile name : {path}")
    print(f"\tx : {X.shape}")
    print(f"\ty : {y.shape}\n")


def load_data(path, feat_l, parm_l):
    df = DataFrame(read_csv(path, encoding="utf-8"))
    features = df[feat_l].values
    parameters = df[parm_l].values

    return features, parameters


def exhaustion(n, k, at, temp_list, lists):
    if len(temp_list) == k:
        lists.append(temp_list)
        return
    for i in range(at, n):
        if i not in temp_list:
            exhaustion(n, k, i, temp_list + [i], lists)


def printResult(best_list, best_models):
    print(f"Best Features : {best_list}\n")
    print("Weights : ")
    for j in range(len(best_models)):
        if best_models[j].weights is None:
            w = None
        else:
            w = [f"{i:>7.4f}" for i in best_models[j].weights]
        print(f"\t{best_models[j].name:^10} : {w}")

    tot = 0
    print("\nTest Loss : ")
    for j in range(len(best_models)):
        tot += best_models[j].test_loss
        print(f"\t{best_models[j].name:^10} : {best_models[j].test_loss:>7.4f}")
    print(f"\t{'Total':^10} : {tot:>7.4f}")

    print("\nTrain Loss : ")
    for j in range(len(best_models)):
        print(f"\t{best_models[j].name:^10} : {best_models[j].train_loss:>7.4f}")


def writeRes(path, pred, parms, output):
    pred_round = np.round(pred, 4)

    src_df = DataFrame(read_csv(path, encoding="utf-8"))
    src_df = src_df.iloc[:, :7]
    elements_name = src_df.keys().values
    elements_percent = src_df.values

    data = np.hstack((elements_percent, pred_round))
    col = np.hstack((elements_name, parms))

    df = DataFrame(data, columns=col)
    df.to_csv(f"res/{output}", index=False, encoding="utf-8")


def saveWeights(feats, models):
    if not os.path.exists("weights"):
        os.mkdir("weights")
    else:
        for file in os.listdir("weights"):
            os.remove(f"weights/{file}")

    print()
    for m in models:
        joblib.dump(m.ensemble_model, f"weights/{m.name}.pkl")
        print(f"Save : weights/{m.name}.pkl")

    joblib.dump(feats, f"weights/feats.pkl")
    print(f"Save : weights/feats.pkl")
