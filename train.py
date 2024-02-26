import numpy as np
import utils
from models.regression import Regression
from models.svr import SVR_Model
from models.random_forest import RandomForest
from models.gradient_boost import GradientBoost


def main():
    # config
    cfg = utils.read_yaml("config.yaml")

    B19_name = cfg["B19"]["train_dataset"]
    B19_parms = cfg["B19"]["parms"]

    B2_name = cfg["B2"]["train_dataset"]
    B2_parms = cfg["B2"]["parms"]

    feats = cfg["feature"]

    # model_type = cfg["model_type"]
    model_classes = {
        "regression": ["Linear Regression", Regression],
        "svr": ["Support Vector Regression", SVR_Model],
        "random_forest": ["Random Forest Regression", RandomForest],
        "gradient_boost": ["Gradient Boosting Regression", GradientBoost],
    }

    # random seed
    np.random.seed(0)

    # load data
    train_x2, train_y2 = utils.load_data(B2_name, feats, B2_parms)
    train_x19, train_y19 = utils.load_data(B19_name, feats, B19_parms)

    # print detail
    if cfg["B2"]["train"]:
        utils.printDetail(cfg["B2"]["name"], B2_name, train_x2, train_y2)
    if cfg["B19"]["train"]:
        utils.printDetail(cfg["B19"]["name"], B19_name, train_x19, train_y19)

    # find all combination
    choose = []
    # for i in range(train_x19.shape[1], train_x19.shape[1] + 1):
    for i in range(4, train_x19.shape[1] + 1):
        utils.exhaustion(train_x19.shape[1], i, 0, [], choose)

    # find best combination
    best_loss = float("inf")
    best_feats = None
    best_models = None

    for i in choose:
        model_list = []

        # B2
        if cfg["B2"]["train"]:
            for parm in range(len(B2_parms)):
                model_type = cfg["B2"]["model_types"][parm]
                model = model_classes[model_type][1](
                    f"B2_{B2_parms[parm]}", train_x2[:, i], train_y2[:, parm]
                )
                model.fit()
                model_list.append(model)

        # B19'
        if cfg["B19"]["train"]:
            for parm in range(len(B19_parms)):
                model_type = cfg["B19"]["model_types"][parm]
                model = model_classes[model_type][1](
                    f"B19_{B19_parms[parm]}", train_x19[:, i], train_y19[:, parm]
                )
                model.fit()
                model_list.append(model)

        total_loss = 0
        for j in range(len(model_list)):
            total_loss += model_list[j].test_loss

        if total_loss < best_loss:
            best_loss = total_loss
            best_feats = [feats[name] for name in i]
            best_models = model_list

    utils.printResult(best_feats, best_models)
    utils.saveWeights(best_feats, best_models)


if __name__ == "__main__":
    main()
