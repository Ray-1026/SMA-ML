import joblib
import numpy as np
import utils
import os


def main():
    # config
    cfg = utils.read_yaml("config.yaml")

    B19_name = cfg["B19"]["eval_dataset"]
    B19_parms = cfg["B19"]["parms"]

    B2_name = cfg["B2"]["eval_dataset"]
    B2_parms = cfg["B2"]["parms"]

    # random seed
    np.random.seed(0)

    # load best features
    if not os.path.exists(f"weights/feats.pkl"):
        raise Exception("Train your model first to get best features")
    best_feats = joblib.load(f"weights/feats.pkl")

    # load data
    eval_x2, eval_y2 = utils.load_data(B2_name, best_feats, B2_parms)
    eval_x19, eval_y19 = utils.load_data(B19_name, best_feats, B19_parms)

    # print detail
    if cfg["B2"]["eval"]:
        utils.printDetail(cfg["B2"]["name"], B2_name, eval_x2, eval_y2)
    if cfg["B19"]["eval"]:
        utils.printDetail(cfg["B19"]["name"], B19_name, eval_x19, eval_y19)

    # load models and predict
    print("RMSE Loss :")
    if cfg["B2"]["eval"]:
        b2_pred = np.empty((eval_x2.shape[0], 0))
        for parm in range(len(B2_parms)):
            if not os.path.exists(f"weights/B2_{B2_parms[parm]}.pkl"):
                raise Exception(
                    f"Train your model first to save weights of {B2_parms[parm]} in B2."
                )

            model = joblib.load(f"weights/B2_{B2_parms[parm]}.pkl")
            pred = model.predict(eval_x2)
            b2_pred = np.hstack([b2_pred, pred.reshape(-1, 1)])

            # rmse loss
            loss = np.sqrt(np.mean((pred - eval_y2[:, parm]) ** 2))
            print(f"\tB2_{B2_parms[parm]} : {loss:.4f}")

        utils.writeRes(B2_name, b2_pred, B2_parms, cfg["B2"]["eval_output"])

    if cfg["B19"]["eval"]:
        b19_pred = np.empty((eval_x19.shape[0], 0))
        for parm in range(len(B19_parms)):
            if not os.path.exists(f"weights/B19_{B19_parms[parm]}.pkl"):
                raise Exception(
                    f"Train your model first to save weights of {B19_parms[parm]} in B19'."
                )

            model = joblib.load(f"weights/B19_{B19_parms[parm]}.pkl")
            pred = model.predict(eval_x19)
            b19_pred = np.hstack([b19_pred, pred.reshape(-1, 1)])

            # rmse loss
            loss = np.sqrt(np.mean((pred - eval_y19[:, parm]) ** 2))
            print(f"\tB19_{B19_parms[parm]} : {loss:.4f}")

        utils.writeRes(B19_name, b19_pred, B19_parms, cfg["B19"]["eval_output"])


if __name__ == "__main__":
    main()
