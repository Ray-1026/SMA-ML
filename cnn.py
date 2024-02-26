import numpy as np
import utils
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from models.regression import Regression
from models.svr import SVR_Model
from models.random_forest import RandomForest


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv1d(1, 3, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2),
        #     # nn.Conv1d(4, 8, 3, 1, 1),
        #     # nn.ReLU(),
        #     # nn.MaxPool1d(2),
        #     # nn.Conv1d(8, 16, 3, 1, 1),
        #     # nn.ReLU(),
        #     # nn.MaxPool1d(2),
        # )
        self.dense = nn.Sequential(
            nn.Linear(11, 8),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Linear(4, 1),
            nn.Linear(8, 4),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Linear(4, 1),
        )

        self.batchnorm = nn.BatchNorm1d(1)

    def forward(self, x):
        # print(x.shape)

        # x = self.conv(x)
        x = torch.flatten(x, 1)
        # x = nn.LeakyReLU()(x)
        # print(x.shape)
        x = self.dense(x)
        return x


if __name__ == "__main__":
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
    }

    # random seed
    np.random.seed(0)

    # load data
    train_x2, train_y2 = utils.load_data(B2_name, feats, B2_parms)
    train_x19, train_y19 = utils.load_data(B19_name, feats, B19_parms)

    # print detail
    # if cfg["B2"]["train"]:
    #     utils.printDetail(cfg["B2"]["name"], B2_name, train_x2, train_y2)
    if cfg["B19"]["train"]:
        utils.printDetail(cfg["B19"]["name"], B19_name, train_x19, train_y19)

    # # find all combination
    # choose = []
    # for i in range(1, train_x19.shape[1] + 1):
    #     utils.exhaustion(train_x19.shape[1], i, 0, [], choose)

    # find best combination
    best_loss = float("inf")
    best_paras = None
    best_models = None

    # dataset
    train_x19 = torch.from_numpy(train_x19).float()
    train_y19 = torch.from_numpy(train_y19).float()
    train_x19 = train_x19.unsqueeze(1)
    # print(train_x19.shape)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        train_x19, train_y19, test_size=0.2, random_state=42
    )

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    # model
    cnn = CNN().cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
    loss_func = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=0.0001
    )

    # train
    for epoch in range(200):
        cnn.train()
        train_losses, test_losses = [], []
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = b_x.cuda(), b_y.cuda()
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        cnn.eval()
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(test_loader):
                b_x, b_y = b_x.cuda(), b_y.cuda()
                output = cnn(b_x)
                loss = loss_func(output, b_y)
                test_losses.append(loss.item())
        print(
            f"Epoch : {epoch+1} | Loss : {sum(train_losses) / len(train_losses):.5f} | Test Loss : {sum(test_losses) / len(test_losses):.5f}"
        )

    torch.save(cnn.state_dict(), "weights/cnn.pth")

    cnn = CNN().cuda()
    cnn.load_state_dict(torch.load("weights/cnn.pth"))

    cnn.eval()
    with torch.no_grad():
        pred = cnn(train_x19.cuda())
        pred = pred.cpu()
        rmse = torch.sqrt(torch.mean((train_y19 - pred) ** 2, dim=0))
        print(f"RMSE : {rmse}")

    # for i in choose:
    #     model_list = []

    #     # # B2
    #     # for para in range(len(B2_paras)):
    #     #     model = model_classes[model_type][1](
    #     #         f"B2_{B2_paras[para]}", train_x2[:, i], train_y2[:, para]
    #     #     )
    #     #     model.fit()
    #     #     model_list.append(model)

    #     # B19'
    #     for para in range(len(B19_paras)):
    #         model = model_classes[model_type][1](
    #             f"B19_{B19_paras[para]}", train_x19[:, i], train_y19[:, para]
    #         )
    #         model.fit()
    #         model_list.append(model)

    #     total_loss = 0
    #     for j in range(len(model_list)):
    #         total_loss += model_list[j].test_loss

    #     if total_loss < best_loss:
    #         best_loss = total_loss
    #         best_paras = [feats[name] for name in i]
    #         best_models = model_list

    # utils.printResult(best_paras, best_models)

    # if save_file:
    #     utils.writeB2(save_path[0], B2_paras, best_models)
    #     utils.writeB19(save_path[1], B19_paras, best_models)
