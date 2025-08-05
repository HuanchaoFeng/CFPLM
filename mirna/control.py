import random
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from CFPLM import CFPLM_Model
from calculate import *
import torch
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def single_run(data_dict, device):
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_x = torch.tensor(data_dict['train_x'], dtype=torch.long).to(device)
    train_y = torch.tensor(data_dict['train_y'], dtype=torch.float).to(device)
    test_x = torch.tensor(data_dict['test_x'], dtype=torch.long).to(device)
    test_y = torch.tensor(data_dict['test_y'], dtype=torch.float).to(device)
    model = CFPLM_Model(data_dict, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(1, 401):
            model.train()
            optimizer.zero_grad()
            out, loss = model(train_x, train_y, "train")
            loss.backward()
            optimizer.step()

            if epoch == 400:
                model.eval()
                with torch.no_grad():
                    outs, eval_loss = model(test_x, test_y, "test")
                    AUC, AUPR = cal_auc_aupr(test_y.cpu().numpy(), outs.cpu().numpy())
                    ACC, PRE, F1,REC = calculate_metrics(test_y, outs)

    return AUC, AUPR,ACC, PRE, F1,REC
