import pandas as pd
from torch.nn import MSELoss
import torch
import os
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from utils import process_csv_pairs

def main():
    loss = MSELoss()

    path="results/example"

    scores = {}

    for root, dirs, _ in os.walk(path):
        if len(dirs) == 0:
            try:
                test, test_predictions = process_csv_pairs([("{}/one_forecast.csv".format(root), "{}/test.csv".format(root))])
            except Exception:
                continue
            rmse = (loss(torch.tensor(test.values), torch.tensor(test_predictions.values)).item())
            key = '/'.join(root.split("/")[3:10])
            if key in scores:
                scores[key] += rmse
            else:
                scores[key] = rmse

    scoresSorted = sorted(scores.items(), key=lambda x: x[1])
    with open("scores.txt", "w") as fout:
        for key, val in scoresSorted:
            fout.write("{} : {}\n".format(key, val))


if __name__ == "__main__":
    main()