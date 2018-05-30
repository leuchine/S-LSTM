import sys
import numpy as np

def write_to_file(path, data):
    with open(path, "w") as f:
        for d in data:
            f.write(d)

dataset_name = sys.argv[1]
dev_percentage=0.1

data_points = []
with open("sst_data/%s_trn" % dataset_name) as f:
    for line in f:
        data_points += [line]
    np.random.shuffle(data_points)
      
    train_data_points = data_points[int(dev_percentage * len(data_points)): ]
    dev_data_points= data_points[:int(dev_percentage * len(data_points))]
    write_to_file("sst_data/%s_trn" % dataset_name, train_data_points)
    write_to_file("sst_data/%s_dev" % dataset_name, dev_data_points)