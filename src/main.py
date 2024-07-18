
import numpy as np
import jax.numpy as jnp
import numpy as np
import os
import torch
from torchvision import datasets

# Import of custom modules
import datahelper
import models


if __name__ == "__main__":
    # load data
    # IJCNN1
    dirData1 = "../data/ijcnn1/ijcnn1"
    testData1 = datahelper.load_ijcnn1(dirData1 + ".t")
    trainData1 = datahelper.load_ijcnn1(dirData1 + ".tr")
    # CoverType



