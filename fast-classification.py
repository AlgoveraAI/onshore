import json
import os
import pickle
import numpy as np

from fastai.vision.all import *

def get_input():

    dids = os.getenv('DIDS', None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f'data/inputs/{did}/0'  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename

def run_fast_classification():

    filename = get_input()
    if not filename:
        print("Could not retrieve filename.")
        return

    # path = untar_data(URLs.IMAGENETTE_160)

    # dls = ImageDataLoaders.from_folder(path, train='train', valid='val', 
    #                 item_tfms=RandomResizedCrop(128, min_scale=0.35), 
    #                 batch_tfms=Normalize.from_stats(*imagenet_stats))

    # learn = cnn_learner(dls, resnet34, metrics=accuracy, pretrained=False)

    # learn.fit_one_cycle(5, 1e-3)

    # preds, targs = learn.get_preds()

    preds = np.ones(5)

    filename = "/data/outputs/result"
    with open(filename, 'wb') as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(preds, pickle_file)

if __name__ == "__main__":
    run_fast_classification()