from pathlib import Path
from fastai.vision.all import *


def run_fast_classification():

    path = untar_data(URLs.IMAGENETTE_160)

    dls = ImageDataLoaders.from_folder(path, train='train', valid='val', 
                    item_tfms=RandomResizedCrop(128, min_scale=0.35), 
                    batch_tfms=Normalize.from_stats(*imagenet_stats))

    learn = cnn_learner(dls, resnet34, metrics=accuracy, pretrained=False)

    learn.fit_one_cycle(5, 1e-3)

    preds, targs = learn.get_preds()

    filename = 'results.pickle'
    with open(filename, 'wb') as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(preds, pickle_file)

if __name__ == "__main__":
    # local = (len(sys.argv) == 2 and sys.argv[1] == "local")
    run_fast_classification()