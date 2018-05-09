from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import sys
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6"
def main():
    # create instance of config
    config = Config()
    config.layer=int(sys.argv[1])
    config.step=int(sys.argv[2])

    if config.task=='pos':
        print("USING POS")
        config.filename_train = "data/train.pos" # test
        config.filename_dev= "data/dev.pos"
        config.filename_test= "data/test.pos"
    else:
        print("USING NER")      
    print("iteration: "+str(config.layer))
    print("step: "+str(config.step))

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    # train model
    model.train(train, dev, test)

if __name__ == "__main__":
    main()
