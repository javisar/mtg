import argparse
import sys
import os

# Add the path to your library
sys.path.append(os.path.abspath("mtg"))

# Now you can import the module
from mtg.scripts.train_builder import main

FLAGS = argparse.Namespace()
FLAGS.expansion_fname = "pkl/BRO.pkl"
FLAGS.batch_size = 32 * 1
FLAGS.train_p = 1.0
FLAGS.emb_dim = None
FLAGS.draft_model = "models/draft_BRO"
FLAGS.dropout = 0.2
FLAGS.lr_warmup = 4000
FLAGS.cmc_lambda = 0.1
FLAGS.epochs = 1
FLAGS.verbose = True
FLAGS.model_name = "models/build_BRO"
main(FLAGS)

