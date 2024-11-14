import argparse
import sys
import os

# Add the path to your library
sys.path.append(os.path.abspath("mtg"))

# Now you can import the module
from mtg.scripts.train_drafter import main

FLAGS = argparse.Namespace()
FLAGS.expansion_fname = "pkl/BRO.pkl"
FLAGS.batch_size = 32 * 1
FLAGS.train_p = .9
FLAGS.emb_dim = 128
FLAGS.num_encoder_heads = 8
FLAGS.num_decoder_heads = 8
FLAGS.pointwise_ffn_width = 512
FLAGS.num_encoder_layers = 2
FLAGS.num_decoder_layers = 2
FLAGS.emb_dropout = 0.0
FLAGS.transformer_dropout = 0.1
FLAGS.lr_warmup = 2000
FLAGS.emb_margin = 1.0
FLAGS.emb_lambda = 0.5
FLAGS.rare_lambda = 10.0
FLAGS.cmc_lambda = 0.1
FLAGS.epochs = 2
FLAGS.verbose = True
FLAGS.model_name = "models/draft_BRO"
main(FLAGS)

