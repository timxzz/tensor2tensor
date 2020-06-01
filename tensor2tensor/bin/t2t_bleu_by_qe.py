# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate BLEU score for all checkpoints/translations in a given directory.

This script can be used in two ways.


To evaluate one already translated file:

```
t2t-bleu --translation=my-wmt13.de --reference=wmt13_deen.de
```

To evaluate all translations in a given directory (translated by
`t2t-translate-all`):

```
t2t-bleu
  --translations_dir=my-translations
  --reference=wmt13_deen.de
  --event_dir=events
```

In addition to the above-mentioned required parameters,
there are optional parameters:
 * bleu_variant: cased (case-sensitive), uncased, both (default).
 * tag_suffix: Default="", so the tags will be BLEU_cased and BLEU_uncased.
   tag_suffix can be used e.g. for different beam sizes if these should be
   plotted in different graphs.
 * min_steps: Don't evaluate checkpoints with less steps.
   Default=-1 means check the `last_evaluated_step.txt` file, which contains
   the number of steps of the last successfully evaluated checkpoint.
 * report_zero: Store BLEU=0 and guess its time based on the oldest file in the
   translations_dir. Default=True. This is useful, so TensorBoard reports
   correct relative time for the remaining checkpoints. This flag is set to
   False if min_steps is > 0.
 * wait_minutes: Wait upto N minutes for a new translated file. Default=0.
   This is useful for continuous evaluation of a running training, in which case
   this should be equal to save_checkpoints_secs/60 plus time needed for
   translation plus some reserve.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import itertools
from tensor2tensor.utils import bleu_hook
from tensor2tensor.data_generators import text_encoder
import tensorflow as tf
from collections import Counter
import numpy as np


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("source", None,
                    "Path to the source-language file to be translated")
flags.DEFINE_string("reference", None, "Path to the reference translation file")
flags.DEFINE_string("translation", None,
                    "Path to the MT system translation file")
flags.DEFINE_string("translations_dir", None,
                    "Directory with translated files to be evaluated.")
flags.DEFINE_string("event_dir", None, "Where to store the event file.")

flags.DEFINE_string("bleu_variant", "both",
                    "Possible values: cased(case-sensitive), uncased, "
                    "both(default).")
flags.DEFINE_string("tag_suffix", "",
                    "What to add to BLEU_cased and BLEU_uncased tags.")
flags.DEFINE_integer("min_steps", -1,
                     "Don't evaluate checkpoints with less steps.")
flags.DEFINE_integer("wait_minutes", 0,
                     "Wait upto N minutes for a new checkpoint, cf. "
                     "save_checkpoints_secs.")
flags.DEFINE_bool("report_zero", None,
                  "Store BLEU=0 and guess its time based on the oldest file.")
flags.DEFINE_integer("mc_samples", 1,
                     "Number of MC samples for calculating uncertainty.")
flags.DEFINE_string("all_sentences", None, "Path to the all_sentences file")
flags.DEFINE_string("training_data", None, "Path to the training data file")
flags.DEFINE_bool("clean_samples_nl", False, "Clean the MCdropout samples")
flags.DEFINE_bool("clean_samples_de", False, "Clean the MCdropout samples")
flags.DEFINE_bool("hardcode_bleu", False, "Calculate BLEU for hardcoded sentences")
flags.DEFINE_string("sentence_len_ratio", None, "Path to the all_sentences file")
flags.DEFINE_bool("qe_scores", False, "Use QE score file for ordering")

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.qe_scores:
    de_source = "/home/tim/Workspace/test_data/newstest2014.de"
    en_source = "/home/tim/Workspace/test_data/newstest2014.en"

    de_results = "/home/tim/Workspace/translation-qe/en_de.mt"
    de_qe = "/home/tim/Workspace/translation-qe/en_de.mt.qe4"

    de_ref_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(de_source, "r").read()).split("\n")

    de_hyp_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(de_results, "r").read()).split("\n")

    hyp_qe_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(de_qe, "r").read()).split("\n")

    hyp_qe_scores = [float(line) for line in hyp_qe_lines[:-1]]

    ref_lines = [x.lower() for x in de_ref_lines]
    hyp_lines = [x.lower() for x in de_hyp_lines]

    # QE score: the percentage of the sentence that you would need to change to create a correct translation
    sorted_qe_key = sorted(range(len(hyp_qe_scores)), key=lambda k: hyp_qe_scores[k])

    sorted_by_qe_ref = [ref_lines[sorted_qe_key[index]] for index in range(len(sorted_qe_key))]
    sorted_by_qe_hyp = [hyp_lines[sorted_qe_key[index]] for index in range(len(sorted_qe_key))]
    ref_tokens = [bleu_hook.bleu_tokenize(x) for x in sorted_by_qe_ref]
    hyp_tokens = [bleu_hook.bleu_tokenize(x) for x in sorted_by_qe_hyp]

    accumulated_bleus_qe = 100 * bleu_hook.compute_accumulated_bleu(ref_tokens, hyp_tokens)

    print(accumulated_bleus_qe)
    np.savetxt("AccBLEU_en2de_nt2014_QE_4.csv", accumulated_bleus_qe,
              delimiter=",", fmt='%s')



if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
