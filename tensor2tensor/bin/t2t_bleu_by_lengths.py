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

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.hardcode_bleu:
    s0="Basically vegan dishes are for everyone one two three four five."
    s1="Basically vegan dishes are for everyone one two three four five."
    s2="Basically, vegan dishes are there for everyone one two three four five."
    s3="Essentially, vegan dishes are available for everyone. one two three four five"
    s4="Basically vegan dishes are for everyone one two three four five."
    s5="Basically, vegane dishes are there for all one two three four five."
    s6="Basically vegan dishes are there for everyone one two three four five."
    s7="Basically, vegan dishes are there for all one two three four five."
    s8="Basically vegan dishes are there for everyone one two three four five."
    s9="Basically, vegan dishes are for everyone one two three four five."

    lines = [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9]

    num_MC_samples = 10
    bleu_MC_batch = []
    bleu_MC_batchs = []
    mean_bleu_sums = [0] * num_MC_samples
    mean_samples = []
    for i,j in itertools.permutations(list(range(num_MC_samples)),2):
      # 1-BLEU as the larger the BLEU the closer two sentence are
      b = 100 * (1 - bleu_hook.bleu_one_pair(lines[i],lines[j]))
      bleu_MC_batch.append(b)
      print("BLEU: {}".format(100 - b))
      print(lines[i])
      print(lines[j])
      print("####################")
      # Adding the distance, the min corresponds to mean
      mean_bleu_sums[i] += b
      mean_bleu_sums[j] += b
    index_mean = min(range(len(mean_bleu_sums)), key=mean_bleu_sums.__getitem__)
    mean_samples.append(lines[index_mean])
    # mean_samples = mle_result
    bleu_MC_batchs.append(bleu_MC_batch)

    bleu_square = np.array(bleu_MC_batchs) ** 2
    uncertainties = np.sum(bleu_square, axis=1) / (num_MC_samples ** 2)

    # print(bleu_MC_batch)
    print(uncertainties)


  if FLAGS.all_sentences:
    all_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(FLAGS.all_sentences, "r").read()).split("\n")

    uncertainties = all_lines[0::4]
    individual_bleu = all_lines[1::4]

    sorted_ref = all_lines[2::4]
    sorted_hyp = all_lines[3::4]

    ref_lines = [x.lower() for x in sorted_ref]
    hyp_lines = [x.lower() for x in sorted_hyp]
    ref_tokens = [bleu_hook.bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_hook.bleu_tokenize(x) for x in hyp_lines]

    ref_lengths = [len(x) for x in ref_tokens]
    hyp_lengths = [len(x) for x in hyp_tokens]

    ref_tokens_len10=[]
    ref_tokens_len20=[]
    ref_tokens_len30=[]
    ref_tokens_len40=[]
    ref_tokens_len50=[]
    ref_tokens_len50plus=[]

    hyp_tokens_len10=[]
    hyp_tokens_len20=[]
    hyp_tokens_len30=[]
    hyp_tokens_len40=[]
    hyp_tokens_len50=[]
    hyp_tokens_len50plus=[]

    BLEUVar_len10=[]
    BLEUVar_len20=[]
    BLEUVar_len30=[]
    BLEUVar_len40=[]
    BLEUVar_len50=[]
    BLEUVar_len50plus=[]

    for i in range(len(sorted_hyp)):
      if hyp_lengths[i] <= 10:
        ref_tokens_len10.append(ref_tokens[i])
        hyp_tokens_len10.append(hyp_tokens[i])
        BLEUVar_len10.append(float(uncertainties[i]))
      elif hyp_lengths[i] <= 20:
        ref_tokens_len20.append(ref_tokens[i])
        hyp_tokens_len20.append(hyp_tokens[i])
        BLEUVar_len20.append(float(uncertainties[i]))
      elif hyp_lengths[i] <= 30:
        ref_tokens_len30.append(ref_tokens[i])
        hyp_tokens_len30.append(hyp_tokens[i])
        BLEUVar_len30.append(float(uncertainties[i]))
      elif hyp_lengths[i] <= 40:
        ref_tokens_len40.append(ref_tokens[i])
        hyp_tokens_len40.append(hyp_tokens[i])
        BLEUVar_len40.append(float(uncertainties[i]))
      elif hyp_lengths[i] <= 50:
        ref_tokens_len50.append(ref_tokens[i])
        hyp_tokens_len50.append(hyp_tokens[i])
        BLEUVar_len50.append(float(uncertainties[i]))
      else:
        ref_tokens_len50plus.append(ref_tokens[i])
        hyp_tokens_len50plus.append(hyp_tokens[i])
        BLEUVar_len50plus.append(float(uncertainties[i]))

    # bleu10 = bleu_hook.compute_bleu(ref_tokens_len10, hyp_tokens_len10)
    # bleu20 = bleu_hook.compute_bleu(ref_tokens_len20, hyp_tokens_len20)
    # bleu30 = bleu_hook.compute_bleu(ref_tokens_len30, hyp_tokens_len30)
    # bleu40 = bleu_hook.compute_bleu(ref_tokens_len40, hyp_tokens_len40)
    # bleu50 = bleu_hook.compute_bleu(ref_tokens_len50, hyp_tokens_len50)
    # bleu50plus = bleu_hook.compute_bleu(ref_tokens_len50plus, hyp_tokens_len50plus)

    # print("BLEU Lengths  1-10: {}".format(bleu10))
    # print("BLEU Lengths 11-20: {}".format(bleu20))
    # print("BLEU Lengths 21-30: {}".format(bleu30))
    # print("BLEU Lengths 31-40: {}".format(bleu40))
    # print("BLEU Lengths 41-50: {}".format(bleu50))
    # print("BLEU Lengths 51+  : {}".format(bleu50plus))

    bleuvar10 = sum(BLEUVar_len10) / len(BLEUVar_len10)
    bleuvar20 = sum(BLEUVar_len20) / len(BLEUVar_len20)
    bleuvar30 = sum(BLEUVar_len30) / len(BLEUVar_len30)
    bleuvar40 = sum(BLEUVar_len40) / len(BLEUVar_len40)
    bleuvar50 = sum(BLEUVar_len50) / len(BLEUVar_len50)
    bleuvar50plus = sum(BLEUVar_len50plus) / len(BLEUVar_len50plus)
    print("BLEUVar Lengths  1-10: {}".format(bleuvar10))
    print("BLEUVar Lengths 11-20: {}".format(bleuvar20))
    print("BLEUVar Lengths 21-30: {}".format(bleuvar30))
    print("BLEUVar Lengths 31-40: {}".format(bleuvar40))
    print("BLEUVar Lengths 41-50: {}".format(bleuvar50))
    print("BLEUVar Lengths 51+  : {}".format(bleuvar50plus))

  if FLAGS.clean_samples_nl:
    pwd = "/home/tim/Workspace/translation-acl/de+nl2en/"
    nl_samples_dir = ["do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-306481_730954.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-354619_782949.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-4079_408351.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-529201_462722.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-536474_973488.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-740518_829690.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-831960_391823.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-868498_15792.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-963161_57822.bs",
                      "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.seed-999606_639192.bs"]
    nl_results = "do_10_data_nl2en-nc-v14_model_de2en-full_350k_bleuvar.en.variance_full.all_sentences.csv"
    nl_source = "/home/tim/Workspace/test_data/nc_v14_3k.nl"
    en_source = "/home/tim/Workspace/test_data/nc_v14_3k.en"

    samples1 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[0], "r").read()).split("\n")
    samples2 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[1], "r").read()).split("\n")
    samples3 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[2], "r").read()).split("\n")
    samples4 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[3], "r").read()).split("\n")
    samples5 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[4], "r").read()).split("\n")
    samples6 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[5], "r").read()).split("\n")
    samples7 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[6], "r").read()).split("\n")
    samples8 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[7], "r").read()).split("\n")
    samples9 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[8], "r").read()).split("\n")
    samples10 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_samples_dir[9], "r").read()).split("\n")

    all_results = text_encoder.native_to_unicode(tf.gfile.Open(pwd+nl_results, "r").read()).split("\n")
    sorted_hyp = all_results[3::4]

    nl_source = text_encoder.native_to_unicode(tf.gfile.Open(nl_source, "r").read()).split("\n")
    en_source = text_encoder.native_to_unicode(tf.gfile.Open(en_source, "r").read()).split("\n")

    all_samples = [None]*(len(samples1)*10)
    all_samples[::10] = samples1
    all_samples[1::10] = samples2
    all_samples[2::10] = samples3
    all_samples[3::10] = samples4
    all_samples[4::10] = samples5
    all_samples[5::10] = samples6
    all_samples[6::10] = samples7
    all_samples[7::10] = samples8
    all_samples[8::10] = samples9
    all_samples[9::10] = samples10


    all_samples_sorted=[]
    nl_source_sorted=[]
    en_source_sorted=[]
    for hyp in sorted_hyp:
      idx = all_samples.index(hyp)
      all_samples_sorted.append(all_samples[int((idx//10)*10):(int(idx//10)*10 + 10)])
      nl_source_sorted.append(nl_source[int(idx//10)])
      en_source_sorted.append(en_source[int(idx//10)])


    # np.savetxt("nl2en_all_10samples_sorted", all_samples_sorted,
    #           delimiter="\n", fmt='%s')
    np.savetxt("nl2en_nl_source_sorted", nl_source_sorted,
              delimiter="\n", fmt='%s')
    np.savetxt("nl2en_en_source_sorted", en_source_sorted,
              delimiter="\n", fmt='%s')


  if FLAGS.clean_samples_de:
    pwd = "/home/tim/Workspace/translation-acl/de+nl2en/"
    de_samples_dir = ["do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-306481_730954.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-354619_782949.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-4079_408351.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-529201_462722.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-536474_973488.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-740518_829690.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-831960_391823.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-868498_15792.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-963161_57822.bs",
                      "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.seed-999606_639192.bs"]
    de_results = "do_10_data_de2en-nt2014_model_de2en-full_350k_bleuvar.en.variance_full.all_sentences.csv"
    de_source_path = "/home/tim/Workspace/test_data/newstest2014.de"
    en_source_path = "/home/tim/Workspace/test_data/newstest2014.en"

    samples1 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[0], "r").read()).split("\n")
    samples2 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[1], "r").read()).split("\n")
    samples3 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[2], "r").read()).split("\n")
    samples4 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[3], "r").read()).split("\n")
    samples5 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[4], "r").read()).split("\n")
    samples6 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[5], "r").read()).split("\n")
    samples7 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[6], "r").read()).split("\n")
    samples8 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[7], "r").read()).split("\n")
    samples9 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[8], "r").read()).split("\n")
    samples10 = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_samples_dir[9], "r").read()).split("\n")

    all_results = text_encoder.native_to_unicode(tf.gfile.Open(pwd+de_results, "r").read()).split("\n")
    sorted_hyp = all_results[3::4]

    de_source = text_encoder.native_to_unicode(tf.gfile.Open(de_source_path, "r").read()).split("\n")
    en_source = text_encoder.native_to_unicode(tf.gfile.Open(en_source_path, "r").read()).split("\n")

    all_samples = [None]*(len(samples1)*10)
    all_samples[::10] = samples1
    all_samples[1::10] = samples2
    all_samples[2::10] = samples3
    all_samples[3::10] = samples4
    all_samples[4::10] = samples5
    all_samples[5::10] = samples6
    all_samples[6::10] = samples7
    all_samples[7::10] = samples8
    all_samples[8::10] = samples9
    all_samples[9::10] = samples10


    all_samples_sorted=[]
    de_source_sorted=[]
    en_source_sorted=[]
    for hyp in sorted_hyp:
      idx = all_samples.index(hyp)
      all_samples_sorted.append(all_samples[int((idx//10)*10):(int(idx//10)*10 + 10)])
      de_source_sorted.append(de_source[int(idx//10)])
      en_source_sorted.append(en_source[int(idx//10)])


    np.savetxt("de2en_all_10samples_sorted", all_samples_sorted,
              delimiter="\n", fmt='%s')
    np.savetxt("de2en_de_source_sorted", de_source_sorted,
              delimiter="\n", fmt='%s')
    np.savetxt("de2en_en_source_sorted", en_source_sorted,
              delimiter="\n", fmt='%s')


  if FLAGS.training_data:
    pwd = "/home/tim/tmp/t2t_tmp/"
    nc_path = "training-parallel-nc-v13/news-commentary-v13.de-en.en"
    cc_path = "commoncrawl.de-en.en"
    europarl_path = "training/europarl-v7.de-en.en"

    all_lengths=[]

    all_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(pwd+nc_path, "r").read()).split("\n")
    print("1")
    all_lines_lower = [x.lower() for x in all_lines]
    all_tokens = [bleu_hook.bleu_tokenize(x) for x in all_lines_lower]
    all_lengths1=[len(x) for x in all_tokens]
    print("1.")
    all_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(pwd+cc_path, "r").read()).split("\n")
    print("2")
    all_lines_lower = [x.lower() for x in all_lines]
    all_tokens = [bleu_hook.bleu_tokenize(x) for x in all_lines_lower]
    all_lengths2=[len(x) for x in all_tokens]
    print("2.")
    all_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(pwd+europarl_path, "r").read()).split("\n")
    print("3")
    all_lines_lower = [x.lower() for x in all_lines]
    all_tokens = [bleu_hook.bleu_tokenize(x) for x in all_lines_lower]
    all_lengths3=[len(x) for x in all_tokens]
    print("3.")

    all_lengths.extend(all_lengths1)
    all_lengths.extend(all_lengths2)
    all_lengths.extend(all_lengths3)
    print("4")
    count = Counter(all_lengths)

    print(count)


  if FLAGS.sentence_len_ratio:

    de_source = "/home/tim/Workspace/test_data/newstest2014.de"
    en_source = "/home/tim/Workspace/test_data/newstest2014.en"

    de_source_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(de_source, "r").read()).split("\n")
    en_source_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(en_source, "r").read()).split("\n")

    all_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(FLAGS.sentence_len_ratio, "r").read()).split("\n")

    uncertainties = all_lines[0::4]
    individual_bleu = all_lines[1::4]

    sorted_ref = all_lines[2::4]
    sorted_hyp = all_lines[3::4]

    sorted_de_source=[]
    for line in sorted_ref:
      idx = en_source_lines.index(line)
      sorted_de_source.append(de_source_lines[idx])

    ref_lines = [x.lower() for x in sorted_ref]
    hyp_lines = [x.lower() for x in sorted_hyp]
    de_src_lines = [x.lower() for x in sorted_de_source]
    ref_tokens = [bleu_hook.bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_hook.bleu_tokenize(x) for x in hyp_lines]
    de_src_tokens = [bleu_hook.bleu_tokenize(x) for x in de_src_lines]

    hyp_lengths = [len(x) for x in hyp_tokens]
    de_src_lengths = [len(x) for x in de_src_tokens]

    max_hyp_len = max(hyp_lengths)
    hyp_len_ratio = [x / max_hyp_len for x in hyp_lengths]

    # max_hyp_len = max(ref_lengths)
    # hyp_len_ratio = [x / max_hyp_len for x in ref_lengths]

    sorted_len_ratio_key = sorted(range(len(hyp_len_ratio)), key=lambda k: hyp_len_ratio[k])

    sorted_by_len_ratio_ref = [ref_lines[sorted_len_ratio_key[index]] for index in range(len(sorted_len_ratio_key))]
    sorted_by_len_ratio_hyp = [hyp_lines[sorted_len_ratio_key[index]] for index in range(len(sorted_len_ratio_key))]
    ref_tokens = [bleu_hook.bleu_tokenize(x) for x in sorted_by_len_ratio_ref]
    hyp_tokens = [bleu_hook.bleu_tokenize(x) for x in sorted_by_len_ratio_hyp]

    # print([len(x) for x in ref_tokens])

    accumulated_bleus_len_ratio = 100 * bleu_hook.compute_accumulated_bleu(ref_tokens, hyp_tokens)
    # bleu_len_p1 = 100 * bleu_hook.compute_bleu(ref_tokens[:300], hyp_tokens[:300])
    # bleu_len_p2 = 100 * bleu_hook.compute_bleu(ref_tokens[:600], hyp_tokens[:600])
    # bleu_len_p3 = 100 * bleu_hook.compute_bleu(ref_tokens[:900], hyp_tokens[:900])
    # bleu_len_p4 = 100 * bleu_hook.compute_bleu(ref_tokens[:1200], hyp_tokens[:1200])

    rand_key = np.random.permutation(len(sorted_len_ratio_key))
    rand_ref = [ref_lines[rand_key[index]] for index in range(len(rand_key))]
    rand_hyp = [hyp_lines[rand_key[index]] for index in range(len(rand_key))]
    ref_tokens = [bleu_hook.bleu_tokenize(x) for x in rand_ref]
    hyp_tokens = [bleu_hook.bleu_tokenize(x) for x in rand_hyp]

    accumulated_bleus_rand = 100 * bleu_hook.compute_accumulated_bleu(ref_tokens, hyp_tokens)

    # bleu_rand_p1 = 100 * bleu_hook.compute_bleu(ref_tokens[:300], hyp_tokens[:300])
    # bleu_rand_p2 = 100 * bleu_hook.compute_bleu(ref_tokens[:600], hyp_tokens[:600])
    # bleu_rand_p3 = 100 * bleu_hook.compute_bleu(ref_tokens[:900], hyp_tokens[:900])
    # bleu_rand_p4 = 100 * bleu_hook.compute_bleu(ref_tokens[:1200], hyp_tokens[:1200])

    # np.savetxt("bleuvar_de2en_len_ration_and_rand.csv", np.column_stack((accumulated_bleus_len_ratio, accumulated_bleus_rand)),
    #           delimiter=",", fmt='%s')
    np.savetxt("test-bleuvar_de2en_len_ration_and_rand.csv", np.column_stack((accumulated_bleus_len_ratio, accumulated_bleus_rand)),
              delimiter=",", fmt='%s')


# def _main(_):
#   tf.logging.set_verbosity(tf.logging.INFO)
#   if FLAGS.translation:
#     if FLAGS.translations_dir:
#       raise ValueError(
#           "Cannot specify both --translation and --translations_dir.")
#     if FLAGS.bleu_variant in ("uncased", "both"):
#       bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, FLAGS.translation,
#                                           case_sensitive=False, 
#                                           num_MC_samples=FLAGS.mc_samples)
#       print("BLEU_uncased = %6.2f" % bleu)
#     if FLAGS.bleu_variant in ("cased", "both"):
#       bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, FLAGS.translation,
#                                           case_sensitive=True)
#       print("BLEU_cased = %6.2f" % bleu)
#     return

#   if not FLAGS.translations_dir:
#     raise ValueError(
#         "Either --translation or --translations_dir must be specified.")
#   transl_dir = os.path.expanduser(FLAGS.translations_dir)
#   if not os.path.exists(transl_dir):
#     exit_time = time.time() + FLAGS.wait_minutes * 60
#     tf.logging.info("Translation dir %s does not exist, waiting till %s.",
#                     transl_dir, time.asctime(time.localtime(exit_time)))
#     while not os.path.exists(transl_dir):
#       time.sleep(10)
#       if time.time() > exit_time:
#         raise ValueError("Translation dir %s does not exist" % transl_dir)

#   last_step_file = os.path.join(FLAGS.event_dir, "last_evaluated_step.txt")
#   if FLAGS.min_steps == -1:
#     if tf.gfile.Exists(last_step_file):
#       with open(last_step_file) as ls_file:
#         FLAGS.min_steps = int(ls_file.read())
#     else:
#       FLAGS.min_steps = 0
#   if FLAGS.report_zero is None:
#     FLAGS.report_zero = FLAGS.min_steps == 0

#   writer = tf.summary.FileWriter(FLAGS.event_dir)
#   for transl_file in bleu_hook.stepfiles_iterator(
#       transl_dir, FLAGS.wait_minutes, FLAGS.min_steps, path_suffix=""):
#     # report_zero handling must be inside the for-loop,
#     # so we are sure the transl_dir is already created.
#     if FLAGS.report_zero:
#       all_files = (os.path.join(transl_dir, f) for f in os.listdir(transl_dir))
#       start_time = min(
#           os.path.getmtime(f) for f in all_files if os.path.isfile(f))
#       values = []
#       if FLAGS.bleu_variant in ("uncased", "both"):
#         values.append(tf.Summary.Value(
#             tag="BLEU_uncased" + FLAGS.tag_suffix, simple_value=0))
#       if FLAGS.bleu_variant in ("cased", "both"):
#         values.append(tf.Summary.Value(
#             tag="BLEU_cased" + FLAGS.tag_suffix, simple_value=0))
#       writer.add_event(tf.summary.Event(summary=tf.Summary(value=values),
#                                         wall_time=start_time, step=0))
#       FLAGS.report_zero = False

#     filename = transl_file.filename
#     tf.logging.info("Evaluating " + filename)
#     values = []
#     if FLAGS.bleu_variant in ("uncased", "both"):
#       bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, filename,
#                                           case_sensitive=False)
#       values.append(tf.Summary.Value(tag="BLEU_uncased" + FLAGS.tag_suffix,
#                                      simple_value=bleu))
#       tf.logging.info("%s: BLEU_uncased = %6.2f" % (filename, bleu))
#     if FLAGS.bleu_variant in ("cased", "both"):
#       bleu = 100 * bleu_hook.bleu_wrapper(FLAGS.reference, filename,
#                                           case_sensitive=True)
#       values.append(tf.Summary.Value(tag="BLEU_cased" + FLAGS.tag_suffix,
#                                      simple_value=bleu))
#       tf.logging.info("%s: BLEU_cased = %6.2f" % (transl_file.filename, bleu))
#     writer.add_event(tf.summary.Event(
#         summary=tf.Summary(value=values),
#         wall_time=transl_file.mtime, step=transl_file.steps))
#     writer.flush()
#     with open(last_step_file, "w") as ls_file:
#       ls_file.write(str(transl_file.steps) + "\n")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
