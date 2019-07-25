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

r"""Decode from trained T2T models.

This binary performs inference using the Estimator API.

Example usage to decode from dataset:

  t2t-decoder \
      --data_dir ~/data \
      --problem=algorithmic_identity_binary40 \
      --model=transformer
      --hparams_set=transformer_base

Set FLAGS.decode_interactive or FLAGS.decode_from_file for alternative decode
sources.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
import csv
from subprocess import call

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import bleu_hook

import tensorflow as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
                    "must be in the format input \t target.")
flags.DEFINE_bool("decode_in_memory", False, "Decode in memory.")
flags.DEFINE_string("reference", None, "Path to the reference translation file, "
                    "used for uncertainty benchmarking.")
flags.DEFINE_bool("mc_sampling", False, "Whether or not to turn on MC sampling for "
                  "uncertainty evaluation.")
flags.DEFINE_integer("mc_dropout_seed", None, "Random seed for dropout turned on "
                     "during the MC sampling stage.")


def create_hparams(mc_dropout_seed=None):
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      mc_dropout_seed=mc_dropout_seed,
      problem_name=FLAGS.problem)


def create_decode_hparams(mc_sampling=False, uncertainty_over_prob=False):
  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.shards = FLAGS.decode_shards
  decode_hp.shard_id = FLAGS.worker_id
  decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
  decode_hp.decode_in_memory = decode_in_memory
  decode_hp.decode_to_file = FLAGS.decode_to_file
  decode_hp.decode_reference = FLAGS.decode_reference
  decode_hp.mc_sampling = mc_sampling
  decode_hp.uncertainty_over_prob=uncertainty_over_prob
  if mc_sampling:
    decode_hp.beam_size = 1
  return decode_hp


def decode(estimator, hparams, decode_hp, seq_prob_result=None):
  """Decode from estimator. From file with confidence."""
  result = []
  if seq_prob_result is None:
    result = decoding.decode_from_file(estimator, FLAGS.decode_from_file, hparams,
                                       decode_hp, FLAGS.decode_to_file,
                                       checkpoint_path=FLAGS.checkpoint_path)
  else:
    result = decoding.decode_from_file_with_confidence(estimator, 
                                        FLAGS.decode_from_file, hparams,
                                        decode_hp, seq_prob_result,
                                        FLAGS.decode_to_file,
                                        checkpoint_path=FLAGS.checkpoint_path)
  if FLAGS.checkpoint_path and FLAGS.keep_timestamp:
    ckpt_time = os.path.getmtime(FLAGS.checkpoint_path + ".index")
    os.utime(FLAGS.decode_to_file, (ckpt_time, ckpt_time))
  return result


def accumulated_bleu(reference, decodes, sorted_key, uncertainty, name):
  tf.logging.info("Calculating the accumulated BLEU scores ordered by " + name) 
  ref_lines = [reference[sorted_key[index]] for index in range(len(sorted_key))]
  hyp_lines = [decodes[sorted_key[index]] for index in range(len(sorted_key))]
  #if not case_sensitive:
  ref_lines = [x.lower() for x in ref_lines]
  hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_hook.bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_hook.bleu_tokenize(x) for x in hyp_lines] 
  hyp_lengths = [len(x) for x in hyp_tokens]
  accumulated_bleus = 100 * bleu_hook.compute_accumulated_bleu(ref_tokens, hyp_tokens)

  individual_bleus = []
  for (ref, hyp) in zip(ref_tokens, hyp_tokens):
    individual_bleu = 100 * bleu_hook.compute_bleu(ref, hyp)
    individual_bleus.append(individual_bleu)

  csv_filename = FLAGS.decode_to_file + "." + name + ".csv"
  tf.logging.info("Writing " + name + " into %s" % csv_filename)
  np.savetxt(csv_filename, np.column_stack((uncertainty, accumulated_bleus, individual_bleus, hyp_lengths)),
              delimiter=",", fmt='%s')


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)


  # ====================== Sequence Prob Prediction =====================
  hp = create_hparams()
  decode_hp = create_decode_hparams(uncertainty_over_prob=True)

  estimator = trainer_lib.create_estimator(
      FLAGS.model,
      hp,
      t2t_trainer.create_run_config(hp),
      decode_hparams=decode_hp,
      use_tpu=FLAGS.use_tpu)

  seq_prob_result, bs_scores, bs_log_probs = decode(estimator, hp, decode_hp)

  if not FLAGS.mc_sampling:
    return

  # Recalculate scores and log_probs
  hp = create_hparams()
  decode_hp = create_decode_hparams()

  estimator = trainer_lib.create_estimator(
      FLAGS.model,
      hp,
      t2t_trainer.create_run_config(hp),
      decode_hparams=decode_hp,
      use_tpu=FLAGS.use_tpu)

  result, seq_prob_scores, seq_prob_log_probs, seq_prob_token_log_probs = decode(estimator, hp, decode_hp, seq_prob_result)
  all_token_log_probs = [seq_prob_token_log_probs]
  # __________________________________________________
  try:
    assert seq_prob_result == result
  except AssertionError:
    seq_prob_result = np.array(seq_prob_result).flatten('F')
    result = np.array(result).flatten('F')
    a = np.setdiff1d(seq_prob_result, result)
    b = np.setdiff1d(result, seq_prob_result)
    tf.logging.info("a-b")
    tf.logging.info(a)
    tf.logging.info("b-a")
    tf.logging.info(b)
    tf.logging.info(print("Assertion error line 169"))
  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  # tpu_seq_prob_result = ['Reports in Australien sagten, dass sie in der Zwischenzeit Urlaub in der Ferienregion Krabi in Süd-Thailand.', 'Bamford wurde von einer lokalen Rechtsanwältin in Phuket vertreten, warnte aber, dass die Berufung dazu führen könnte, dass das Gericht ihr Urteil um bis zu zwei Jahre verlängert und sie zwingt, es in einem erwachsenen Gefängnis zu verbüßen.', 'Nach der jüngsten Ermordung des australischen Reiseagenten Michelle Smith in Phuket, Thailand, könnte auch versucht werden, sein beschädigtes Touristenimage zu reparieren, was zu einem Freispruch führt.']
  # seq_prob_result = tpu_seq_prob_result
  
  # cpu_seq_prob_result = ['Reports in Australien sagten, dass sie in der Zwischenzeit Urlaub in der Ferienregion Krabi in Süd-Thailand.', 'Bamford wurde von einer lokalen Rechtsanwältin in Phuket vertreten, warnte aber, dass die Berufung dazu führen könnte, dass das Gericht ihre Strafe um bis zu zwei Jahre verlängert und sie zwingt, sie in einem erwachsenen Gefängnis zu verbüßen.', 'Nach dem jüngsten Mord an dem australischen Reiseagenten Michelle Smith in Phuket, Thailand, könnte auch versucht werden, sein beschädigtes Touristenimage zu reparieren, was zu einem Freispruch führt.']
  # seq_prob_result = cpu_seq_prob_result

  # ==================== Model Confidence Calculation ===================

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-> hp
  num_MC_samples=20 #------------------------!!!

  mc_scores = np.array([0] * len(seq_prob_scores), dtype=float)
  mc_log_probs = np.array([0] * len(seq_prob_log_probs), dtype=float)

  mc_dropout_seeds = np.random.randint(1000000, size=(num_MC_samples,2))
  # mc_dropout_seeds = np.array([[853751, 85362], [529532, 454878], [446227, 437121], [875822, 31542], [476300, 999464], [161050, 549147], [27724, 731808], [98251, 977235], [847405, 584430], [430167, 582189]])
  # mc_dropout_seeds = np.array([[853751, 85362], [853751, 85362]])
  for i in range(num_MC_samples):

    mc_dropout_seed = None
    if FLAGS.mc_sampling:
      tf.logging.info("------------ MC Sampling: {}/{} -------------".format((i+1), num_MC_samples))
      mc_dropout_seed = mc_dropout_seeds[i]

    # hp = create_hparams() #!!!!!!!!!!!!!!!!!!! both this and the next line
    # decode_hp = create_decode_hparams(mc_sampling=FLAGS.mc_sampling)

    hp = create_hparams(mc_dropout_seed=mc_dropout_seed)
    decode_hp = create_decode_hparams(mc_sampling=FLAGS.mc_sampling)

    estimator = trainer_lib.create_estimator(
        FLAGS.model,
        hp,
        t2t_trainer.create_run_config(hp),
        decode_hparams=decode_hp,
        use_tpu=FLAGS.use_tpu)

    result, scores, log_probs, token_log_probs = decode(estimator, hp, decode_hp, seq_prob_result)
    all_token_log_probs.append(token_log_probs)
    # __________________________________________________
    try:
      assert np.array_equal(seq_prob_result, result)
    except AssertionError:
      seq_prob_result = np.array(seq_prob_result).flatten('F')
      result = np.array(result).flatten('F')
      a = np.setdiff1d(seq_prob_result, result)
      b = np.setdiff1d(result, seq_prob_result)
      tf.logging.info("a-b")
      tf.logging.info(a)
      tf.logging.info("b-a")
      tf.logging.info(b)
      tf.logging.info(print("Assertion error line 242"))
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    mc_scores += np.array(scores).flatten('F')
    mc_log_probs += np.array(log_probs).flatten('F')

  mc_scores = mc_scores / num_MC_samples
  mc_log_probs = mc_log_probs / num_MC_samples

  tf.logging.info("Finshed MC sampling, MC dropout random seeds:")
  tf.logging.info(mc_dropout_seeds.tolist())
  
  # ======================= Confidence ===========================
  decodes = np.array(seq_prob_result).flatten('F')

  tf.logging.info("Ordering the seqence prob result by BS scores and BS log probs.")
  sorted_bs_scores_key = sorted(range(len(bs_scores)), key=lambda k: bs_scores[k], reverse=True)
  sorted_bs_log_probs_key = sorted(range(len(bs_log_probs)), key=lambda k: bs_log_probs[k], reverse=True) 

  tf.logging.info("Ordering the seqence prob result by seq_prob scores and seq_prob log probs.")
  sorted_seq_prob_scores_key = sorted(range(len(seq_prob_scores)), key=lambda k: seq_prob_scores[k], reverse=True)
  sorted_seq_prob_log_probs_key = sorted(range(len(seq_prob_log_probs)), key=lambda k: seq_prob_log_probs[k], reverse=True) 

  tf.logging.info("Ordering the seqence prob result by MC scores and MC log probs.")
  sorted_mc_scores_key = sorted(range(len(mc_scores)), key=lambda k: mc_scores[k], reverse=True)
  sorted_mc_log_probs_key = sorted(range(len(mc_log_probs)), key=lambda k: mc_log_probs[k], reverse=True)  

  # Reorder according to the sorted scores and log probs
  bs_scores = [bs_scores[sorted_bs_scores_key[index]] for index in range(len(sorted_bs_scores_key))]
  bs_log_probs = [bs_log_probs[sorted_bs_log_probs_key[index]] for index in range(len(sorted_bs_log_probs_key))]

  seq_prob_scores = [seq_prob_scores[sorted_seq_prob_scores_key[index]] for index in range(len(sorted_seq_prob_scores_key))]
  seq_prob_log_probs = [seq_prob_log_probs[sorted_seq_prob_log_probs_key[index]] for index in range(len(sorted_seq_prob_log_probs_key))]

  mc_scores = [mc_scores[sorted_mc_scores_key[index]] for index in range(len(sorted_mc_scores_key))]
  mc_log_probs = [mc_log_probs[sorted_mc_log_probs_key[index]] for index in range(len(sorted_mc_log_probs_key))]


  # ------------ For accumulated BLEU ------------------
  # Calculating the accumulated BLEU scores
  reference = text_encoder.native_to_unicode(
      tf.gfile.Open(FLAGS.reference, "r").read()).split("\n")

  accumulated_bleu(reference, decodes, sorted_bs_scores_key, bs_scores, "bs_scores")
  accumulated_bleu(reference, decodes, sorted_bs_log_probs_key, bs_log_probs, "bs_log_probs")

  accumulated_bleu(reference, decodes, sorted_seq_prob_scores_key, seq_prob_scores, "seq_prob_scores")
  accumulated_bleu(reference, decodes, sorted_seq_prob_log_probs_key, seq_prob_log_probs, "seq_prob_log_probs")

  accumulated_bleu(reference, decodes, sorted_mc_scores_key, mc_scores, "mc_scores")
  accumulated_bleu(reference, decodes, sorted_mc_log_probs_key, mc_log_probs, "mc_log_probs")

  # ------------ For token log prob ------------------
  aligned_token_log_probs = [seq for tup in zip(*all_token_log_probs) for seq in tup]
  csv_filename = FLAGS.decode_to_file + ".tokens_log_probs.csv"
  tf.logging.info("Writing tokens_log_probs into %s" % csv_filename)
  outfile = open(csv_filename, "w")
  writer = csv.writer(outfile)
  writer.writerows(aligned_token_log_probs)
  outfile.close()

  # num_of_len_subset = 10
  # sorted_hyp_lengths = sorted(hyp_lengths)
  # splited_hyp_lengths = np.array_split(sorted_hyp_lengths, num_of_len_subset)

  # for i in range(num_of_len_subset):
  #   ref_tokens_sub = []
  #   hyp_tokens_sub = []
  #   hyp_lengths_sub = []
  #   uncertainties_sub = []
  #   for (ref_token, hyp_token, hyp_length, variance) in zip(ref_tokens, hyp_tokens, hyp_lengths, uncertainties):
  #     if hyp_length in splited_hyp_lengths[i]:
  #       ref_tokens_sub.append(ref_token)
  #       hyp_tokens_sub.append(hyp_token)
  #       hyp_lengths_sub.append(hyp_length)
  #       uncertainties_sub.append(variance)
  #   accumulated_bleus_sub = 100 * bleu_hook.compute_accumulated_bleu(ref_tokens_sub, hyp_tokens_sub)
  #   csv_filename = decode_to_file + ".uncertainties_sub" + str(i) + ".csv"
  #   tf.logging.info("Writing uncertainties into %s" % csv_filename)
  #   np.savetxt(csv_filename, np.column_stack((uncertainties_sub, accumulated_bleus_sub, hyp_lengths_sub)), 
  #               delimiter=",", fmt='%s')

  # tf.logging.info("Calculating the individual BLEU scores for each MC mean samples towards "
  #                 "target output.") 
  # individual_bleus = []
  # for (ref, hyp) in zip(ref_tokens, hyp_tokens):
  #   individual_bleu = 100 * bleu_hook.compute_bleu(ref, hyp)
  #   individual_bleus.append(individual_bleu)

  # csv_filename = ""
  # if decode_hp.uncertainty_over_prob:
  #   csv_filename = FLAGS.decode_to_file + ".prob_full.csv"
  #   tf.logging.info("Writing sequence prob into %s" % csv_filename)
  # else:
  #   csv_filename = FLAGS.decode_to_file + ".variance_full.csv"
  #   tf.logging.info("Writing variance into %s" % csv_filename)
  # np.savetxt(csv_filename, np.column_stack((uncertainties, accumulated_bleus, hyp_lengths)),
  #             delimiter=",", fmt='%s')

  # ----------------------------------------------------

  # mean_csv_filename = decode_to_file + ".mean"
  # tf.logging.info("Writing means into %s" % mean_csv_filename)
  # outfile = tf.gfile.Open(mean_csv_filename, "w")
  # for index in range(len(sorted_inputs) // num_MC_samples):
  #   outfile.write("%s%s" % (mean_samples[index], decode_hp.delimiter))
  # outfile.flush()
  # outfile.close()




if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
