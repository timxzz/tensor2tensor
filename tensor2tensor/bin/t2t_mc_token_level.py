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


def create_hparams(mc_dropout_seed=None, mc_dropout_seeds=None):
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      mc_dropout_seed=mc_dropout_seed,
      mc_dropout_seeds=mc_dropout_seeds,
      problem_name=FLAGS.problem)


def create_decode_hparams():
  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.shards = FLAGS.decode_shards
  decode_hp.shard_id = FLAGS.worker_id
  decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
  decode_hp.decode_in_memory = decode_in_memory
  decode_hp.decode_to_file = FLAGS.decode_to_file
  decode_hp.decode_reference = FLAGS.decode_reference
  decode_hp.mc_sampling = FLAGS.mc_sampling
  return decode_hp


def decode(estimator, hparams, decode_hp):
  """Decode from estimator. Interactive, from file, or from dataset."""
  result = decoding.decode_from_file_mc_token(estimator, FLAGS.decode_from_file, hparams,
                                               decode_hp, FLAGS.decode_to_file,
                                               checkpoint_path=FLAGS.checkpoint_path)
  if FLAGS.checkpoint_path and FLAGS.keep_timestamp:
    ckpt_time = os.path.getmtime(FLAGS.checkpoint_path + ".index")
    os.utime(FLAGS.decode_to_file, (ckpt_time, ckpt_time))
  return result


def score_file(filename):
  """Score each line in a file and return the scores."""
  # Prepare model.
  hparams = create_hparams()
  encoders = registry.problem(FLAGS.problem).feature_encoders(FLAGS.data_dir)
  has_inputs = "inputs" in encoders

  # Prepare features for feeding into the model.
  if has_inputs:
    inputs_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    batch_inputs = tf.reshape(inputs_ph, [1, -1, 1, 1])  # Make it 4D.
  targets_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
  batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])  # Make it 4D.
  if has_inputs:
    features = {"inputs": batch_inputs, "targets": batch_targets}
  else:
    features = {"targets": batch_targets}

  # Prepare the model and the graph when model runs on features.
  model = registry.model(FLAGS.model)(hparams, tf.estimator.ModeKeys.EVAL)
  _, losses = model(features)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Load weights from checkpoint.
    ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
    ckpt = ckpts.model_checkpoint_path
    saver.restore(sess, ckpt)
    # Run on each line.
    with tf.gfile.Open(filename) as f:
      lines = f.readlines()
    results = []
    for line in lines:
      tab_split = line.split("\t")
      if len(tab_split) > 2:
        raise ValueError("Each line must have at most one tab separator.")
      if len(tab_split) == 1:
        targets = tab_split[0].strip()
      else:
        targets = tab_split[1].strip()
        inputs = tab_split[0].strip()
      # Run encoders and append EOS symbol.
      targets_numpy = encoders["targets"].encode(
          targets) + [text_encoder.EOS_ID]
      if has_inputs:
        inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
      # Prepare the feed.
      if has_inputs:
        feed = {inputs_ph: inputs_numpy, targets_ph: targets_numpy}
      else:
        feed = {targets_ph: targets_numpy}
      # Get the score.
      np_loss = sess.run(losses["training"], feed)
      results.append(np_loss)
  return results


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)


  if FLAGS.score_file:
    filename = os.path.expanduser(FLAGS.score_file)
    if not tf.gfile.Exists(filename):
      raise ValueError("The file to score doesn't exist: %s" % filename)
    results = score_file(filename)
    if not FLAGS.decode_to_file:
      raise ValueError("To score a file, specify --decode_to_file for results.")
    write_file = tf.gfile.Open(os.path.expanduser(FLAGS.decode_to_file), "w")
    for score in results:
      write_file.write("%.6f\n" % score)
    write_file.close()
    return

  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-> hp
  num_MC_samples=10 #------------------------!!!

  num_runs = 1
  if FLAGS.mc_sampling:
    num_runs = num_MC_samples

  results = []
  probs = []

  mc_dropout_seeds = None
  if FLAGS.mc_sampling:
    # mc_dropout_seeds = np.random.randint(1000000, size=(num_MC_samples,2))
    mc_dropout_seeds = np.array([[853751, 85362], [529532, 454878], [446227, 437121], [875822, 31542], [476300, 999464], [161050, 549147], [27724, 731808], [98251, 977235], [847405, 584430], [430167, 582189]])
  # mc_dropout_seeds = np.array([[853751, 85362], [853751, 85362]])

  hp = create_hparams(mc_dropout_seeds=mc_dropout_seeds)
  decode_hp = create_decode_hparams()

  estimator = trainer_lib.create_estimator(
      FLAGS.model,
      hp,
      t2t_trainer.create_run_config(hp),
      decode_hparams=decode_hp,
      use_tpu=FLAGS.use_tpu)

  result, prob_scores, _ = decode(estimator, hp, decode_hp)
  # if not (FLAGS.mc_sampling or decode_hp.uncertainty_over_prob):
  return



  results.append(result)
  probs = prob_scores

  if FLAGS.mc_sampling:
    tf.logging.info("Finshed MC sampling, MC dropout random seeds:")
    tf.logging.info(mc_dropout_seeds.tolist())
    
  decodes = np.array(results).flatten('F')
  # ======================= Uncertainty ===========================
  uncertainties = 0
  mean_samples = []
  sorted_uncertainties_key = []
  if decode_hp.uncertainty_over_prob:
    tf.logging.info("Calculating variance through prob of top beam.")
    uncertainties = probs
    sorted_uncertainties_key = sorted(range(len(uncertainties)), key=lambda k: uncertainties[k], reverse=True)
    mean_samples = decodes
  else:
    # Calculating variance of MC samples
    tf.logging.info("Calculating the mutual BLEU score "
                    "among the MC samples for each input.")
    bleu_MC_batchs = []
    MC_batchs = np.reshape(decodes, (-1, num_MC_samples))
    for one_batch in MC_batchs:
      bleu_MC_batch = []
      mean_bleu_sums = [0] * num_MC_samples
      for i,j in itertools.combinations(list(range(num_MC_samples)),2):
        # 1-BLEU as the larger the BLEU the closer two sentence are
        b = 100 * (1 - bleu_hook.bleu_one_pair(one_batch[i],one_batch[j]))
        bleu_MC_batch.append(b)
        # Adding the distance, the min corresponds to mean
        mean_bleu_sums[i] += b
        mean_bleu_sums[j] += b
      index_mean = min(range(len(mean_bleu_sums)), key=mean_bleu_sums.__getitem__)
      mean_samples.append(one_batch[index_mean])
      bleu_MC_batchs.append(bleu_MC_batch)

    bleu_square = np.array(bleu_MC_batchs) ** 2
    uncertainties = np.sum(bleu_square, axis=1) / (num_MC_samples ** 2)
    sorted_uncertainties_key = sorted(range(len(uncertainties)), key=lambda k: uncertainties[k])

  # Reorder according to sorted uncertainties
  uncertainties = [uncertainties[sorted_uncertainties_key[index]] for index in range(len(sorted_uncertainties_key))]
  # Calculating BLEU of each MC mean samples
  tf.logging.info("Calculating the accumulated BLEU scores for each MC mean samples towards "
                  "target output.") 
  ref_lines = text_encoder.native_to_unicode(
      tf.gfile.Open(FLAGS.reference, "r").read()).split("\n")
  ref_lines = [ref_lines[sorted_uncertainties_key[index]] for index in range(len(sorted_uncertainties_key))]
  hyp_lines = [mean_samples[sorted_uncertainties_key[index]] for index in range(len(sorted_uncertainties_key))]
  #if not case_sensitive:
  ref_lines = [x.lower() for x in ref_lines]
  hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_hook.bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_hook.bleu_tokenize(x) for x in hyp_lines]

  # ------------ For accumulated BLEU ------------------
  hyp_lengths = [len(x) for x in hyp_tokens]

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

  accumulated_bleus = 100 * bleu_hook.compute_accumulated_bleu(ref_tokens, hyp_tokens)

  tf.logging.info("Calculating the individual BLEU scores for each MC mean samples towards "
                  "target output.") 
  individual_bleus = []
  for (ref, hyp) in zip(ref_tokens, hyp_tokens):
    individual_bleu = 100 * bleu_hook.compute_bleu(ref, hyp)
    individual_bleus.append(individual_bleu)

  csv_filename = ""
  if decode_hp.uncertainty_over_prob:
    csv_filename = FLAGS.decode_to_file + ".prob_full.csv"
    tf.logging.info("Writing sequence prob into %s" % csv_filename)
  else:
    csv_filename = FLAGS.decode_to_file + ".variance_full.csv"
    tf.logging.info("Writing variance into %s" % csv_filename)
  np.savetxt(csv_filename, np.column_stack((uncertainties, accumulated_bleus, individual_bleus, hyp_lengths)),
              delimiter=",", fmt='%s')

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
