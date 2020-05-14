""" XLNet as a language model"""
import tensorflow as tf
import sys
import os
import numpy as np

import modeling
import xlnet
import model_utils

class XLNetLM:
    """ Using XLNet pretrained model as a language model"""
    def __init__(self, model_path):
        config_file = os.path.join(model_path, 'xlnet_config.json')
        spiece_file = os.path.join(model_path, 'spiece.model')
        ckpt_path = os.path.join(model_path, 'xlnet_model.ckpt')
        self.predict_batch_size = 100
        self.max_seq_length = 256
        self.max_predictions_per_seq = 20

        is_training = False

        # construct xlnet config and save to model_dir
        xlnet_config = xlnet.XLNetConfig(json_path=config_file)

        # construct run config from FLAGS
        # self.run_config = xlnet.create_run_config(is_training, False, FLAGS)
        run_config = xlnet.RunConfig(is_training, False, False, 0.0, 0.0,
         init="normal", init_range=0.1, init_std=0.02, mem_len=None,
         reuse_len=256, bi_data=False, clamp_len=-1, same_length=False)

        graph = tf.Graph()
        with graph.as_default():
            self.session = tf.Session()

            # shape: max_sentence_length x num_sentence
            self.input_ids = tf.placeholder(tf.int32, [None, None], name="input_ids")
            # shape: max_sentence_length x num_sentence
            self.seg_ids = tf.placeholder(tf.int32, [None, None], name="seg_ids")
            # shape: max_sentence_length x num_sentence
            self.input_mask = tf.placeholder(tf.int32, [None, None], name="input_mask")
            # shape: max_sentence_length x max_sentence_length x num_sentence
            self.perm_mask = tf.placeholder(tf.int32, [None, None, None], name="perm_mask")
            # shape: max_predictions_per_seq x max_sentence_length x num_sentence
            self.target_mapping = tf.placeholder(tf.int32, [None, None, None], name="target_mapping")
            # shape: max_sentence_length x num_sentence
            self.inp_q = tf.placeholder(tf.int32, [None, None], name="inp_q")
            # shape: max_sentence_length x num_sentence
            self.target = tf.placeholder(tf.int32, [None, None], name="target")
            # # shape: bool scaler
            # self.is_training = tf.placeholder(tf.bool, name="is_training")

            xlnet_model = xlnet.XLNetModel(
                xlnet_config=xlnet_config,
                run_config=run_config,
                input_ids=self.input_ids,
                seg_ids=self.seg_ids,
                input_mask=self.input_mask,
                mems=None,
                perm_mask=self.perm_mask,
                target_mapping=self.target_mapping,
                inp_q=self.inp_q)

            output = xlnet_model.get_sequence_output()
            lookup_table = xlnet_model.get_embedding_table()
            initializer = xlnet_model.get_initializer()

            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                # LM loss
                lm_loss, logits = modeling.lm_loss(
                    hidden=output,
                    target=self.target,
                    n_token=xlnet_config.n_token,
                    d_model=xlnet_config.d_model,
                    initializer=initializer,
                    lookup_table=lookup_table,
                    tie_weight=True,
                    bi_data=run_config.bi_data,
                    use_tpu=run_config.use_tpu)


            self.masked_lm_example_loss = run_lm_predict.get_masked_lm_output(
                self.bert_config, model.get_sequence_output(), model.get_embedding_table(),
                self.masked_lm_positions, self.masked_lm_ids)

            # load the pretrained bert model parameters
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                tf.trainable_variables(), bert_ckpt)
            tf.train.init_from_checkpoint(bert_ckpt, assignment_map)

            #### load pretrained models
            scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

            self.session.run(tf.global_variables_initializer())

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

if __name__ == '__main__':
    xlnet_lm = XLNetLM(r'C:\Works\PretrainedModel\chinese_xlnet_base_L-12_H-768_A-12')
