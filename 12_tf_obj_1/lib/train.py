from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v1 as tf

from object_detection import model_lib


import os
import json



def main(_):
    with open('system_dict.json') as json_file:
        args = json.load(json_file) 

    config = tf.estimator.RunConfig(model_dir=args["model_dir"])
    
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
                                      run_config=config,
                                      pipeline_config_path=args["pipeline_config_path"],
                                      train_steps=args["num_train_steps"],
                                      sample_1_of_n_eval_examples=args["sample_1_of_n_eval_examples"],
                                      sample_1_of_n_eval_on_train_examples=(
                                          args["sample_1_of_n_eval_on_train_examples"]))
    
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    
    if args["checkpoint_dir"]:
        if args["eval_training_data"]:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if args["run_once"]:
            estimator.evaluate(input_fn,
                                steps=None,
                                checkpoint_path=tf.train.latest_checkpoint(
                                    args["checkpoint_dir"]))
        else:
            model_lib.continuous_eval(estimator, args["checkpoint_dir"], input_fn,
                                    train_steps, name, args["max_eval_retries"])
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
                                        train_input_fn,
                                        eval_input_fns,
                                        eval_on_train_input_fn,
                                        predict_input_fn,
                                        train_steps,
                                        eval_on_train_data=False)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

tf.app.run(main=main)
