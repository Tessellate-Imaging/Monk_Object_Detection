import os
import json
from absl import flags
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

def main(_):
    with open('system_dict.json') as json_file:
        args = json.load(json_file) 
    
    tf.config.set_soft_device_placement(True)
    
    if args["checkpoint_dir"]:
        model_lib_v2.eval_continuously(
            pipeline_config_path=args["pipeline_config_path"],
            model_dir=args["model_dir"],
            train_steps=args["num_train_steps"],
            sample_1_of_n_eval_examples=args["sample_1_of_n_eval_examples"],
            sample_1_of_n_eval_on_train_examples=(
                args["sample_1_of_n_eval_on_train_examples"]),
            checkpoint_dir=args["checkpoint_dir"],
            wait_interval=300, timeout=args["eval_timeout"]);
        
    else:
        if args["use_tpu"]:
            # TPU is automatically inferred if tpu_name is None and
            # we are running under cloud ai-platform.
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                args["tpu_name"])
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
        elif args["num_workers"] > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()
        
        with strategy.scope():
            model_lib_v2.train_loop(
                        pipeline_config_path=args["pipeline_config_path"],
                        model_dir=args["model_dir"],
                        train_steps=args["num_train_steps"],
                        use_tpu=args["use_tpu"],
                        checkpoint_every_n=args["checkpoint_every_n"],
                        record_summaries=args["record_summaries"])
        
tf.compat.v1.app.run(main=main)         