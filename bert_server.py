import json
from flask import Flask, request
import tensorflow as tf
import run_squad

FLAGS = tf.flags.FLAGS

app = Flask(__name__)

@app.before_first_request
def setup():
    BERT_BASE_DIR = "../../dev-bert/uncased_L-12_H-768_A-12"
    SQUAD_DIR = "../../dev-bert/"
    
    #    FLAGS.request_data = data
    FLAGS.predict_file = SQUAD_DIR + "/dev-v2.0.json"
    FLAGS.train_file = SQUAD_DIR + "/train-v2.0.json"
    FLAGS.init_checkpoint = BERT_BASE_DIR + "/bert_model.ckpt"
    FLAGS.bert_config_file = BERT_BASE_DIR + "/bert_config.json"
    FLAGS.vocab_file = BERT_BASE_DIR + "/vocab.txt"
    FLAGS.output_dir = "/tmp/squad_base/"
    FLAGS.train_batch_size=12
    FLAGS.learning_rate=3e-5
    FLAGS.num_train_epochs=2.0
    FLAGS.max_seq_length=256
    FLAGS.doc_stride=128
    FLAGS.version_2_with_negative = True
    FLAGS.do_predict = False
    tf.flags.FLAGS.do_train = False
    run_squad.main(None)
    print("BERT initialized.")


@app.route("/bert", methods=["POST"])
def bert():
    data = json.loads(request.data)
    result = run_squad.answer(data)
    return json.dumps(result)


if __name__ =='__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)

