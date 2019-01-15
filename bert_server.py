import json
from flask import Flask, request
import tensorflow as tf
import run_squad

# curl -X POST http://localhost:5555/bert -H 'Content-Type: application/json' -d'{"question":"Who began life as a network protocol?", "passage":"Kermit is a frog, but he began life as a network protocol."}' 

FLAGS = tf.flags.FLAGS
BERT_BASE_DIR = "uncased_L-12_H-768_A-12"
SQUAD_DIR = "squad"

app = Flask(__name__)

def setup_squad1():
    FLAGS.predict_file = SQUAD_DIR + "/dev-v1.1.json"
    FLAGS.train_file = SQUAD_DIR + "/train-v1.1.json"
    FLAGS.output_dir = "squad1_base"
    FLAGS.version_2_with_negative = False
    print("SQuAD 1.1 Active.")

def setup_squad2():
    FLAGS.predict_file = SQUAD_DIR + "/dev-v2.0.json"
    FLAGS.train_file = SQUAD_DIR + "/train-v2.0.json"
    FLAGS.output_dir = "squad2_base"
    FLAGS.version_2_with_negative = True
    print("SQuAD 2.0 Active.")

@app.before_first_request
def setup():
    setup_squad1()
    FLAGS.init_checkpoint = BERT_BASE_DIR + "/bert_model.ckpt"
    FLAGS.bert_config_file = BERT_BASE_DIR + "/bert_config.json"
    FLAGS.vocab_file = BERT_BASE_DIR + "/vocab.txt"
    FLAGS.train_batch_size=12
    FLAGS.learning_rate=3e-5
    FLAGS.num_train_epochs=2.0
    FLAGS.max_seq_length=256
    FLAGS.doc_stride=128
    FLAGS.do_predict = False
    tf.flags.FLAGS.do_train = False
    run_squad.main(None)
    print("BERT Initialized.")


@app.route("/predict", methods=["POST"])
def bert():
    data = json.loads(request.data)
    result = run_squad.answer(data)
    return json.dumps(result)


if __name__ =='__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)

