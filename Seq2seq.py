import tensorflow as tf

class RNNModel:
    def __init__(self, batch_size, hidden_size, hidden_layers, vocab_size, learning_rate):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        # 定义占位符
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 定义词嵌入层
        embedding = tf.Variable(tf.random.uniform([vocab_size, hidden_size]), name='embedding')
        emb_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        emb_inputs = tf.nn.dropout(emb_inputs, rate=1 - self.keep_prob)

        # 搭建LSTM结构
        lstm_cell = tf.keras.layers.LSTMCell(hidden_size)
        dropout_cell = tf.keras.layers.Dropout(1 - self.keep_prob)(lstm_cell)
        cell = tf.keras.layers.StackedRNNCells([dropout_cell] * hidden_layers)
        self.initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        
        outputs, self.final_state = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)(emb_inputs, initial_state=self.initial_state)

        # 重塑输出并计算logits
        outputs = tf.reshape(outputs, [-1, hidden_size])
        w = tf.Variable(tf.random.normal([hidden_size, vocab_size]), name='outputs_weight')
        b = tf.Variable(tf.zeros([vocab_size]), name='outputs_bias')
        logits = tf.matmul(outputs, w) + b

        # 计算损失
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(tf.reshape(self.targets, [-1]),
                                                                   tf.reshape(logits, [-1, vocab_size]))
        self.cost = tf.reduce_mean(self.loss)

        # 优化算法
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                       global_step,
                                                                       decay_steps=BATCH_NUMS,
                                                                       decay_rate=0.99,
                                                                       staircase=True)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        trainable_vars = tf.trainable_variables()
        gradients = tf.gradients(self.cost, trainable_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
        self.opt = optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))

        # 预测输出
        self.predict = tf.argmax(logits, axis=1)


# 预定义模型参数
VOCAB_SIZE = len(vocab)
EPOCHS = 1000
BATCH_SIZE = 8
TIME_STEPS = 100
BATCH_NUMS = len(numdata) // (BATCH_SIZE * TIME_STEPS)
HIDDEN_SIZE = 512
HIDDEN_LAYERS = 6
MAX_GRAD_NORM = 1
LEARNING_RATE = 0.05

# 模型训练
model = RNNModel(BATCH_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE, LEARNING_RATE)
print(model)

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=5)

# 设置TensorBoard日志
summary_writer = tf.summary.create_file_writer('logs/tensorboard')

for k in range(EPOCHS):
    state = model.initial_state
    train_data = data_generator(numdata, BATCH_SIZE, TIME_STEPS)
    
    total_loss = 0.
    for i in range(BATCH_NUMS):
        xs, ys = next(train_data)
        with tf.GradientTape() as tape:
            feed = {model.inputs: xs, model.targets: ys, model.keep_prob: 0.8, model.initial_state: state}
            costs = model.cost(feed_dict=feed)
            total_loss += costs
        
        gradients = tape.gradient(costs, model.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
        model.optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables), global_step=tf.train.get_or_create_global_step())
        
        if (i + 1) % 50 == 0:
            print('Epoch:', k + 1, 'Iter:', i + 1, 'Cost:', total_loss / (i + 1))
            
    with summary_writer.as_default():
        tf.summary.scalar('Loss', total_loss / BATCH_NUMS, step=k)
    
    # 保存模型
    manager.save(checkpoint_number=k)

summary_writer.close()