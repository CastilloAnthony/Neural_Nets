### NON-FUNCTIONAL TUTORIAL METHODS###
# Originally a part of the dataHandler class.
from windowGenerator import WindowGenerator
from baseline import Baseline
from residualWrapper import ResidualWrapper

def window_test(self):
    w1 = WindowGenerator(input_width=24, label_width=1, shift=24,train_df=self.__normalizedData[0], val_df=self.__normalizedData[1], test_df=self.__normalizedData[2],
                    label_columns=self.__validWavelengths[-2:-1])
    print(w1)

    w2 = WindowGenerator(input_width=6, label_width=1, shift=1, train_df=self.__normalizedData[0], val_df=self.__normalizedData[1], test_df=self.__normalizedData[2],
                    label_columns=self.__validWavelengths[-2:-1])
    print(w2)

    example_window = tf.stack([np.array(self.__normalizedData[0][:w2.total_window_size]), 
                            np.array(self.__normalizedData[0][100:100+w2.total_window_size]), 
                            np.array(self.__normalizedData[0][200:200+w2.total_window_size])])
    example_inputs, example_labels = w2.split_window(example_window)
    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'Labels shape: {example_labels.shape}')
    w2.plot()
    # w2.example = example_inputs, example_labels
    w2.plot(plot_col=self.__validWavelengths[-3])

    # Each element is an (inputs, label) pair.
    print(w2.train.element_spec)

    for example_inputs, example_labels in w2.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

    single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, train_df=self.__normalizedData[0], val_df=self.__normalizedData[1], test_df=self.__normalizedData[2],
        label_columns=['AOD_380nm-Total'])
    print(single_step_window)

    for example_inputs, example_labels in single_step_window.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')
        
    baseline = Baseline(label_index=w2.getColumnIndicies()['AOD_380nm-Total'])

    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1, train_df=self.__normalizedData[0], val_df=self.__normalizedData[1], test_df=self.__normalizedData[2],
        label_columns=['AOD_380nm-Total'])

    print(wide_window)
    print('Input shape:', wide_window.example[0].shape)
    print('Output shape:', baseline(wide_window.example[0]).shape)
    wide_window.plot(baseline)
    plt.savefig('graphs/'+'model_baseline'+'.png')
    # plt.show()

    linear = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1)
        ])

    print('Input shape:', single_step_window.example[0].shape)
    print('Output shape:', linear(single_step_window.example[0]).shape)

    # compile_and_fit
    # print(type(linear), type(single_step_window))
    history = self.compile_and_fit(linear, single_step_window)

    val_performance['Linear'] = linear.evaluate(single_step_window.val, return_dict=True)
    performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0, return_dict=True)

    print('Input shape:', wide_window.example[0].shape)
    print('Output shape:', linear(wide_window.example[0]).shape)

    wide_window.plot(linear)
    plt.savefig('graphs/'+'model_linear'+'.png')
    # plt.show()
    plt.bar(x = range(len(self.__normalizedData[0].columns)),
            height=linear.layers[0].kernel[:,0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(self.__normalizedData[0].columns)))
    _ = axis.set_xticklabels(self.__normalizedData[0].columns, rotation=90)
    plt.savefig('graphs/'+'test1'+'.png')
    # plt.show()

    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    history = self.compile_and_fit(dense, single_step_window)

    val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
    performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)

    single_step_window.plot(dense)
    plt.savefig('graphs/'+'model_dense'+'.png')

    CONV_WIDTH = 3
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        train_df=self.__normalizedData[0], 
        val_df=self.__normalizedData[1], 
        test_df=self.__normalizedData[2],
        label_columns=['AOD_380nm-Total'])

    conv_window
    conv_window.plot()
    plt.suptitle("Given 3 hours of inputs, predict 1 hour into the future.")
    # plt.show()

    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])

    print('Input shape:', conv_window.example[0].shape)
    print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

    history = self.compile_and_fit(multi_step_dense, conv_window)

    IPython.display.clear_output()
    val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val, return_dict=True)
    performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0, return_dict=True)
    print('Multi step dense')
    # plt.figure(figsize=(12, 8))
    conv_window.plot(multi_step_dense)
    plt.savefig('graphs/'+'model_multi_step_dense'+'.png')
    # plt.show()
    
    print('Input shape:', wide_window.example[0].shape)
    try:
        print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
    except Exception as e:
        print(f'\n{type(e).__name__}:{e}')

    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(CONV_WIDTH,),
                            activation='relu'),
        # tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    print("Conv model on `conv_window`")
    print('Input shape:', conv_window.example[0].shape)
    print('Output shape:', conv_model(conv_window.example[0]).shape)

    history = self.compile_and_fit(conv_model, conv_window)

    IPython.display.clear_output()
    val_performance['Conv'] = conv_model.evaluate(conv_window.val, return_dict=True)
    performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0, return_dict=True)

    print("Wide window")
    print('Input shape:', wide_window.example[0].shape)
    print('Labels shape:', wide_window.example[1].shape)
    print('Output shape:', conv_model(wide_window.example[0]).shape)

    LABEL_WIDTH = 24
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=1,
        train_df=self.__normalizedData[0], 
        val_df=self.__normalizedData[1], 
        test_df=self.__normalizedData[2],
        label_columns=['AOD_380nm-Total'])

    print(wide_conv_window)

    print("Wide conv window")
    print('Input shape:', wide_conv_window.example[0].shape)
    print('Labels shape:', wide_conv_window.example[1].shape)
    print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

    wide_conv_window.plot(conv_model)
    plt.savefig('graphs/'+'model_conv_model'+'.png')
    # plt.show()

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    print('Input shape:', wide_window.example[0].shape)
    print('Output shape:', lstm_model(wide_window.example[0]).shape)

    history = self.compile_and_fit(lstm_model, wide_window)

    IPython.display.clear_output()
    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val, return_dict=True)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0, return_dict=True)

    wide_window.plot(lstm_model)
    plt.savefig('graphs/'+'model_lstm_model'+'.png')
    # plt.show()

    cm = lstm_model.metrics[1]
    print(cm.metrics)

    print(val_performance)

    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in val_performance.values()]
    test_mae = [v[metric_name] for v in performance.values()]

    plt.figure(figsize=(12, 8))
    plt.ylabel('mean_absolute_error [T (degC), normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
            rotation=45)
    _ = plt.legend()
    # plt.show()
    plt.savefig('graphs/Performance_Single_Step.png')

    for name, value in performance.items():
        print(f'{name:12s}: {value[metric_name]:0.4f}')

    single_step_window = WindowGenerator(
        # `WindowGenerator` returns all features as labels if you
        # don't set the `label_columns` argument.
        input_width=1, label_width=1, shift=1,
        train_df=self.__normalizedData[0], 
        val_df=self.__normalizedData[1], 
        test_df=self.__normalizedData[2],
        )

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=self.__normalizedData[0], 
        val_df=self.__normalizedData[1], 
        test_df=self.__normalizedData[2],
        )

    for example_inputs, example_labels in wide_window.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

    baseline = Baseline()
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(wide_window.val, return_dict=True)
    performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0, return_dict=True)

    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=self.__num_features)
    ])

    history = self.compile_and_fit(dense, single_step_window)

    IPython.display.clear_output()
    val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
    performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)

    single_step_window.plot(dense)
    plt.savefig('graphs/'+'model_dense_single_step_window'+'.png')

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=self.__normalizedData[0], 
        val_df=self.__normalizedData[1], 
        test_df=self.__normalizedData[2],
        )

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=self.__num_features)
    ])

    history = self.compile_and_fit(lstm_model, wide_window)

    IPython.display.clear_output()
    val_performance['LSTM'] = lstm_model.evaluate( wide_window.val, return_dict=True)
    performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0, return_dict=True)

    wide_window.plot(lstm_model)
    plt.savefig('graphs/'+'model_lstm_wide_window'+'.png')
    # print()
# end window

def multiWindowTest(self):
    single_step_window = WindowGenerator(
        # `WindowGenerator` returns all features as labels if you
        # don't set the `label_columns` argument.
        input_width=3,
        label_width=1,
        shift=1,
        train_df=self.__normalizedData[0], 
        val_df=self.__normalizedData[1], 
        test_df=self.__normalizedData[2],
        )

    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=1,
        train_df=self.__normalizedData[0], 
        val_df=self.__normalizedData[1], 
        test_df=self.__normalizedData[2],
        )

    for example_inputs, example_labels in wide_window.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')
    
    val_performance = {}
    performance = {}

    # Baseline

    # baseline = Baseline()
    # baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
    #                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

    # val_performance['Baseline'] = baseline.evaluate(wide_window.val, return_dict=True)
    # performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0, return_dict=True)

    # wide_window.plot(baseline)
    # plt.savefig('graphs/'+'multi_model_baseline_wide_window'+'.png')

    # Dense

    # dense = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=self.__num_features)
    # ])

    # history = self.compile_and_fit(dense, single_step_window)

    # IPython.display.clear_output()
    # val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
    # performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)

    # single_step_window.plot(dense)
    # plt.savefig('graphs/'+'multi_model_dense_single_step_window'+'.png')

    # wide_window = WindowGenerator(
    #     input_width=24, label_width=24, shift=1,
    #     train_df=self.__normalizedData[0], 
    #     val_df=self.__normalizedData[1], 
    #     test_df=self.__normalizedData[2],
    #     )

    # lstm_model = tf.keras.models.Sequential([
    #     # Shape [batch, time, features] => [batch, time, lstm_units]
    #     tf.keras.layers.LSTM(32, return_sequences=True),
    #     # Shape => [batch, time, features]
    #     tf.keras.layers.Dense(units=self.__num_features)
    # ])

    # history = self.compile_and_fit(lstm_model, wide_window)

    # IPython.display.clear_output()
    # val_performance['LSTM'] = lstm_model.evaluate( wide_window.val, return_dict=True)
    # performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0, return_dict=True)

    # wide_window.plot(lstm_model)
    # plt.savefig('graphs/'+'multi_model_lstm_wide_window'+'.png')

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=self.__normalizedData[0], 
        val_df=self.__normalizedData[1], 
        test_df=self.__normalizedData[2],
        # label_columns=['AOD_1640nm-Total'],
        )

    residual_lstm = ResidualWrapper(
        tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(
            self.__num_features,
            # The predicted deltas should start small.
            # Therefore, initialize the output layer with zeros.
            kernel_initializer=tf.initializers.zeros())
    ]))

    history = self.compile_and_fit(residual_lstm, wide_window)

    IPython.display.clear_output()
    val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val, return_dict=True)
    performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0, return_dict=True)

    wide_window.plot(residual_lstm, plot_col='AOD_1640nm-Total')
    plt.savefig('graphs/'+'multi_model_residual_lstm_wide_window'+'.png')

    x = np.arange(len(performance))
    width = 0.3

    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in val_performance.values()]
    test_mae = [v[metric_name] for v in performance.values()]

    plt.figure(figsize=(12, 8))
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
            rotation=45)
    plt.ylabel('MAE (average over all outputs)')
    _ = plt.legend()
    plt.savefig('graphs/Performance_Multi_Step.png')
    # plt.show()

    for name, value in performance.items():
        print(f'{name:15s}: {value[metric_name]:0.4f}')
# end _multiWindowTest

def compile_and_fit(self, model, window, patience=2, MAX_EPOCHS=20):
    # print(type(model), type(window), type(patience))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min') # 'val_accuracy',
    
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()]) # 'accuracy'])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history
# end compile_and_fit