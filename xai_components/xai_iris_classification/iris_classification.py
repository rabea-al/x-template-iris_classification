from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component

#------------------------------------------------------------------------------
#                    Xircuits Component : LoadDatasetURL
#------------------------------------------------------------------------------
@xai_component
class LoadDatasetURL(Component):
    url: InArg[str]
    column_names: InArg[list]

    def execute(self, ctx) -> None:
        import pandas as pd
        url = self.url.value
        column_names = self.column_names.value
        dataset = pd.read_csv(url, names=column_names)
        # dataset["class"] = dataset["class"].str.replace("Iris-","") # Iris-setosa --> setosa
        ctx.update({'dataset': dataset})

#------------------------------------------------------------------------------
#                    Xircuits Component : VisualizeData
#------------------------------------------------------------------------------
@xai_component
class VisualizeData(Component):
    target_column: InArg[str]

    def execute(self, ctx) -> None:
        import seaborn as sns
        dataset = ctx['dataset']
        print(dataset.head())
        target_column = self.target_column.value
        sns.pairplot(dataset, hue=target_column)

#------------------------------------------------------------------------------
#                    Xircuits Component : SplitDataAndLabel
#------------------------------------------------------------------------------
@xai_component
class SplitDataAndLabel(Component):
    label_column_index: InArg[int]

    def execute(self, ctx) -> None:
        dataset = ctx['dataset']
        label_column_index = self.label_column_index.value
        X = dataset.values[:, 0:label_column_index]
        Y = dataset.values[:, label_column_index]

        ctx.update({'X': X, 'Y':Y})

#------------------------------------------------------------------------------
#                    Xircuits Component : TrainTestSplit
#------------------------------------------------------------------------------
@xai_component
class TrainTestSplit(Component):
    test_percentage: InArg[float]

    def execute(self, ctx) -> None:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        import numpy as np

        X = ctx['X']
        Y = ctx['Y']
        test_percentage = self.test_percentage.value

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_percentage)

        scaler = StandardScaler()
        scaler.fit(X_train)
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)

        # One hot encoding
        enc = OneHotEncoder()
        y_train = enc.fit_transform(y_train[:, np.newaxis]).toarray()
        y_test = enc.fit_transform(y_test[:, np.newaxis]).toarray()

        print(f'Training data shape: {x_train.shape}')
        print(f'Training label shape: {y_train.shape}')
        print(f'Testing data shape: {x_test.shape}')
        print(f'Testing label shape: {y_test.shape}')

        ctx.update(
            {'x_train': x_train,
             'x_test': x_test,
             'y_train': y_train,
             'y_test': y_test})

#------------------------------------------------------------------------------
#                    Xircuits Component : Create1DModel
#------------------------------------------------------------------------------
@xai_component
class Create1DModel(Component):
    loss: InArg[str]
    optimizer: InArg[str]

    model: OutArg[any]

    def execute(self, ctx) -> None:
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        from tensorflow import keras
        x_train = ctx['x_train']
        y_train = ctx['y_train']

        x_shape = x_train.shape
        y_shape = y_train.shape
        model = keras.Sequential([
                    keras.layers.Dense(512, activation='relu', input_shape=(x_shape[1],)),
                    keras.layers.Dropout(rate=0.5),
                    keras.layers.Dense(y_shape[1], activation='softmax')
        ])

        model.compile(
            loss=self.loss.value,
            optimizer=self.optimizer.value,
            metrics=['accuracy']
        )
        model.summary()
        self.model.value = model

#------------------------------------------------------------------------------
#                    Xircuits Component : TrainNNModel
#------------------------------------------------------------------------------
@xai_component
class TrainNNModel(Component):
    model: InArg[any]
    training_epochs: InArg[int]

    training_metrics: OutArg[dict]

    def execute(self, ctx) -> None:

        model = self.model.value
        x_train = ctx['x_train']
        y_train = ctx['y_train']

        train = model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=self.training_epochs.value,
            validation_split=0.1
        )

        ctx.update({'trained_model': model})
        self.training_metrics.value = train.history

#------------------------------------------------------------------------------
#                    Xircuits Component : PlotTrainingMetrics
#------------------------------------------------------------------------------
@xai_component
class PlotTrainingMetrics(Component):
    training_metrics: InArg[dict]

    def execute(self, ctx) -> None:
        import matplotlib.pyplot as plt
        history = self.training_metrics.value

        acc = history['accuracy']
        val_acc = history['val_accuracy']

        loss = history['loss']
        val_loss = history['val_loss']

        epochs = range(1, len(acc)+1)

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.ylim([0, 1])
        plt.xticks(epochs)
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.xticks(epochs)
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epochs')
        plt.show()

#------------------------------------------------------------------------------
#                    Xircuits Component : EvaluateNNModel
#------------------------------------------------------------------------------
@xai_component
class EvaluateNNModel(Component):


    def execute(self, ctx) -> None:
        import numpy as np
        from sklearn.metrics import classification_report
        model = ctx['trained_model']
        x_test = ctx['x_test']
        y_test = ctx['y_test']

        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Testing Loss: {:.2f}".format(loss))
        print("Testing Accuracy: {:.2f}".format(accuracy))

        y_pred = model.predict(x_test)
        y_pred = [np.argmax(i) for i in y_pred]
        y_test = [np.argmax(i) for i in y_test]
        print(classification_report(y_test, y_pred, digits=10))

#------------------------------------------------------------------------------
#                    Xircuits Component : SaveNNModel
#------------------------------------------------------------------------------
@xai_component
class SaveNNModel(Component):
    save_model_path: InArg[str]
    keras_format: InArg[bool]

    def execute(self, ctx):
        import os
        model = ctx['trained_model']
        model_name = self.save_model_path.value

        dirname = os.path.dirname(model_name)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        if self.keras_format.value:
            model_name = model_name + '.h5'
        else:
            model_name = model_name + '.keras'

        model.save(model_name)
        print(f"Saving model at: {model_name}")
        ctx.update({'saved_model_path': model_name})

#------------------------------------------------------------------------------
#                    Xircuits Component : ConvertTFModelToOnnx
#------------------------------------------------------------------------------
@xai_component
class ConvertTFModelToOnnx(Component):
    output_onnx_path: InArg[str]

    def execute(self, ctx):
        import os
        import tensorflow as tf
        import tf2onnx
        import onnx

        saved_model_path = ctx['saved_model_path']
        onnx_path = self.output_onnx_path.value
        dirname = os.path.dirname(onnx_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Load the Keras model
        model = tf.keras.models.load_model(saved_model_path)

        # Convert the Keras model to ONNX format
        input_signature = [tf.TensorSpec([None, *model.input_shape[1:]], tf.float32, name='input')]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=11)

        # Save the ONNX model
        onnx.save(onnx_model, onnx_path + '.onnx')
        print(f'Converted {saved_model_path} to {onnx_path}.onnx')
