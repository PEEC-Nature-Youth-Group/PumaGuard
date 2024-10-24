Classify
========

.. raw:: html

    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-wasm/dist/tfjs-wasm.js"></script>
    <script>
        tf.ready().then(() => {
        tf.loadGraphModel('path/to/your/model/model.json').then(model => {
            // Use the model for inference
            const inputData = tf.tensor([/* your input data */]);
            const output = model.predict(inputData);
            console.log(output);
        });
    });
    </script>
