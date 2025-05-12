import { MnistData } from "./data.js";

async function showExamples(data) {
  const surface = tfvis
    .visor()
    .surface({ name: "Input Data Examples", tab: "Input Data" });

  // Se puede seguir visualizando 1000 ejemplos o ajustar según convenga.
  const examples = data.nextTestBatch(1000);
  const numExamples = examples.xs.shape[0];

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);
    imageTensor.dispose();
  }
}

async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);
  const model = getModel();
  tfvis.show.modelSummary({ name: "Model Architecture", tab: "Model" }, model);
  await train(model, data);
  await model.save("downloads://modelo_entrenado");
}

document.addEventListener("DOMContentLoaded", run);

function getModel() {
  const model = tf.sequential();
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // Primera capa convolucional
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 16, // se incrementa el número de filtros
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    }),
  );

  // Se agrega batch normalization
  model.add(tf.layers.batchNormalization());

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Segunda capa convolucional
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 32, // se incrementa el número de filtros aún más
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    }),
  );

  // Agregar batch normalization y pooling
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Aplanar para la capa densa
  model.add(tf.layers.flatten());

  // Capa densa intermedia con Dropout
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.5 }));

  // Capa de salida
  const NUM_OUTPUT_CLASSES = 10;
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    }),
  );

  // Configurar el modelo con un optimizador y la función de pérdida
  const optimizer = tf.train.adam(0.001);
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function train(model, data) {
  // Monitorización de métricas con tfvis
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = {
    name: "Model Training",
    tab: "Model",
    styles: { height: "1000px" },
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 6000; // Incrementar los datos de entrenamiento
  const TEST_DATA_SIZE = 1000;

  // Se puede aplicar data augmentation aquí si se implementa
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 30, // aumentar la cantidad de epochs para un entrenamiento más fuerte
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1,
  ]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = { name: "Accuracy", tab: "Evaluation" };
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = { name: "Confusion Matrix", tab: "Evaluation" };
  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: classNames,
  });
  labels.dispose();
}
