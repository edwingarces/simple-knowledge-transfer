const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

let net;
let webcam;
async function app() {
  net= await mobilenet.load();
	webcam = await tf.data.webcam(webcamElement);

  while (true) {
    const img = await webcam.capture();
    const activation = net.infer(img, 'conv_preds');
    let result2;
    try {
      result2 = await classifier.predictClass(activation);
    } catch (error) {
      result2 = {};
    }
    const classes = ["Untrained", "Id", "Not Id"]
    try {
      document.getElementById("result").innerText = `
        Prediction: ${classes[result2.label]}\n
        Probability: ${result2.confidences[result2.label]}
      `;
    } catch (error) {
      document.getElementById("result").innerText = `
      Prediction: Untrained
      Probability: 0
    `;
    }
    // Dispose the tensor to release the memory.
    img.dispose();
    // Give some breathing room by waiting for the next animation frame to fire.
    await tf.nextFrame();
  }
}

//add example
async function addExample (classId) {
  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, classId);
  img.dispose()
}

const saveClassifier = async () => {
    let strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    const storageKey = 'classifier';
    localStorage.setItem(storageKey, strClassifier);
};


const loadClassifier = async ()=>{
    const storageKey ='classifier';
    let datasetJson = localStorage.getItem(storageKey);
    classifier.setClassifierDataset(Object.fromEntries(JSON.parse(datasetJson).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
};

app();
