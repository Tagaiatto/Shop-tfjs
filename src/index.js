

async function run() {

    const numShops = 5;
    const numElements = 500;
    const range = 10;

    const data = generateData(range,numShops,numElements);

    //Print subtitle in the page
    const subtitleDiv = document.getElementById('subtitle');
    subtitleDiv.innerHTML +=`Data should be array which contains ${numShops} randomly generated ratings and one choice`;

    //Print each element in the generated data
    const examplesDiv = document.getElementById('dataExamples');
    for (let i = 0; i< numElements;i++){
        examplesDiv.innerHTML += `${i+1} element in data: ${data[i]}`+'<br>';
    }

    //Convert the data to a tensor form we can use for training.
    const tensorData = convertToTensor(data,numShops);
    const {inputs, labels} = tensorData;

    //Create model to train
    const model = createModel(numShops);
    //Print the status of models
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    //Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    //Test the model
    testModel(model, data, tensorData,numShops, numElements);

  }

  run();
