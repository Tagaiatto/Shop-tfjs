async function testModel(model, inputData, normalizationData, numShops, numElements) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  
    //Generate predictions for a uniform range of numbers between 0 and 1;
    //We un-normalize the data by doing the inverse of the min-max scaling
  
    const totalNum = numShops*numElements;
  
    const xsi = tf.linspace(0,1,totalNum);
    console.log("Before shuffle xsi: ",xsi);
    const xsarray = xsi.dataSync();
    tf.util.shuffle(xsarray);
    console.log("After shuffle, xsarray: ",xsarray);
    const xst = tf.tensor1d(xsarray);
    console.log("From xsarray to xst tensor: ",xst.print());
  
    xaxis = tf.linspace(1,numElements,numElements).dataSync();
    console.log("xaxis x:",Array.from(xaxis));
  
  
    const [xs, preds] = tf.tidy(() => {
  
          const preds = model.predict(xst.reshape([numElements, numShops]));
          console.log("Before change to unNormpreds: ",preds.print());
  
      const unNormXs = xst
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
  
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
  
        console.log("After change to unNormpreds: ",unNormPreds);
  
      //Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.arraySync()];
    });
  
    console.log("xs: ",xs);
    console.log("preds: ",preds);
  
    let maxIndexp = 0;
  
    let originalp = [];
  
    for (let k = 0; k < numElements; k++) {
  
      for(let i = 0; i<numShops;i++){
        const seed = preds[k][i];
        if(seed>preds[k][maxIndexp]){
            maxIndexp = i;
        };
  
        if(i == numShops-1){
          originalp.push(maxIndexp);
          maxIndexp = 0;
        };
      }
  
    };
  
    const predictedPoints = Array.from(xaxis).map((val, i) => {
      return {x: val, y: originalp[i]}
    });
  
    let maxIndex = 0;
  
    let original = [];
  
    for (let i = 0; i < totalNum; i++) {
  
        const seed = xsarray[i];
        if(seed>xsarray[maxIndex]){
            maxIndex = i;
        };
  
        if(i%numShops == numShops-1){
          original.push(maxIndex%numShops);
          maxIndex = i+1;
        };
    };
  
    console.log("orginal y:",original);
  
  
    const originalPoints = Array.from(xaxis).map((d,i) => {
      return{x: d, y: original[i]}
    });
  
    console.log("predicted: ",predictedPoints);
  
    console.log("original: ",originalPoints);
  
    let ratio = 0;
    for(let i=0;i<numElements;i++){
      if(predictedPoints[i]['y']==originalPoints[i]['y']){
        ratio++;
      };
    }
    ratio = ratio/numElements*100;
  
    console.log("Hit ratio:",ratio);
  
    const hitratioDiv = document.getElementById('hitratio');
  
    hitratioDiv.innerHTML +=`HIT Ratio:${ratio}%`+`<br>`;
  
  
    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data', tab: 'Evaluation'},
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
      {
        xLabel: `rating from ${numShops} shops`,
        yLabel: 'choice',
        height: 300
      }
    );
  
    const confusionMatrix = await tfvis.metrics.confusionMatrix(tf.tensor1d(original),tf.tensor1d(originalp));
  
    const container = {name: 'confusionMatrix', tab: 'Evaluation'};
  
    const choices = tf.linspace(1,numShops,numShops).arraySync();
  
    tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: Array.from(choices)});
  }