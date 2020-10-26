function convertToTensor(data,numShops) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
      //Shuffle the data
      tf.util.shuffle(data);


      //Convert data to Tensor
      const inputs = data.map(d => d.slice(0,numShops));

      const labels = data.map( d => {
        temparray = new Array(numShops).fill(0);
        temparray[d[numShops]]=1;
        return temparray;
      });

      const inputTensor = tf.tensor2d(inputs, [data.length,numShops]);
      const labelTensor = tf.tensor2d(labels, [labels.length, numShops]);

      //Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        //Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }
