function createModel(numShops) {
    //Create a sequential model
    const model = tf.sequential();
  
    //Add a single hidden layer
    model.add(tf.layers.dense({inputShape: [numShops], units: 20, useBias: true}));
  
    //Add second layer
    model.add(tf.layers.dense({units: 18, activation: 'sigmoid', useBias: true}));
  
    //Add an output layer
    model.add(tf.layers.dense({units: numShops, activation: 'sigmoid'}));
  
    return model;
  }