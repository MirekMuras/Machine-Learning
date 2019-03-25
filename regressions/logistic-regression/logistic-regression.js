const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);          //[342,1] 
    this.cost_History_Array = [];              // Mean Square Error
      
    this.options = Object.assign(
       { learningRate: 0.1, iterations: 1000, decisionBoundry: 0.5 },
       options
      );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  } 
  
  // vectorize solution
  //@dev: each iteration will update 'm' and 'b'
  gradientDecsent(features,labels) {
    const currentGuesses = features.matMul(this.weights).sigmoid();
    const differences = currentGuesses.sub(labels);

    const slopes = features
     .transpose()
     .matMul(differences)
     .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));    // W = W - (slopes * learningRate)
  }

  //@dev: 
  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
      );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const jIndex = j*this.options.batchSize;
        const {batchSize} = this.options;

        const features_Slice = this.features.slice(
          [jIndex,0],
          [batchSize, -1] 
        );

          const labels_Slice = this.labels.slice(
          [jIndex,0],
          [batchSize, -1] 
        );

        this.gradientDecsent(features_Slice, labels_Slice);        
      }

      this.recordCost
      
      
      ;
      this.update_Learning_Rate();
    }
  }

  predict(observations) {

    return this.processFeatures(observations)
    .matMul(this.weights)
    .sigmoid()
    .greater(this.options.decisionBoundry)
    .cast('float32');

  }

  //@dev: evaluate the accuracy of calculated 'm' and 'b'
  test(testFeatures, testLabels) {
   const prediction = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);

    const incorrect =  prediction
    .sub(testLabels)
    .abs()
    .sum()
    .get();

    return (prediction.shape[0] - incorrect) / prediction.shape[0];
  }

  processFeatures(features) {

    features = tf.tensor(features);                                    
    features = tf.ones([features.shape[0], 1]).concat(features, 1);   // add column of 1's

    if(this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    }
    else {
      features = this.standardize(features);
    }

    return features;
  }

  standardize(features) {
    const {mean, variance} = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance;

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  // build vectorized MSE method 
  recordCost () {
    const m = this.features.shape[0];
    const hx = this.features.matMul(this.weights).sigmoid();
    const y = this.labels;


    //@dev: p = (y' * log(hx))
    const J = y
    .transpose()
    .matMul(hx.log());

    //@dev: (1-y)' * log(1-hx)
    const p = y
    .mul(-1)
    .add(1)
    .transpose()
    .matMul(hx
      .mul(-1)
      .add(1)
      .log()
    );

   //@dev: -(1/m) * J + p
    const cost = J.add(p).div(m).mul(-1).get(0,0);

    this.cost_History_Array.unshift(cost);


  }

  /** Custom Learning Rate Optimizier
   * - Calculate the exact iteration value of MSE , 
   *   store and compare against tth eold MSE   */
  update_Learning_Rate() {
    if (this.cost_History_Array.length < 2) return;        // if empty or just one , return
    // if MSE gretaer then the old MSE , divide by 2
    if (this.cost_History_Array[0] > this.cost_History_Array[1]) this.options.learningRate /= 2 ;   
    // if MSE smaller than , we are going in the right direction, multiply by 0.5
    else this.options.learningRate *= 0.5;
  }

}
module.exports = LogisticRegression;
