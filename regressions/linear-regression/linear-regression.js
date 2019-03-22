const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

/**
 * Labels = Tensor of label data
 * Features = Tensor of feature data
 * n = Number of observations
 * W (Weights) = M and B in a tensor
 * ( ' ) = transpose 
 * 
 * Slope of MSE with respect to M and B function
 * dMSE / M and B = Features' * ( (Features*W) - Labels) / n
 */

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);          //[342,1] 
    this.MSE_History_Array = [];              // Mean Square Error
      
    this.options = Object.assign(
       { learningRate: 0.1, iterations: 1000 },
       options
      );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

 
  
  // vectorize solution
  //@dev: each iteration will update 'm' and 'b'
  gradientDecsent(features,labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);

    // calculate slope/grsdience of MSE with respect to M and B
    // Labels = Tensor of label data
    // Features = Tensor of feature data
    // n = number of observations
    // W (weights) = M & B in a tansor
    const slopes = features
     .transpose()
     .matMul(differences)
     .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));    // W = W - (slopes * learningRate)
  }

  //@dev: run GD until we get good values for 'm' and 'b'
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

      this.MSE_History_Array;
      this.update_Learning_Rate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }

  //@dev: evaluate the accuracy of calculated 'm' and 'b'
  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);  // this.features.shape [50, 1]
    testLabels = tf.tensor(testLabels);                 // this.labels.shape [50, 1]

    const predictions = testFeatures.matMul(this.weights);    //error: multiplicatiuon NOT ALLOWED = [50,1] * [2, 1] 

    //@dev: result in (-) negative number = result of if res > tot 
    const res = testLabels
    .sub(predictions)
    .pow(2)
    .sum() 
    .get();
    const tot = testLabels
    .sub(testLabels.mean())
    .pow(2)
    .sum()
    .get();

    return 1 - res / tot;
  }

  processFeatures(features) {
    features = tf.tensor(features);                                   // convert to the tensor
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
  recordMSE () {
    const mse = this.features   
    .mul(this.weights)
    .sub(this.labels)
    .pow(2)
    .sum()
    .div(this.features.shape[0])
    .get();

    this.MSE_History_Array.unshift(mse);
  }

  /** Custom Learning Rate Optimizier
   * - Calculate the exact iteration value of MSE , 
   *   store and compare against tth eold MSE   */
  update_Learning_Rate() {
    if (this.MSE_History_Array.length < 2) return;        // if empty or just one , return
    // if MSE gretaer then the old MSE , divide by 2
    if (this.MSE_History_Array[0] > this.MSE_History_Array[1]) this.options.learningRate /= 2 ;   
    // if MSE smaller than , we are going in the right direction, multiply by 0.5
    else this.options.learningRate *= 0.5;
  }

}
module.exports = LinearRegression;
