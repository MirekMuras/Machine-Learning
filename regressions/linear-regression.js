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
    this.labels = tf.tensor(labels);
      
    this.options = Object.assign(
       { learningRate: 0.1, iterations: 1000 },
       options
      );

    this.W = tf.zeros([this.features.shape[1],1]);
  
  }

  // vectorize solution
gradientDecsent() {
    const currentGuesses = this.features.matMul(this.W);
    const differences = currentGuesses.sub(this.labels)

    const slopes = this.features
     .transpose()
     .matMul(differences)
     .div(this.features.shape[0]);

    this.W =  this.W.sub(slopes.mul(this.options.learningRate));    // W.shape [2 ,1] now
    

  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDecsent();
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);  // this.features.shape [50, 1]
    testLabels = tf.tensor(testLabels);                 // this.labels.shape [50, 1]

    const predictions = testFeatures.matMul(this.W);    //error: multiplicatiuon NOT ALLOWED = [50,1] * [2, 1] 

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
    features = tf.tensor(features);  

    if(this.mean && this.variance) {
      return features.sub(this.mean).div(this.variance.pow(0.5));
    }
    else {
      features = this.standardize(features);
    }
    
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standardize(features) {
    const {mean, variance} = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance;

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }


}

module.exports = LinearRegression;
