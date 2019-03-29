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

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  } 
  
  // vectorize solution
  //@dev: each iteration will update 'm' and 'b'
  gradientDecsent(features,labels) {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    const slopes = features
     .transpose()
     .matMul(differences)
     .div(features.shape[0]);

    return  this.weights.sub(slopes.mul(this.options.learningRate));    // W = W - (slopes * learningRate)
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
        
        this.weights = tf.tidy(() => {
          const features_Slice = this.features.slice(
            [jIndex,0],
            [batchSize, -1] 
            );
  
            const labels_Slice = this.labels.slice(
            [jIndex,0],
            [batchSize, -1] 
            );

            return this.gradientDecsent(features_Slice, labels_Slice);
           });             
         }
      
      this.recordCost();
      this.update_Learning_Rate();
    }
  }

  predict(observations) {

    return this.processFeatures(observations)
    .matMul(this.weights)
    .softmax()
    .argMax(1)

  }

  //@dev: evaluate the accuracy of calculated 'm' and 'b'
  test(testFeatures, testLabels) {
   const prediction = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect =  prediction
    .notEqual(testLabels)
    .sum()
    .get();

    return (prediction.shape[0] - incorrect) / prediction.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);                                    
    //features = tf.ones([features.shape[0], 1]).concat(features, 1);   // add column of 1's
    
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

    const filler = variance.cast('bool').logicalNot().cast('float32');
    
    this.mean = mean;
    this.variance = variance.add(filler);

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  // build vectorized MSE method 
  recordCost () {
    const m = this.features.shape[0];
    const y = this.labels;

    const cost = tf.tidy(() => {
      const hx = this.features.matMul(this.weights).sigmoid();

      //@dev: p = (y' * log(hx))
      const J = y.transpose().matMul(hx.add(1e-7).log());
  
      //@dev: (1-y)' * log(1-hx)
      const p = y
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(hx
        .mul(-1)
        .add(1)
        .add(1e-7)          // Add a constant to avoid log(0) , log(0) == infinity , 1e-7 == 1 x 10 ^ -7 = 0.00000001
        .log()
      );  
     //@dev: -(1/m) * J + p
      return J.add(p).div(m).mul(-1).get(0,0);
    });   

    this.cost_History_Array.unshift(cost);
  }

  /** Custom Learning Rate Optimizier
   * - Calculate the exact iteration value of MSE , 
   *   store and compare against tth eold MSE   */
  update_Learning_Rate() {
    if (this.cost_History_Array.length < 2) {
      return;
    }        // if empty or just one , return
    // if MSE gretaer then the old MSE , divide by 2
    if (this.cost_History_Array[0] > this.cost_History_Array[1]){ 
      this.options.learningRate /= 2 ;   
    // if MSE smaller than , we are going in the right direction, multiply by 0.5
    }else {
      this.options.learningRate *= 1.05;
    }
  }

}
module.exports = LogisticRegression;
