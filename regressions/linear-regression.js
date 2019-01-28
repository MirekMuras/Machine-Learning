const tf = require('@tensorflow/tfjs');
const _ = require('loadash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = features;
    this.labels = labels;
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    this.m = 100;
    this.b = 500;
    
  }



  gradientDecsent() {
    const currentGuessesForMPG = this.features.map((row) => { return this.m * row[0] + this.b; });
    const hx = currentGuessesForMPG.map((guess, i) => guess - this.labels[i][0]);
    const B_Slope = ( (_sum(hx) * 2) / this.features.length; 
     
    const m_Slope = (_sum(currentGuessesForMPG.map((guess, i) =>{
      return -1 * this.features[i][0] + (this.labels[i][0 - guess])
    }) * 2) / this.features.length) ; 

    this.m = this.m - m_Slope * this.options.learningRate;
    this.b = this.b - B_Slope * this.options.learningRate;
  }


  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDecsent();
    }
  }
}

module.exports = LinearRegression;