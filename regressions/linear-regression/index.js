require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');
      
let {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],  // variable 'x' ==> input x[i][j]
  labelColumns: ['mpg']                                   // veriable 'Actual' ==> output y[i]
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.3,
  iterations: 5,
  batchSize: 10
});

//regression.features.print();
regression.train();
const r2  = regression.test(testFeatures, testLabels);

//console.log("MSE History :" , regression.MSE_History_Array);

plot({  
  x: regression.MSE_History_Array.reverse(),
  xLabel: 'Iteration #',
  ylabel: "Mean Square Error"
});

 console.log("The r2 is: " , r2);

// check the values
 console.log(
   'Updated M is :', 
   regression.weights.get(1, 0), 
   'Updated B is :',
   regression.weights.get(0, 0)
   );

   regression.predict([
     [120,2,380]
   ]).print();