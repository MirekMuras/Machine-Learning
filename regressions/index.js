require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv.js');
const LinearRegression = require('./linear-regression.js');

let {features, labels, testFeatures, testLabels} = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacment'],
  labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.001,
  iterations: 100
});

regression.features.print();
regression.train();

const r2  = regression.test(testFeatures, testLabels);

console.log("The r2 is: " , r2);

// check the values
// console.log('Updated M is :', regression.W.get(1, 0), 'Updated B is :',regression.W.bet(0, 0));


