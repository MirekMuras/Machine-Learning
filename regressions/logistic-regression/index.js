require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');

const{features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    dataColumns:['horsepower','displacement','weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,                                  // shuffle all values in the data set
    splitTest: 10,                                  // 
    converters: {                                   // convert string value to an integer
        passedemissions: (value) => {
            return value === 'TRUE' ? 1 : 0       // from passedmission column , if the value is 'TRUE' return 1, else 0
            }
        }
    }
);

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.3,
    iterations: 100,
    batchSize: 50,
});

regression.train();
//regression.predict([[175, 400, 2.57], [90,140,1.13] , [88,150,1.16] ]).print();
console.log(regression.test(testFeatures, testLabels));

plot({
    x: regression.cost_History_Array.reverse()
});