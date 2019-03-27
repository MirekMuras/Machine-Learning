require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

// LOAD Training Set where 'x' = number of images == .training(0,x)
const mnistData = mnist.training(0,5000);

// format mnistData to one long array for an each observation
const features = mnistData.images.values.map(image => _.flatMap(image));
const encodedLabels = mnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
});

//console.log(mnistData.images.values);
//console.log('\nNumbers : ' + mnistData.labels.values + '\n Labels : ');
//console.log(encodedLabels);
const regression = new LogisticRegression(features, encodedLabels, {
    learningRate: 1,
    iteration: 20,
    batchSize: 100
});

regression.train();

const testMnistData = mnist.testing(0,100);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;

});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('Accuracy is : '+accuracy *100 + ' %. ')

