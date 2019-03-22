require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');

const{features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    dataColumns:[
        'horsepower',
        'displacment',
        'weight'
    ],
    labelColumns: ['passedmissions'],
    shuffle: true,
    splitTest: 50


});