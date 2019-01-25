require ('@tensorflow/tfjs-node');
const tf = require ('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

function knn (features, labels, predicitonPoint, k ) {
    // impliment STANDARDISATION = Value - Average / StandardVariant 
    const {mean, variance} = tf.moments(features, 0);

    const scaledPrediction = predicitonPoint.sub(mean).div(variance.pow(0.5));

    return (
        features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        .sort((a,b) => a.get(0) > b.get(0) ? 1 : -1)
        .slice(0,k)
        .reduce((acc, pair) => acc + pair.get(1), 0) / k 
    );   
}

// load data from .csv file into features and labels
let { features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', 
    {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', sqrt_lot],
    labelColumns: ['price']
    }
);

features = tf.tensor(features);
labels = tf.tensor(labels);
//console.log(features.shape);
//console.log(labels.shape);


testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[0][0] -result) / testLabels[0][0];

    // Print out the result of the property price proediciton with Error percentile
    console.log('My prediction for the property is :', result, testLabels[0][0]);
    console.log('Error :', err * 100);
    });

// using Standardization using tenserflow
/**
 *  tf.moments();                       // calling method moments() directly from tenserflow.js
 *  cosnt StandardDeviation = sqrt(variance);
 *  const Standardization = (Value - Average) / StandardDeviation;
 */

