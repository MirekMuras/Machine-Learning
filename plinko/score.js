

const outputs = [];


function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  // code here to analyze stuff
  const testSetSize = 50;
  const [testSet, trainingSet] = splitDataset(minMAX(outputs,3), testSetSize);

 
  _.range(1,15).forEach( k => {
     const accuracy =  _.chain(testSet)
    .filter(
      testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3]
      )
    .size()
    .divide(testSetSize)
    .value();

    console.log('For each k of ',k,' accuracy using KNN algorithm is ', accuracy);
  });

}

/**********************************************************************************/

// IF Prediction was bad !
// 1) Adjust the parameters of the analysis
// 2) Add more features to explain the analysis
// 3) Change the prediction point
// 4) Accept that maybe there isn't a good correlation


/***********************************************************************************/

function knn (data, point, k) {
     return  _.chain(data)
      .map(row => {
        return [
          distance(_.initial(row), point), 
          _.last(row)
        ];          
      })
      .sortBy(row => row[0])
      .slice(0, k)
      .countBy(row => row[1])
      .toPairs()
      .sortBy(row => row[1])
      .last()
      .first()
      .parseInt()
      .value()
}


function distance(pointA, pointB) {
  return _.chain (pointA)
        .zip(pointB)
        .map(([a, b]) => (a-b) ** 2)
        .sum()
        .value() ** 0.5 
}

// @dev: split dataset into two subsets "training" set & "test" set
// @info: function  SPLITDATASET(data, testCount), where current dataset is input and testCount is output
function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);                   // suffle dataset

  const testSet = _.slice(shuffled, 0, testCount);    // test = shuffled data from 0 to testCount
  const trainingSet = _.slice(shuffled, testCount)       // trainSet = shuffled data from testCount to the end

  return [testSet, trainingSet];
}

function minMAX(data, featureCount) {
    const clonedData = _.cloneDeep(data);

    // column of an array
    for(let i = 0; i < featureCount; i++) {
      const column = clonedData.map(row => row[i]);
      const min = _.min(column);
      const max = _.max(column);    

    // row of an array
      for( let j = 0; j < clonedData.length; j++) {
        clonedData[j][i] =(clonedData[j][i] - min) / (max - min);
        }
    }

    return clonedData;
}

