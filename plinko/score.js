const outputs = [];
const predictionPoint = 300;
const k = 3;

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);

  //console.log(outputs);
}

function runAnalysis() {
  // code here to analyze stuff
   const bucket = _.chain(outputs)
      .map(row => [distance(row[0]), row[3]])
      .sortBy(row => row[0])
      .slice(0, k)
      .countBy(row => row[1])
      .toPairs()
      .sortBy(row => row[1])
      .last()
      .first()
      .parseInt()
      .value()

    console.log('The ball will probably fall into ',bucket);
}

function distance(point) {
  return Math.abs(point - predictionPoint);
}

// @dev: split dataset into two subsets "training" set & "test" set
// @info: function  SPLITDATASET(data, testCount), where current dataset is input and testCount is output
function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);                   // suffle dataset

  const testSet = _.slice(shuffled, 0, testCount);    // test = shuffled data from 0 to testCount
  const trainSet = _.slice(shuffled, testCount)       // trainSet = shuffled data from testCount to the end

  return [testSet, trainSet];



  

}

