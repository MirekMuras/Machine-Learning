const _ = require('lodash');

const loadData = () => {
    const random = _.range(0, 999999);           //create reference to big array with range from 0 to 999999

    return random;
};

// without random reference to the array , the Garbage Collection with kick in here and save some memory
const data = loadData();

debugger;