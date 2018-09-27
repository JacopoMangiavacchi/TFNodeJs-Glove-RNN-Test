'use strict';

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const fs = require('fs');


module.exports = class SentimentModel {
  async prepareModel() {
    console.log('Loading Tensorflow model...');
    this.model = await tf.loadModel('file://../Python/tfjsmodel/model.json');

    console.log('Loading word mappings...');


    let rawdata = fs.readFileSync('../Python/words.json');  
    this.words = JSON.parse(rawdata);
  }

  cleanText(text) {
    return text
      .replace(/[^\w\n\s]/g, '')
      .replace('\n', ' \n ');
  }

  textToVec(text) {
    text = this.cleanText(text);
    const tokens = text.split(' ');
    const tokenVector = [];
    let i = 0;
    while (i < tokens.length) {
      const newToken = this.words[tokens[i]];
      if (typeof newToken === 'undefined') {
        tokens.splice(i, 1);
        continue;
      }
      tokenVector.push(newToken);
      i++;
    }
    return tokenVector;
  }

  padVector(vector, length) {
    const t = tf.tensor(vector);
    return t.pad([[length - t.shape[0], 0]]);
  }

  async predict(text) {
    const tokenVector = this.textToVec(text);
    console.log('Token vector', tokenVector);

    const paddedTensor = this.padVector(tokenVector, 50).reshape([1, 50]); // Sync with Python training
    paddedTensor.print();

    const prediction = await this.model.predict(paddedTensor);
    prediction.print()

    return prediction.toString();
  }
}
