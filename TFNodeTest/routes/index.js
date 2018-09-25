const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

class LyricsGenerator {
  async prepareGenerator(logger) {
    logger('Loading Tensorflow model...');
    this.model = await tf.loadModel('data/model.json');

    logger('Loading word mappings...');
    this.words = await fetch('data/words.json')
      .then(res => res.json());

    logger('Creating an inverse word lookup table...');
    this.reverseWords = Object.keys(this.words).reduce((obj, word) => {
      const index = this.words[word];
      obj[index] = word;
      return obj;
    }, {});
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

  async createLyric(textSeed, textLength, randomness) {
    randomness = parseFloat(randomness);
    if (!this.words || !this.model) return '';

    let textOutput = textSeed;
    console.log(`Generating lyric from "${textSeed}" with randomness ${randomness}...`);

    while (textOutput.length < textLength) {
      const tokenVector = this.textToVec(textOutput);
      if (this.debug) console.log('Token vector', tokenVector);

      const paddedTensor = this.padVector(tokenVector, 14).reshape([1, 14]); // TODO
      if (this.debug) paddedTensor.print();

      const prediction = await this.model.predict(paddedTensor);
      if (this.debug) prediction.print()

      // The prediction is a 2D of potentially multiple predictions.
      // Squeeze removes one of the dimensions to make it nicer to work with :-)
      const index = await softmaxSample(prediction.squeeze(), randomness);
      const word = this.reverseWords[index[0]];
      textOutput += ' ' + word;
    }

    return textOutput;
  }
}


var express = require('express');
var router = express.Router();

router.get('/', async function(req, res, next) {
  const generator = new LyricsGenerator();
  await generator.prepareGenerator(console.log);
  res.json({'resp' : 1 });
});

module.exports = router;
