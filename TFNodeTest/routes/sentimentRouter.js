var SentimentModel = require('../models/sentimentModel');

var express = require('express');
var router = express.Router();

router.get('/', async function(req, res, next) {
  const generator = new SentimentModel();
  await generator.prepareModel();
  res.json({'resp' : 1 });
});

module.exports = router;
