var SentimentModel = require('../models/sentimentModel');

var express = require('express');
var router = express.Router();

const model = new SentimentModel();
model.prepareModel();


router.get('/:text', async function(req, res, next) {
  console.log(req.params.text);
  res.json({'sentiment' : await model.predict(req.params.text) });
});

router.post('/', async function(req, res, next){
  console.log(req.body);      
  res.json(await model.predict(req.body.text));
});

module.exports = router;
