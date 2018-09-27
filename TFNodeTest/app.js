var createError = require('http-errors');
var express = require('express');
var path = require('path');
var logger = require('morgan');

var sentimentRouter = require('./routes/sentimentRouter');

var app = express();

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use('/', sentimentRouter);

module.exports = app;
