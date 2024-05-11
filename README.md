# private-detector-js

This repo is for using Bumble's Private Detector™ model in TypeScript.

## Model

The model was converted from Python using TensorFlow.js's converter tool. The SavedModel can be created by following the instructions [here](https://www.npmjs.com/package/@tensorflow/tfjs-converter).


## Install
```shell
npm install private-detector-js
```

## Initialize

### ESM
```javascript
const { RunInference } = import 'private-detector-js'
```
### CommonJS
```javascript
const pdjs = require('private-detector-js')
```
## Usage
```javascript
const probabilities = await RunInference(modelPaths, filePaths)
```
