# private-detector-js

This repo is for using Bumble's Private Detector™ model in TypeScript.

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
const probabilities = await RunInference(modelPath, filePaths, options)
```

modelPath: string
- path of the model.json file

filePaths: string[]
- path(s) of the images to be evaluated

options: { 
    weightPathPrefix?: string, 
    weightUrlConverter?: (filename: string) => Promise<string> 
}
- options to manually provide paths to the model's weight files. More information [here](https://js.tensorflow.org/api/latest/)

## Model

The model was converted from Python using TensorFlow.js's converter tool. The SavedModel can be created by following the instructions [here](https://www.npmjs.com/package/@tensorflow/tfjs-converter).