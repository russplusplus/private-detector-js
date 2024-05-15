# private-detector-js

This package provides the necessary image preprocessing to use Bumble's Private Detector™ model in JavaScript or TypeScript.

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
const { RunInference } = require('private-detector-js')
```
## Usage
```javascript
const probabilities = await RunInference(modelPath, filePaths, options)
```

| Parameter | Type   | Description                                                                                                                   |
| --------- | ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| modelPath | String | path to the model.json file                                                                                                   |
| filePaths | Array  | path(s) to the image(s) to be evaulated                                                                                         |
| options   | Object | options to manually provide paths to the model's weight files. Currently, weightUrlConverter and weightPathPrefix are supported. More information [here](https://js.tensorflow.org/api/latest/) |



## Model

The model can be found [here](https://huggingface.co/russplusplus/bumble-private-detector-js/tree/main).

This model was converted from Python using TensorFlow.js's converter tool, as described [here](https://www.npmjs.com/package/@tensorflow/tfjs-converter) to create a SavedModel.