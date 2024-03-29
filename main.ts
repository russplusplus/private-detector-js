import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import { read_image } from './utils/preprocess';

console.log("Hello via Bun!");

async function loadModel() {
    const model = await tf.loadGraphModel('file://model/model.json');
    console.log("Model loaded. model");

    // const imageBuffer = Buffer.from(base64str, 'base64')
    // const imageBuffer = Buffer.from('./samples/no/1.jpg')
    // const imageBuffer = fs.readFileSync('./samples/no/1.jpg')
    // console.log('imageBuffer:', imageBuffer)
    // get the tensor
    // decodeImage can only return 3D or 4D tensor
    // decomeImage also takes dtype, only int32 is supported currently
    // const imageTensor = tf.node.decodeImage(imageBuffer)

    const filepaths = {
        no: ['./samples/no/1.jpg',
            './samples/no/2.jpg',
            './samples/no/3.jpg',
            './samples/no/4.jpg',
            './samples/no/5.jpg',
            './samples/no/6.jpg',
            './samples/no/7.jpg'
        ],
        yes: ['./samples/yes/1.jpg',
            './samples/yes/2.jpg',
            './samples/yes/3.jpg'
        ]
    }



    // imageTensor.reshape([-1, 691200])
    // const imageTensor2D = 

    // const tensor3D = imageTensor.expandDims(2)
    // const newJpg = tf.node.encodeJpeg(tensor3D)
    // input must be tf.float16 2D tensor
    for (let filepath of filepaths.no) {
        const imageTensor = await read_image(filepath)
        const output = await model.predict(imageTensor)
        const data = await output.data()
        console.log('NO - output:', data[0]*100, '%')
    }
    for (let filepath of filepaths.yes) {
        const imageTensor = await read_image(filepath)
        const output = await model.predict(imageTensor)
        const data = await output.data()
        console.log('YES - output:', data[0]*100, '%')
    }

    // const imageTensor = await read_image('./samples/no/1.jpg')
    // console.log('imageTensor:', imageTensor)
    // const output = model.predict(imageTensor)
    // // console.log('output:', output)
    //     const data = await output.data()
    //     console.log('NO - output:', data[0]*100, '%')

    
}

loadModel()