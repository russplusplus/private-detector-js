/*
Module for preprocessing images fed to the model
*/

import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';

export async function read_image(filename: string): tf.Tensor {
    // read the image from the file
    console.log('in read_image. filename:', filename)
    const imageBuffer = fs.readFileSync(filename)
    const imageTensor: tf.Tensor = tf.node.decodeImage(imageBuffer)
    console.log('imageTensor shape:', imageTensor.shape)
    // console.log('imageTensor:', await imageTensor.data())
    const preprocessedImage = preprocess_for_evaluation(imageTensor, 480);
    console.log('preprocessedImage shape:', preprocessedImage.shape)
    // console.log('preprocessedImage:', await preprocessedImage.data())
    const reshapedImage = tf.reshape(preprocessedImage, [-1, 691200]);
    console.log('reshapedImage shape:', reshapedImage.shape)
    return reshapedImage
}

export function preprocess_for_evaluation(image: tf.Tensor, image_size: number): tf.Tensor {
    
    console.log('in preprocess_for_evaluation. image shape:', image.shape, '. image_size:', image_size)
    let tensor = pad_resize_image(image, [image_size, image_size]);

    // TENSOR IS DIFFERENT THAN IN PYTHON HERE

    // const one_string = image.toString()
    // fs.writeFile('after_pad_resize.txt', one_string, (err) => {
    //     if (err) {
    //         console.log('Error writing after_pad_resize:', err)
    //     }
    // })
    tensor = tf.cast(tensor, 'float32');
    tensor = tensor.sub(128);
    tensor = tensor.div(128);
    return tensor
}

export function pad_resize_image(image: tf.Tensor, dims: [number, number]): tf.Tensor {
    console.log('in pad_resize_image. image shape:', image.shape, '. dims:', dims)
    // // pad the image to make it square
    // const [height, width] = dims
    // const [origHeight, origWidth] = image.shape

    // const [height, width] = dims;
    // const [origHeight, origWidth] = image.shape;
    // const pad = Math.abs(origHeight - origWidth) / 2;
    // const padDims = [[0, 0], [pad, pad], [0, 0]];
    // const paddedImage = tf.pad(image, padDims, 255);
    // // resize the image to the desired dimensions
    // const resizedImage = tf.image.resizeBilinear(paddedImage, [height, width]);
    // return resizedImage;



    // const [height, width] = dims
    // const [origHeight, origWidth] = image.shape
    // const pad = Math.abs(origHeight - origWidth) / 2
    // const padDims = [[0, 0], [pad, pad], [0, 0]]
    // const paddedImage = tf.pad(image, padDims, 255)

    // // Because we cannot set preserve_aspect_ratio to true with tensorflowjs, we resize after the image is padded into a square
    // const resizedImage = tf.image.resizeBilinear(paddedImage, [height, width]);



    const aspectRatio: number = image.shape[1] / image.shape[0];
    // console.log('aspectRatio:', aspectRatio)
    const targetHeight: number = dims[0];
    // console.log('targetHeight:', targetHeight)
    const targetWidth: number = Math.round(targetHeight * aspectRatio);
    // console.log('targetWidth:', targetWidth)

    // const one_string = image.toString()
    // fs.writeFile('before_image_resize.txt', one_string, (err) => {
    //     if (err) {
    //         console.log('Error writing after_pad_resize:', err)
    //     }
    // })

    
    let tensor: tf.Tensor3D = tf.image.resizeBilinear(
        image,
        [targetHeight, targetWidth],
        false,
        true
    )

    // TENSOR IS SAME AS IN PYTHON HERE
    // THANKS TO halfPixelCenters = true

    const shape: [number, number, number] = tensor.shape

    const sxd = dims[1] - shape[1]
    const syd = dims[0] - shape[0]

    const sx = tf.cast(
        sxd / 2,
        'int32'
    )
    console.log('sx:', sx.dataSync()[0])

    const sy = tf.cast(
        syd / 2,
        'int32'
    )
    console.log('sy:', sy.dataSync()[0])

    const a = sy.dataSync()[0]
    const b = syd - sy.dataSync()[0]
    const c = sx.dataSync()[0]
    const d = sxd - sx.dataSync()[0]

    console.log('a:', a)
    console.log('b:', b)
    console.log('c:', c)
    console.log('d:', d)
    
    const paddingsArr = [
        [a, b],
        [c, d],
        [0, 0]
    ]
    console.log('paddingsArr:', paddingsArr)
    // const paddings = tf.tensor(paddingsArr, [3,2], 'int32')
    // const paddingsTensor = paddings.reshape([3,2])

    // const paddings = tf.tensor([
    //     [sy, syd - sy],
    //     [sx, sxd - sx],
    //     [0,0]
    // ])
    // console.log('paddingsTensor:', await paddingsTensor.data())

    console.log('tensor shape before pad:', tensor.shape)

    const tensorPadded: tf.Tensor = tf.pad(tensor, paddingsArr, 128)

    console.log('tensor shape after pad:', tensorPadded.shape)
    
    // const one_string = image.toString()
    // fs.writeFile('after_image_resize.txt', one_string, (err) => {
    //     if (err) {
    //         console.log('Error writing after_pad_resize:', err)
    //     }
    // })


    return tensorPadded
}