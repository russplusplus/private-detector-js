/*
Module for preprocessing images fed to the model
*/

import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import type { Paddings } from '../types';
import axios from 'axios'

export async function read_image(filepath: string): Promise<tf.Tensor2D> {

    let imageBuffer: Buffer

    if (filepath.startsWith('http')) {
        const imageRes = await axios.get(filepath, { responseType: 'arraybuffer' })
        imageBuffer = Buffer.from(imageRes.data, "utf-8")
    } else {
        imageBuffer = fs.readFileSync(filepath)
    }

    // tf.node.decodeImage will always return a 3D tensor when 3 channels are specified, but TypeScript doesn't know this so we need to assert the type
    const imageTensor: tf.Tensor3D = tf.node.decodeImage(imageBuffer, 3) as tf.Tensor3D

    const preprocessedImage: tf.Tensor3D = preprocess_for_evaluation(imageTensor, 480)
    
    const reshapedImage: tf.Tensor2D = tf.reshape(preprocessedImage, [-1, 691200]);

    return reshapedImage
}

function preprocess_for_evaluation(image: tf.Tensor3D, image_size: number): tf.Tensor3D {
    
    const paddedResizedImage: tf.Tensor3D = pad_resize_image(image, [image_size, image_size]);

    const castImage: tf.Tensor3D = tf.cast(paddedResizedImage, 'float32');
    const subtractedImage: tf.Tensor3D = castImage.sub(128);
    const dividedImage: tf.Tensor3D = subtractedImage.div(128);

    return dividedImage
}

function pad_resize_image(image: tf.Tensor3D, dims: [number, number]): tf.Tensor3D {
   
    // aspect ratio = width / height
    const aspectRatio: number = image.shape[1] / image.shape[0];

    let targetHeight: number
    let targetWidth: number

    // First, we resize the image so that the longer side is equal to the target size (480), while preserving the aspect ratio.
    // Because tf.image.resizeBilinear does not support the "preserve_aspect_ratio" argument like in Python, we need to calculate the desired dimensions manually.
    if (aspectRatio < 1) {
        targetHeight = dims[0]
        targetWidth = Math.round(targetHeight * aspectRatio)
    } else if (aspectRatio > 1) {
        targetWidth = dims[1]
        targetHeight = Math.round(targetWidth / aspectRatio)
    } else {
        targetHeight = dims[0]
        targetWidth = dims[1]
    }
    
    // halfPixelCenters must be set to true to match the default behavior of tf.image.resize in Python
    let tensor: tf.Tensor3D = tf.image.resizeBilinear(
        image,
        [targetHeight, targetWidth],
        false,
        true
    )

    // Next, we calculate how much padding is needed to make the image square
    const shape: [number, number, number] = tensor.shape

    const dWidth: number = dims[1] - shape[1]
    const dHeight: number = dims[0] - shape[0]

    const paddingX: tf.Tensor1D = tf.cast(
        dWidth / 2,
        'int32'
    ) as tf.Tensor1D 

    const paddingY: tf.Tensor1D = tf.cast(
        dHeight / 2,
        'int32'
    ) as tf.Tensor1D

    // Because we are using integer tensors, padding will be asymmetrical if the required total padding is an odd number of cells. 
    // Therefore, we need to calculate the padding for each side separately.
    const paddingT: number = paddingY.dataSync()[0]
    const paddingB: number = dHeight - paddingY.dataSync()[0]
    const paddingL: number = paddingX.dataSync()[0]
    const paddingR: number = dWidth - paddingX.dataSync()[0]

    const paddingsArr: Paddings = [
        [paddingT, paddingB],
        [paddingL, paddingR],
        [0, 0]
    ]
  
    const tensorPadded: tf.Tensor3D = tf.pad(tensor, paddingsArr, 128)

    return tensorPadded
}