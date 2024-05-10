import * as tf from '@tensorflow/tfjs-node';
import { read_image } from './src/preprocess';

export async function RunInference(modelPath: string, filePaths: string[]): Promise<number[]> {
    return new Promise(async (resolve, reject) => {
        try {
            const model: tf.GraphModel = await tf.loadGraphModel(modelPath);
            let probs: number[] = []
    
            for (let path of filePaths) {
                const imageTensor = await read_image(path)
                const output: tf.Tensor2D = model.predict(imageTensor) as tf.Tensor2D
                const data: Float32Array = output.dataSync() as Float32Array
                probs.push(data[0])
            }
            resolve(probs)
        } catch (error) {
            reject(error)
        }
    })
}