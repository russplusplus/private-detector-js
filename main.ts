import * as tf from '@tensorflow/tfjs-node';
import { read_image } from './utils/preprocess';

const model_path: string = 'file://model/model.json'
const file_paths: string[] = [
    './samples/no/1.jpg',
    './samples/no/2.jpg',
    './samples/no/3.jpg',
    './samples/no/4.jpg',
    './samples/no/5.jpg',
    './samples/no/6.jpg',
    './samples/no/7.jpg',
    './samples/yes/1.jpg',
    './samples/yes/2.jpg',
    './samples/yes/3.jpg'
]

const probs: Promise<number[]> = RunInference(model_path, file_paths)
console.log('probs:', probs)

export async function RunInference(modelPath: string, filePaths: string[]): Promise<number[]> {

    return new Promise(async (resolve, reject) => {
        try {
            const model = await tf.loadGraphModel(modelPath);

            let probs: number[] = []
    
            for (let path of filePaths) {
                const imageTensor = read_image(path)
                const output: tf.Tensor3D = model.predict(imageTensor)
                const data = output.dataSync()
                probs.push(data[0])
                console.log('output:', data[0]*100, '%')
            }
    
            resolve(probs)
        } catch (error) {
            reject(error)
        }
    })
}