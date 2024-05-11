export type Paddings = [
    [number, number], [number, number], [number, number]
]

export type Options = {
    weightUrlConverter?: (filename: string) => Promise<string>,
    weightPathPrefix?: string
}