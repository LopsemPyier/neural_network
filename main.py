from aiaiai import AI
import constant
import numpy as np
import data


def main():
    train_and_save(1)
    ai = load("AI_state/weightsAndBias1.ai")
    test_images=data.load_images("dataset/t10k-images.idx3-ubyte")
    test_labels=data.load_labels("dataset/t10k-labels.idx1-ubyte")
    test(test_images, test_labels, ai)
    

def test(dataset, labels, ai):
    correctanswers=0
    for i,inputs in enumerate(dataset):
        output=ai.run(inputs)
        imax=0
        for j in range(len(output)):
            if output[j]>output[imax]:
                imax=j
        if imax == labels[i]:
            correctanswers+=1
    print(f"correct answer rate ={(correctanswers/len(dataset))*100} %")
        


def train_and_save(n):
    images = data.load_images("dataset/train-images.idx3-ubyte")
    labels = data.load_targets("dataset/train-labels.idx1-ubyte")

    ai = AI(constant.layers, constant.f, constant.fprim)
    ai.train(images, labels)
    data.save_ai(ai, f"AI_state/weightsAndBias{n}.ai")

def load(filename):
    layers, weights, biases = data.load_ai(filename)
    ai=AI(layers, constant.f, constant.fprim, weights, biases)
    return ai





if __name__ == '__main__':
    main()

