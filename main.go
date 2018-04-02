package main

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	// replace myModel and myTag with the appropriate exported names in the chestrays-keras-binary-classification.ipynb
	model, err := tf.LoadSavedModel("myModel", []string{"myTag"}, nil)

	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}

	defer model.Session.Close()

	tensor, _ := tf.NewTensor([1][250][250][3]float32{})

	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("inputLayer_input").Output(0): tensor, // Replace this with your input layer name
		},
		[]tf.Output{
			model.Graph.Operation("inferenceLayer/Sigmoid").Output(0), // Replace this with your output layer name
		},
		nil,
	)

	if err != nil {
		fmt.Printf("Error running the session with input, err: %s\n", err.Error())
		return
	}

	fmt.Printf("Result value: %v \n", result[0].Value())

}
