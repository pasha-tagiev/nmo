package main

import (
	"fmt"
	"math"
	"nmo/extmath"
	"nmo/extmath/matrix"
	"nmo/extmath/vector"
	"slices"
)

func Newton(f extmath.MathFunc, x0 vector.Vector, h float64, maxIteration int) (_ vector.Vector, err error) {
	xn := slices.Clone(x0)

	dim := len(x0)
	grad := extmath.Grad(f, dim, h)
	H := extmath.Hessian(f, dim, h)

	for range maxIteration {
		vectorGrad := grad(xn...)

		var HInv matrix.Matrix
		if HInv, err = matrix.Inverse(H(xn...)); err != nil {
			return
		}

		xn.Sub(matrix.MulVector(HInv, vectorGrad))

		if vectorGrad.Norm2() < h {
			break
		}
	}

	return xn, nil
}

func PrintAnswer(label string, f extmath.MathFunc, x0 ...float64) {
	answer, err := Newton(f, x0, 1e-2, 100)
	if err != nil {
		fmt.Println(label, "вырожденная матрица Гессе")
		return
	}

	fmt.Println(label, "минимум в точке:", answer)
}

func main() {
	// Графики функций и ответы в папке image
	PrintAnswer("(1)", func(x ...float64) float64 {
		return math.Pow(x[0], 4) / math.Log(x[0])
	}, 2)

	PrintAnswer("(2)", func(x ...float64) float64 {
		return math.Pow(x[0], 2) + math.Pow(x[1], 2) - 4*x[0] - 6*x[1] + 13
	}, 1, 1)

	PrintAnswer("(3)", func(x ...float64) float64 {
		return math.Exp(x[0]+x[1]) + math.Pow(x[0], 2) + math.Pow(x[1], 2)
	}, -2, -2)

	// Тут будет вырожденная матрица Гессе
	PrintAnswer("(4)", func(x ...float64) float64 {
		return math.Log(math.Pow(x[0], 2)+math.Pow(x[1], 2)+1) - x[0] - x[1]
	}, 0, 0)

	PrintAnswer("(5)", func(x ...float64) float64 {
		return 3*math.Pow(x[0], 2) + 2*x[0]*x[1] + math.Pow(x[1], 2) - 5*x[0] + 3*x[1] + 10
	}, 1, 1)

	// Ответ должен быть: [2, 3, 4]
	PrintAnswer("(6)", func(x ...float64) float64 {
		return math.Pow(x[0], 2) + math.Pow(x[1], 2) + math.Pow(x[2], 2) - 4*x[0] - 6*x[1] - 8*x[2] + 20
	}, 4, 4, 4)
}
