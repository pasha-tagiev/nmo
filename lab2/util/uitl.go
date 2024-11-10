package util

import (
	"errors"
	"math"
	"nmo/lab2/matrix"
	"nmo/lab2/vector"
	"slices"
)

var epsilon = math.Nextafter(1, 2) - 1

var ErrSingularMatrix = errors.New("singular matrix")

type MathFunc func(x ...float64) float64

func D(f MathFunc, h float64, varIndexes ...int) MathFunc {
	for _, v := range varIndexes {
		c := f
		f = func(x ...float64) float64 {
			lx := slices.Clone(x)
			rx := slices.Clone(x)

			lx[v] += h
			rx[v] -= h

			return (c(lx...) - c(rx...)) / (2 * h)
		}
	}

	return f
}

type GradFunc func(x ...float64) vector.Vector

func Grad(f MathFunc, dim int, h float64) GradFunc {
	part := make([]MathFunc, dim)

	for i := range part {
		part[i] = D(f, h, i)
	}

	return func(x ...float64) vector.Vector {
		grad := make(vector.Vector, dim)

		for i := range grad {
			grad[i] = part[i](x...)
		}

		return grad
	}
}

type HessianFunc func(x ...float64) matrix.Matrix

func Hessian(f MathFunc, dim int, h float64) HessianFunc {
	raw := make([][]MathFunc, dim)

	for i := range raw {
		row := make([]MathFunc, dim)

		for j := range row {
			row[j] = D(f, h, i, j)
		}

		raw[i] = row
	}

	return func(x ...float64) matrix.Matrix {
		res := make(matrix.Matrix, dim)

		for i := range res {
			row := make(vector.Vector, dim)

			for j := range row {
				row[j] = raw[i][j](x...)
			}

			res[i] = row
		}

		return res
	}
}

func Inverse(mx matrix.Matrix) (matrix.Matrix, error) {
	augmented := matrix.Concat(mx, matrix.E(mx.M()))

	for i := range augmented {
		if math.Abs(augmented[i][i]) < epsilon {
			return nil, ErrSingularMatrix
		}

		divisor := augmented[i][i]

		for j := range 2 * mx.N() {
			augmented[i][j] /= divisor
		}

		for k := range mx.M() {
			if i != k {
				factor := augmented[k][i]
				for j := range 2 * mx.N() {
					augmented[k][j] -= factor * augmented[i][j]
				}
			}
		}
	}

	inverse := make(matrix.Matrix, mx.M())
	for i := range inverse {
		inverse[i] = augmented[i][mx.N():]
	}

	return inverse, nil
}
