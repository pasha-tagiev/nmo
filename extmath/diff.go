package extmath

import (
	"nmo/extmath/matrix"
	"nmo/extmath/vector"
	"slices"
)

type MathFunc func(x ...float64) float64

func D(f MathFunc, h float64, varIndexes ...int) MathFunc {
	diff := func(f MathFunc, h float64, v int) MathFunc {
		return func(x ...float64) float64 {
			lx := slices.Clone(x)
			rx := slices.Clone(x)

			lx[v] += h
			rx[v] -= h

			return (f(lx...) - f(rx...)) / (2 * h)
		}
	}

	for _, v := range varIndexes {
		f = diff(f, h, v)
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
