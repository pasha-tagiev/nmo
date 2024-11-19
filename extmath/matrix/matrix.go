package matrix

import (
	"nmo/extmath/vector"
	"slices"
)

type Matrix []vector.Vector

func New(m, n int) Matrix {
	mx := make(Matrix, m)
	for i := range mx {
		mx[i] = make(vector.Vector, n)
	}
	return mx
}

func E(m int) Matrix {
	mx := New(m, m)
	for i := range mx.M() {
		mx[i][i] = 1.0
	}
	return mx
}

func (mx Matrix) M() int {
	return len(mx)
}

func (mx Matrix) N() int {
	if mx.M() == 0 {
		return 0
	}

	return len(mx[0])
}

func Concat(lhv, rhv Matrix) Matrix {
	mx := make(Matrix, lhv.M())
	for i := range lhv {
		mx[i] = slices.Concat(lhv[i], rhv[i])
	}

	return mx
}

func MulVector(lhv Matrix, rhv vector.Vector) vector.Vector {
	res := make(vector.Vector, lhv.N())

	for i := range lhv {
		res[i] = vector.Dot(lhv[i], rhv)
	}

	return res
}
