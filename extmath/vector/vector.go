package vector

import "math"

type Vector []float64

func (v Vector) Sub(rhv Vector) Vector {
	for i := range v {
		v[i] -= rhv[i]
	}

	return v
}

func (v Vector) MulNum(rhv float64) Vector {
	for i := range v {
		v[i] -= rhv
	}

	return v
}

func (v Vector) Norm2() float64 {
	return math.Sqrt(Dot(v, v))
}

func Dot(lhv, rhv Vector) float64 {
	dot := 0.0
	for i := range lhv {
		dot += lhv[i] * rhv[i]
	}

	return dot
}
