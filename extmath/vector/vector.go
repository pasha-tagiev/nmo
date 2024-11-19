package vector

import "math"

type Vector []float64

func Dot(lhv, rhv Vector) float64 {
	dot := 0.0
	for i := range lhv {
		dot += lhv[i] * rhv[i]
	}

	return dot
}

func (v Vector) Sub(rhv Vector) {
	for i := range v {
		v[i] -= rhv[i]
	}
}

func (vec Vector) Norm2() float64 {
	return math.Sqrt(Dot(vec, vec))
}
