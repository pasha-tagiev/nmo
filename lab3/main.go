package main

import (
	"fmt"
	"math"
	"nmo/extmath"
	"nmo/extmath/vector"
	"slices"
)

func SubgradientDescent(f extmath.MathFunc, x0 vector.Vector, h float64, maxIteration int) vector.Vector {
	xn := slices.Clone(x0)
	xb := slices.Clone(x0)

	fxb := f(xb...)

	grad := extmath.Grad(f, len(x0), h)

	for k := range maxIteration {
		g := grad(xn...)
		if g.Norm2() < h {
			break
		}

		gamma := 1 / float64(k+1)

		alpha := (f(xn...) - fxb + gamma)
		alpha /= vector.Dot(g, g)

		xn.Sub(g.MulNum(alpha))

		if f(xn...) < fxb {
			xb = slices.Clone(xn)
			fxb = f(xb...)
		}
	}

	return xb
}

func main() {
	h := 1e-6

	// f1(x) = |x-3|+|x+1|
	f1 := func(x ...float64) float64 {
		return math.Abs(x[0]-3) + math.Abs(x[0]+1)
	}

	x0 := vector.Vector{5}
	res := SubgradientDescent(f1, x0, h, 100)

	fmt.Printf("f1(%v) = %v\n", res, f1(res...))

	// f2(x, y) = |x^2+y^2-4|
	f2 := func(x ...float64) float64 {
		return math.Pow(x[0], 2) + math.Pow(x[1], 2) - 4
	}

	x0 = vector.Vector{1, 1}
	res = SubgradientDescent(f2, x0, h, 1000)

	fmt.Printf("f2(%v) = %v\n", res, f2(res...))

	// f3(x, y) = |x|+|y-3|+3
	f3 := func(x ...float64) float64 {
		return math.Abs(x[0]) + math.Abs(x[1]-3) + 3
	}

	x0 = vector.Vector{3, 3}
	res = SubgradientDescent(f3, x0, h, 1000)

	fmt.Printf("f3(%v) = %v\n", res, f3(res...))

}
