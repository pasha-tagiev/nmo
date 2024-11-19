package matrix

import (
	"errors"
	"math"
)

var ErrSingularMatrix = errors.New("singular matrix")

var epsilon = math.Nextafter(1, 2) - 1

func Inverse(mx Matrix) (Matrix, error) {
	augmented := Concat(mx, E(mx.M()))

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

	inverse := make(Matrix, mx.M())
	for i := range inverse {
		inverse[i] = augmented[i][mx.N():]
	}

	return inverse, nil
}
