import random, unittest
from prepare import smallestenclosingcircle


class SmallestEnclosingCircleTest(unittest.TestCase):

	def test_matching_naive_algorithm(self):
		TRIALS = 1000
		for _ in range(TRIALS):
			points = _make_random_points(random.randint(1, 30))
			print(points)
			reference = _smallest_enclosing_circle_naive(points)
			actual = smallestenclosingcircle.make_circle(points)
			print("r:", reference)
			print("a", actual)
			self.assertAlmostEqual(actual[0], reference[0], delta=_EPSILON)
			self.assertAlmostEqual(actual[1], reference[1], delta=_EPSILON)
			self.assertAlmostEqual(actual[2], reference[2], delta=_EPSILON)

	def test_translation(self):
		TRIALS = 100
		CHECKS = 10
		for _ in range(TRIALS):
			points = _make_random_points(random.randint(1, 300))
			reference = smallestenclosingcircle.make_circle(points)

			print(points)
			print("r", reference)
			for _ in range(CHECKS):
				dx = random.gauss(0, 1)
				dy = random.gauss(0, 1)
				newpoints = [(x + dx, y + dy) for (x, y) in points]

				translated = smallestenclosingcircle.make_circle(newpoints)
				self.assertAlmostEqual(translated[0], reference[0] + dx, delta=_EPSILON)
				self.assertAlmostEqual(translated[1], reference[1] + dy, delta=_EPSILON)
				self.assertAlmostEqual(translated[2], reference[2], delta=_EPSILON)

	def test_scaling(self):
		TRIALS = 100
		CHECKS = 10
		for _ in range(TRIALS):
			points = _make_random_points(random.randint(1, 300))
			reference = smallestenclosingcircle.make_circle(points)

			print(points)
			print("r", reference)
			for _ in range(CHECKS):
				scale = random.gauss(0, 1)
				newpoints = [(x * scale, y * scale) for (x, y) in points]

				scaled = smallestenclosingcircle.make_circle(newpoints)
				self.assertAlmostEqual(scaled[0], reference[0] * scale, delta=_EPSILON)
				self.assertAlmostEqual(scaled[1], reference[1] * scale, delta=_EPSILON)
				self.assertAlmostEqual(scaled[2], reference[2] * abs(scale), delta=_EPSILON)


def _make_random_points(n):
	if random.random() < 0.2:  # Discrete lattice (to have a chance of duplicated points)
		return [(random.randrange(10), random.randrange(10)) for _ in range(n)]
	else:  # Gaussian distribution
		return [(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(n)]


def _smallest_enclosing_circle_naive(points):
	# Degenerate cases
	if len(points) == 0:
		return None
	elif len(points) == 1:
		return (points[0][0], points[0][1], 0)

	# Try all unique pairs
	result = None
	for i in range(len(points)):
		p = points[i]
		for j in range(i + 1, len(points)):
			q = points[j]
			c = smallestenclosingcircle.make_diameter(p, q)
			if (result is None or c[2] < result[2]) and \
					all(smallestenclosingcircle.is_in_circle(c, r) for r in points):
				result = c
	if result is not None:
		return result  # This optimization is not mathematically proven

	# Try all unique triples
	for i in range(len(points)):
		p = points[i]
		for j in range(i + 1, len(points)):
			q = points[j]
			for k in range(j + 1, len(points)):
				r = points[k]
				c = smallestenclosingcircle.make_circumcircle(p, q, r)
				if c is not None and (result is None or c[2] < result[2]) and \
						all(smallestenclosingcircle.is_in_circle(c, s) for s in points):
					result = c

	if result is None:
		raise AssertionError()
	return result


_EPSILON = 1e-12

if __name__ == "__main__":
	unittest.main()