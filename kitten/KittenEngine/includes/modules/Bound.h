#pragma once
// Jerry Hsu, 2021

#include "Common.h"
#include <cmath> // for fminf, fmaxf

namespace Kitten {
	template<int dim = 3, typename Real = float>
	struct Bound {
		vec<dim, Real, defaultp> min;
		vec<dim, Real, defaultp> max;

		KITTEN_FUNC_DECL Bound() :
			min(vec3(INFINITY)),
			max(vec3(-INFINITY)) {
		}

		KITTEN_FUNC_DECL Bound(vec<dim, Real, defaultp> center) : min(center), max(center) {}
		KITTEN_FUNC_DECL Bound(vec<dim, Real, defaultp> min, vec<dim, Real, defaultp> max) : min(min), max(max) {}
		KITTEN_FUNC_DECL Bound(const Bound<dim, Real>& b) : min(b.min), max(b.max) {}

		KITTEN_FUNC_DECL inline vec<dim, Real, defaultp> center() const {
			return (min + max) * 0.5f;
		}

		KITTEN_FUNC_DECL inline void absorb(const Bound<dim, Real>& b) {
			for (int i = 0; i < dim; ++i) {
				min[i] = fminf(min[i], b.min[i]);
				max[i] = fmaxf(max[i], b.max[i]);
			}
		}

		KITTEN_FUNC_DECL inline void absorb(const vec<dim, Real, defaultp>& b) {
			for (int i = 0; i < dim; ++i) {
				min[i] = fminf(min[i], b[i]);
				max[i] = fmaxf(max[i], b[i]);
			}
		}

		KITTEN_FUNC_DECL inline bool contains(const vec<dim, Real, defaultp>& point) const {
			return all(lessThanEqual(min, point)) && all(greaterThanEqual(max, point));
		}

		KITTEN_FUNC_DECL inline bool contains(const Bound<dim, Real>& b) const {
			return all(lessThanEqual(min, b.min)) && all(greaterThanEqual(max, b.max));
		}

		// Euclidean min distance between two AABBs
		KITTEN_FUNC_DECL inline Real minDistance(const Bound<dim, Real>& b) const {
			Real sq_dist = 0;
			for (int i = 0; i < dim; ++i) {
				if (max[i] < b.min[i]) {
					Real d = b.min[i] - max[i];
					sq_dist += d * d;
				} else if (b.max[i] < min[i]) {
					Real d = min[i] - b.max[i];
					sq_dist += d * d;
				}
				// else: they overlap in this dimension, contribute 0
			}
			return glm::sqrt(sq_dist); // Euclidean distance
		}

		// Euclidean intersection with threshold
		KITTEN_FUNC_DECL inline bool intersects(const Bound<dim, Real>& b, Real TH) const {
			return minDistance(b) <= TH;
		}

		// Default exact overlap (TH = 28.0)
		KITTEN_FUNC_DECL inline bool intersects(const Bound<dim, Real>& b) const {
			return intersects(b, 28.0);
		}

		// KITTEN_FUNC_DECL inline bool intersects(const Bound<dim, Real>& b) const {
		// 	return !(any(lessThan(max, b.min)) || any(greaterThan(min, b.max)));
		// }

		// KITTEN_FUNC_DECL inline bool intersects(const Bound<dim, Real>& b) const {
		// 	return !(any(lessThanEqual(max, b.min)) || any(greaterThanEqual(min, b.max)));
		// }

		KITTEN_FUNC_DECL inline void pad(Real padding) {
			min -= vec<dim, Real, defaultp>(padding); 
			max += vec<dim, Real, defaultp>(padding);
		}

		KITTEN_FUNC_DECL inline void pad(const vec<dim, Real, defaultp>& padding) {
			min -= padding; 
			max += padding;
		}

		KITTEN_FUNC_DECL inline Real volume() const {
			vec<dim, Real, defaultp> diff = max - min;
			Real v = diff.x;
			for (int i = 1; i < dim; i++) v *= diff[i];
			return v;
		}

		KITTEN_FUNC_DECL inline vec<dim, Real, defaultp> normCoord(const vec<dim, Real, defaultp>& pos) const {
			return (pos - min) / (max - min);
		}

		KITTEN_FUNC_DECL inline vec<dim, Real, defaultp> interp(const vec<dim, Real, defaultp>& coord) const {
			vec<dim, Real, defaultp> pos;
			vec<dim, Real, defaultp> diff = max - min;
			for (int i = 0; i < dim; i++)
				pos[i] = min[i] + diff[i] * coord[i];
			return pos;
		}
	};

	// Specialization for 1D
	template<typename Real>
	struct Bound<1, Real> {
		Real min;
		Real max;

		KITTEN_FUNC_DECL Bound() :
			min(INFINITY),
			max(-INFINITY) {
		}

		KITTEN_FUNC_DECL Bound(Real center) : min(center), max(center) {}
		KITTEN_FUNC_DECL Bound(Real min, Real max) : min(min), max(max) {}

		KITTEN_FUNC_DECL inline Real center() const {
			return (min + max) * 0.5f;
		}

		KITTEN_FUNC_DECL inline void absorb(const Bound<1, Real>& b) {
			min = fminf(min, b.min);
			max = fmaxf(max, b.max);
		}

		KITTEN_FUNC_DECL inline void absorb(Real b) {
			min = fminf(min, b);
			max = fmaxf(max, b);
		}

		KITTEN_FUNC_DECL inline bool contains(Real point) const {
			return min <= point && point <= max;
		}

		KITTEN_FUNC_DECL inline bool contains(const Bound<1, Real>& b) const {
			return min <= b.min && b.max <= max;
		}

		KITTEN_FUNC_DECL inline bool intersects(const Bound<1, Real>& b) const {
			return max > b.min && min < b.max;
		}

		KITTEN_FUNC_DECL inline void pad(Real padding) {
			min -= padding; 
			max += padding;
		}

		KITTEN_FUNC_DECL inline Real volume() const {
			return max - min;
		}

		KITTEN_FUNC_DECL inline Real normCoord(Real pos) const {
			return (pos - min) / (max - min);
		}

		KITTEN_FUNC_DECL inline Real interp(Real coord) const {
			return min + (max - min) * coord;
		}
	};

	typedef Bound<1, float> Range;
}