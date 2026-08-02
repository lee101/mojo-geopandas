"""Broad-phase spatial kernels exported to the Python bindings."""

from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


@export("mgpd_bbox_intersects")
def mgpd_bbox_intersects(left_addr: Int, right_addr: Int, mask_addr: Int, nleft: Int, nright: Int) abi("C"):
    """Write an nleft-by-nright row-major inclusive AABB intersection mask."""
    def compute_row(i: Int) capturing:
        var left = Ptr(unsafe_from_address=left_addr)
        var right = Ptr(unsafe_from_address=right_addr)
        var mask = BPtr(unsafe_from_address=mask_addr)
        var j = 0
        while j + W <= nright:
            var rxmin = right.load[width=W](j)
            var rymin = right.load[width=W](nright + j)
            var rxmax = right.load[width=W](2 * nright + j)
            var rymax = right.load[width=W](3 * nright + j)
            var hits = (
                SIMD[DType.float64, W](left[i]).le(rxmax)
                & rxmin.le(SIMD[DType.float64, W](left[2 * nleft + i]))
                & SIMD[DType.float64, W](left[nleft + i]).le(rymax)
                & rymin.le(SIMD[DType.float64, W](left[3 * nleft + i]))
            )
            mask.store(
                i * nright + j,
                hits.select(SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)),
            )
            j += W
        while j < nright:
            mask[i * nright + j] = 1 if (
                left[i] <= right[2 * nright + j]
                and right[j] <= left[2 * nleft + i]
                and left[nleft + i] <= right[3 * nright + j]
                and right[nright + j] <= left[3 * nleft + i]
            ) else 0
            j += 1
    for i in range(nleft):
        compute_row(i)


@export("mgpd_bbox_contains")
def mgpd_bbox_contains(left_addr: Int, right_addr: Int, mask_addr: Int, nleft: Int, nright: Int) abi("C"):
    """Write whether each left AABB contains each right AABB."""
    var left = Ptr(unsafe_from_address=left_addr)
    var right = Ptr(unsafe_from_address=right_addr)
    var mask = BPtr(unsafe_from_address=mask_addr)
    for i in range(nleft):
        var j = 0
        while j + W <= nright:
            var rxmin = right.load[width=W](j)
            var rymin = right.load[width=W](nright + j)
            var rxmax = right.load[width=W](2 * nright + j)
            var rymax = right.load[width=W](3 * nright + j)
            var hits = (
                SIMD[DType.float64, W](left[i]).le(rxmin)
                & SIMD[DType.float64, W](left[nleft + i]).le(rymin)
                & SIMD[DType.float64, W](left[2 * nleft + i]).ge(rxmax)
                & SIMD[DType.float64, W](left[3 * nleft + i]).ge(rymax)
            )
            mask.store(
                i * nright + j,
                hits.select(SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)),
            )
            j += W
        while j < nright:
            mask[i * nright + j] = 1 if (
                left[i] <= right[j]
                and left[nleft + i] <= right[nright + j]
                and left[2 * nleft + i] >= right[2 * nright + j]
                and left[3 * nleft + i] >= right[3 * nright + j]
            ) else 0
            j += 1


@export("mgpd_bbox_distance2")
def mgpd_bbox_distance2(left_addr: Int, right_addr: Int, distance_addr: Int, nleft: Int, nright: Int) abi("C"):
    """Write squared minimum distances between every pair of AABBs."""
    var left = Ptr(unsafe_from_address=left_addr)
    var right = Ptr(unsafe_from_address=right_addr)
    var distances = Ptr(unsafe_from_address=distance_addr)
    for i in range(nleft):
        var j = 0
        while j + W <= nright:
            var rxmin = right.load[width=W](j)
            var rymin = right.load[width=W](nright + j)
            var rxmax = right.load[width=W](2 * nright + j)
            var rymax = right.load[width=W](3 * nright + j)
            var dx = max(
                SIMD[DType.float64, W](0.0),
                max(
                    rxmin - SIMD[DType.float64, W](left[2 * nleft + i]),
                    SIMD[DType.float64, W](left[i]) - rxmax,
                ),
            )
            var dy = max(
                SIMD[DType.float64, W](0.0),
                max(
                    rymin - SIMD[DType.float64, W](left[3 * nleft + i]),
                    SIMD[DType.float64, W](left[nleft + i]) - rymax,
                ),
            )
            distances.store(i * nright + j, dx * dx + dy * dy)
            j += W
        while j < nright:
            var dx = 0.0
            var dy = 0.0
            if left[2 * nleft + i] < right[j]:
                dx = right[j] - left[2 * nleft + i]
            elif right[2 * nright + j] < left[i]:
                dx = left[i] - right[2 * nright + j]
            if left[3 * nleft + i] < right[nright + j]:
                dy = right[nright + j] - left[3 * nleft + i]
            elif right[3 * nright + j] < left[nleft + i]:
                dy = left[nleft + i] - right[3 * nright + j]
            distances[i * nright + j] = dx * dx + dy * dy
            j += 1
