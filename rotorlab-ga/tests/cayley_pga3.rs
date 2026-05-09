//! Integration tests for the PGA3 Cayley table.

use rotorlab_ga::pga3::PGA3_CAYLEY;

#[test]
fn scalar_times_scalar() {
    let (sign, blade) = PGA3_CAYLEY[0][0];
    assert_eq!(sign, 1);
    assert_eq!(blade, 0);
}

#[test]
fn e1_squared_is_plus_one() {
    let (sign, blade) = PGA3_CAYLEY[0b0001][0b0001];
    assert_eq!(sign, 1);
    assert_eq!(blade, 0);
}

#[test]
fn e0_squared_is_zero() {
    let (sign, _) = PGA3_CAYLEY[0b1000][0b1000];
    assert_eq!(sign, 0, "e0 is null in PGA3");
}

#[test]
fn e1_e2_anticommutes() {
    let (s12, b12) = PGA3_CAYLEY[0b0001][0b0010];
    let (s21, b21) = PGA3_CAYLEY[0b0010][0b0001];
    assert_eq!(b12, 0b0011);
    assert_eq!(b21, 0b0011);
    assert_eq!(s12, 1);
    assert_eq!(s21, -1);
}

#[test]
fn pseudoscalar_squared_is_zero() {
    let (sign, _) = PGA3_CAYLEY[0b1111][0b1111];
    assert_eq!(
        sign, 0,
        "PGA3 pseudoscalar squares to 0 (degenerate, since e0^2 = 0)"
    );
}

#[test]
fn table_dimensions() {
    assert_eq!(PGA3_CAYLEY.len(), 16);
    assert_eq!(PGA3_CAYLEY[0].len(), 16);
}

use rotorlab_ga::pga3;

#[test]
fn point_at_origin() {
    let p = pga3::point(0.0, 0.0, 0.0);
    // The e_123 coefficient should be 1 (the projective denominator).
    assert_eq!(p.0.get(0b0111), 1.0);
    // No Euclidean offset.
    assert_eq!(p.0.get(0b1110), 0.0);
    assert_eq!(p.0.get(0b1101), 0.0);
    assert_eq!(p.0.get(0b1011), 0.0);
}

#[test]
fn point_at_unit_x() {
    let p = pga3::point(1.0, 0.0, 0.0);
    assert_eq!(p.0.get(0b0111), 1.0);
    assert_eq!(p.0.get(0b1110), 1.0);
}

// ---------------------------------------------------------------------------
// Regression suite for the dual / line-through / plane-through bug fixed
// in 0.0.3. The previous metric-based dual zeroed any blade containing e0,
// which annihilated three of four point components and produced Line(0)
// and Plane(0) for every input. Tests below assert geometric correctness
// against worked-by-hand expected values.
// ---------------------------------------------------------------------------

fn nonzero_blades(mv: &rotorlab_ga::Multivector<rotorlab_ga::pga3::Pga3>) -> Vec<usize> {
    (0..16).filter(|&k| mv.get(k) != 0.0).collect()
}

#[test]
fn dual_of_origin_point_is_pure_e0() {
    // point(0,0,0) = e123 (bitmask 0b0111). J(e123) = +e0 (bitmask 0b1000).
    let p = pga3::point(0.0, 0.0, 0.0);
    let d = p.0.dual();
    assert_eq!(d.get(0b1000), 1.0);
    assert_eq!(nonzero_blades(&d), vec![0b1000]);
}

#[test]
fn dual_of_unit_x_point_has_e0_and_e1_components() {
    // point(1,0,0) = e123 + e023; J(e123) = +e0, J(e023) = -e1.
    // The old bug zeroed the e023 contribution; this catches the regression.
    let p = pga3::point(1.0, 0.0, 0.0);
    let d = p.0.dual();
    assert_eq!(d.get(0b1000), 1.0, "+e0 from e123");
    assert_eq!(
        d.get(0b0001),
        -1.0,
        "-e1 from e023 (was 0 in old buggy dual)"
    );
    assert_eq!(d.get(0b0010), 0.0);
    assert_eq!(d.get(0b0100), 0.0);
}

#[test]
fn line_through_origin_and_unit_x_is_x_axis_bivector() {
    // The x-axis line in plane-based PGA3 is e2 ∧ e3 (intersection of
    // the y=0 and z=0 planes), at bitmask 0b0110.
    let p = pga3::point(0.0, 0.0, 0.0);
    let q = pga3::point(1.0, 0.0, 0.0);
    let line = pga3::line_through(p, q);
    assert_eq!(line.0.get(0b0110), 1.0, "+e23 (the x-axis)");
    // No other blade should be set.
    for k in 0..16 {
        if k != 0b0110 {
            assert_eq!(line.0.get(k), 0.0, "leak at {k:04b}");
        }
    }
}

#[test]
fn line_through_origin_and_unit_y_is_y_axis_bivector() {
    // y-axis line = e3 ∧ e1 (intersection of z=0 and x=0 planes).
    // In bitmask convention e3∧e1 = -e1∧e3 = -(bitmask 0b0101).
    let p = pga3::point(0.0, 0.0, 0.0);
    let q = pga3::point(0.0, 1.0, 0.0);
    let line = pga3::line_through(p, q);
    let v = line.0.get(0b0101);
    assert!(
        v.abs() == 1.0,
        "y-axis must show up at bitmask 0b0101 with unit coeff (got {v})"
    );
    // No other Euclidean-direction bivector should be populated.
    assert_eq!(line.0.get(0b0110), 0.0); // not the x-axis
    assert_eq!(line.0.get(0b0011), 0.0); // not the z-axis
}

#[test]
fn line_through_origin_and_unit_z_is_z_axis_bivector() {
    let p = pga3::point(0.0, 0.0, 0.0);
    let q = pga3::point(0.0, 0.0, 1.0);
    let line = pga3::line_through(p, q);
    // z-axis = e1 ∧ e2, bitmask 0b0011.
    let v = line.0.get(0b0011);
    assert!(
        v.abs() == 1.0,
        "z-axis must be unit at bitmask 0b0011 (got {v})"
    );
    assert_eq!(line.0.get(0b0110), 0.0);
    assert_eq!(line.0.get(0b0101), 0.0);
}

#[test]
fn line_through_self_is_zero() {
    // A point doesn't determine a line; the join of a point with itself
    // must vanish. (Follows from J(p) ∧ J(p) = 0 for grade-1 J(p).)
    let p = pga3::point(2.0, -3.0, 4.0);
    let line = pga3::line_through(p, p);
    for k in 0..16 {
        assert_eq!(line.0.get(k), 0.0, "blade {k:04b} should be zero");
    }
}

#[test]
fn line_through_is_antisymmetric() {
    // line_through(q, p) = -line_through(p, q) — the join is alternating.
    let p = pga3::point(1.0, 2.0, 3.0);
    let q = pga3::point(-1.0, 0.5, 2.0);
    let pq = pga3::line_through(p, q);
    let qp = pga3::line_through(q, p);
    for k in 0..16 {
        assert!(
            (pq.0.get(k) + qp.0.get(k)).abs() < 1e-5,
            "antisymmetry failed at {k:04b}: pq={}, qp={}",
            pq.0.get(k),
            qp.0.get(k),
        );
    }
}

#[test]
fn line_through_lives_at_grade_two() {
    let p = pga3::point(1.0, 0.0, 0.0);
    let q = pga3::point(0.0, 1.0, 0.0);
    let line = pga3::line_through(p, q);
    for k in 0..16 {
        if line.0.get(k) != 0.0 {
            assert_eq!((k as u64).count_ones(), 2, "non-grade-2 leak at {k:04b}");
        }
    }
}

#[test]
fn line_through_two_distinct_points_is_nonzero() {
    // Sweep a handful of point pairs to make sure none collapse to zero.
    let pairs = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)),
        ((0.5, -0.5, 0.0), (0.0, 0.5, -0.5)),
    ];
    for (a, b) in pairs {
        let p = pga3::point(a.0, a.1, a.2);
        let q = pga3::point(b.0, b.1, b.2);
        let line = pga3::line_through(p, q);
        let any_nonzero = (0..16).any(|k| line.0.get(k) != 0.0);
        assert!(any_nonzero, "line_through{a:?}, {b:?} collapsed to zero");
    }
}

#[test]
fn plane_through_xy_axis_points_is_z_equals_zero_plane() {
    // Three points spanning the z=0 plane: origin, +x, +y.
    // Plane-based PGA represents the plane with normal n and offset d as
    //   n_x e1 + n_y e2 + n_z e3 + d e0,
    // so the z=0 plane is a multiple of e3 (bitmask 0b0100).
    let p = pga3::point(0.0, 0.0, 0.0);
    let q = pga3::point(1.0, 0.0, 0.0);
    let r = pga3::point(0.0, 1.0, 0.0);
    let plane = pga3::plane_through(p, q, r);
    let v = plane.0.get(0b0100);
    assert!(
        v.abs() == 1.0,
        "z=0 plane coefficient at e3 should be ±1, got {v}"
    );
    for k in 0..16 {
        if k != 0b0100 {
            assert_eq!(plane.0.get(k), 0.0, "leak at {k:04b}");
        }
    }
}

#[test]
fn plane_through_xz_axis_points_is_y_equals_zero_plane() {
    let p = pga3::point(0.0, 0.0, 0.0);
    let q = pga3::point(1.0, 0.0, 0.0);
    let r = pga3::point(0.0, 0.0, 1.0);
    let plane = pga3::plane_through(p, q, r);
    let v = plane.0.get(0b0010);
    assert!(
        v.abs() == 1.0,
        "y=0 plane coefficient at e2 should be ±1, got {v}"
    );
    for k in 0..16 {
        if k != 0b0010 {
            assert_eq!(plane.0.get(k), 0.0);
        }
    }
}

#[test]
fn plane_through_yz_axis_points_is_x_equals_zero_plane() {
    let p = pga3::point(0.0, 0.0, 0.0);
    let q = pga3::point(0.0, 1.0, 0.0);
    let r = pga3::point(0.0, 0.0, 1.0);
    let plane = pga3::plane_through(p, q, r);
    let v = plane.0.get(0b0001);
    assert!(
        v.abs() == 1.0,
        "x=0 plane coefficient at e1 should be ±1, got {v}"
    );
    for k in 0..16 {
        if k != 0b0001 {
            assert_eq!(plane.0.get(k), 0.0);
        }
    }
}

#[test]
fn plane_through_offset_xy_plane_has_e0_offset() {
    // Three points on z = 1: (0,0,1), (1,0,1), (0,1,1).
    // The plane (e123 + ... + z·e012) ∧ (a·e1+b·e2+c·e3+d·e0) vanishes iff
    // ax + by + cz = d in this crate's plane-based convention. So z = 1
    // corresponds to the multivector `e3 + e0` (up to a global sign), i.e.
    // the e3 and e0 coefficients carry the *same* sign and equal magnitude.
    let p = pga3::point(0.0, 0.0, 1.0);
    let q = pga3::point(1.0, 0.0, 1.0);
    let r = pga3::point(0.0, 1.0, 1.0);
    let plane = pga3::plane_through(p, q, r);
    let normal = plane.0.get(0b0100); // e3 coefficient
    let offset = plane.0.get(0b1000); // e0 coefficient
    assert!(normal.abs() > 1e-5, "z-normal must be present");
    assert!(
        offset.abs() > 1e-5,
        "non-zero offset must be present (z=1 plane)"
    );
    assert_eq!(
        normal, offset,
        "e3 and e0 coefficients must be equal for plane z = 1 (got normal={normal} offset={offset})",
    );
    // No spurious x/y normal components.
    assert_eq!(plane.0.get(0b0001), 0.0);
    assert_eq!(plane.0.get(0b0010), 0.0);
    // Sanity: the original points must satisfy the plane equation.
    // Verified by `incidence_point_on_plane_through_those_points` below.
}

fn point_on_plane(point: &pga3::Point, plane: &pga3::Plane) -> bool {
    // Wedge of point (grade 3) and plane (grade 1) is grade 4 (a scalar
    // multiple of I). Vanishing iff the point lies on the plane.
    let w = point.0.outer_pga3(&plane.0);
    w.get(0b1111).abs() < 1e-4
}

#[test]
fn incidence_each_point_lies_on_plane_through_those_points() {
    // The strongest geometric correctness check: every point used to build
    // a plane must lie on that plane.
    let triples = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0, 2.0, 3.0), (-4.0, 5.0, -6.0), (7.0, -8.0, 9.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
        ((0.5, 0.5, 0.5), (1.5, -0.5, 0.5), (-0.5, 0.5, 1.5)),
    ];
    for (a, b, c) in triples {
        let p = pga3::point(a.0, a.1, a.2);
        let q = pga3::point(b.0, b.1, b.2);
        let r = pga3::point(c.0, c.1, c.2);
        let plane = pga3::plane_through(p, q, r);
        assert!(point_on_plane(&p, &plane), "p {a:?} not on its own plane");
        assert!(point_on_plane(&q, &plane), "q {b:?} not on its own plane");
        assert!(point_on_plane(&r, &plane), "r {c:?} not on its own plane");
    }
}

#[test]
fn plane_through_repeated_point_is_zero() {
    let p = pga3::point(1.0, 2.0, 3.0);
    let q = pga3::point(4.0, 5.0, 6.0);
    let plane = pga3::plane_through(p, p, q);
    for k in 0..16 {
        assert_eq!(plane.0.get(k), 0.0, "blade {k:04b} should be zero");
    }
    let plane2 = pga3::plane_through(p, q, q);
    for k in 0..16 {
        assert_eq!(plane2.0.get(k), 0.0);
    }
}

#[test]
fn plane_through_lives_at_grade_one() {
    let p = pga3::point(1.0, 0.0, 0.0);
    let q = pga3::point(0.0, 1.0, 0.0);
    let r = pga3::point(0.0, 0.0, 1.0);
    let plane = pga3::plane_through(p, q, r);
    for k in 0..16 {
        if plane.0.get(k) != 0.0 {
            assert_eq!((k as u64).count_ones(), 1, "non-grade-1 leak at {k:04b}");
        }
    }
}

#[test]
fn plane_through_three_generic_points_is_nonzero() {
    let p = pga3::point(1.0, 2.0, 3.0);
    let q = pga3::point(-4.0, 5.0, -6.0);
    let r = pga3::point(7.0, -8.0, 9.0);
    let plane = pga3::plane_through(p, q, r);
    let any_nonzero = (0..16).any(|k| plane.0.get(k) != 0.0);
    assert!(any_nonzero);
}
