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
