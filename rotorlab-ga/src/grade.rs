//! Grade projection and blade-grade utilities.

/// The grade (popcount of bitmask) of a blade.
///
/// # Example
///
/// ```
/// use rotorlab_ga::grade::blade_grade;
/// assert_eq!(blade_grade(0b0000), 0); // scalar
/// assert_eq!(blade_grade(0b0001), 1); // vector
/// assert_eq!(blade_grade(0b0011), 2); // bivector
/// assert_eq!(blade_grade(0b1111), 4); // pseudoscalar in 4D
/// ```
pub const fn blade_grade(blade: u64) -> u32 {
    blade.count_ones()
}
