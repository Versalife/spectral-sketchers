use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

/// Represents the result of a truncated SVD computation with enforced dimensions.
/// The inner representation details are opaque and accessible only through methods.
pub struct TruncatedSvdResult<const M: usize, const N: usize, const K: usize> {
    // Private fields to make the struct opaque
    u: Array2<f64>,
    s: Array1<f64>,
    vt: Array2<f64>,
}

impl<const M: usize, const N: usize, const K: usize> TruncatedSvdResult<M, N, K> {
    /// Creates a new TruncatedSvdResult with the given components
    pub fn new(u: Array2<f64>, s: Array1<f64>, vt: Array2<f64>) -> Self {
        // Verify dimensions match the expected sizes
        assert_eq!(u.dim(), (M, K), "U matrix dimensions should be (M, K)");
        assert_eq!(s.len(), K, "Singular values vector length should be K");
        assert_eq!(vt.dim(), (K, N), "V^T matrix dimensions should be (K, N)");

        Self { u, s, vt }
    }

    /// Get a reference to the U matrix (left singular vectors)
    pub fn u(&self) -> &Array2<f64> {
        &self.u
    }

    /// Get a reference to the singular values
    pub fn s(&self) -> &Array1<f64> {
        &self.s
    }

    /// Get a reference to the V^T matrix (right singular vectors transposed)
    pub fn vt(&self) -> &Array2<f64> {
        &self.vt
    }

    /// Provides type-safe access to the U matrix with dimensions guaranteed by the type system
    pub fn u_typed<S>(&self) -> ArrayBase<S, Ix2>
    where
        S: Data<Elem = f64>,
        for<'a> ArrayBase<S, Ix2>: From<&'a Array2<f64>>,
    {
        From::from(&self.u)
    }

    /// Provides type-safe access to the singular values with dimensions guaranteed by the type system
    pub fn s_typed<S>(&self) -> ArrayBase<S, Ix1>
    where
        S: Data<Elem = f64>,
        for<'a> ArrayBase<S, Ix1>: From<&'a Array1<f64>>,
    {
        From::from(&self.s)
    }

    /// Provides type-safe access to the Vt matrix with dimensions guaranteed by the type system
    pub fn vt_typed<S>(&self) -> ArrayBase<S, Ix2>
    where
        S: Data<Elem = f64>,
        for<'a> ArrayBase<S, Ix2>: From<&'a Array2<f64>>,
    {
        From::from(&self.vt)
    }
}

/// SVDSketcher represents an object that can incrementally build an SVD decomposition
/// by observing vectors one at a time and computing a truncated SVD on demand.
///
/// # Type Parameters
/// * `M` - Number of rows in the matrix to be sketched
/// * `N` - Number of columns in the matrix to be sketched
/// * `S` - Sketch parameter
/// * `L` - Sketch parameter (must be <= S)
pub trait SVDSketcher<const M: usize, const N: usize, const S: usize, const L: usize> {
    /// Creates a new SVDSketcher.
    /// The dimensions and sketch parameters are defined by the const generic parameters.
    fn new() -> Self
    where
        Self: Sized;

    /// Observes a single row vector from the dataset.
    ///
    /// # Parameters
    /// * `row` - Vector representing a single row in the dataset
    ///
    /// The method ensures that the row dimension matches N at runtime
    fn observe_row<D>(&mut self, row: &ArrayBase<D, Ix1>)
    where
        D: Data<Elem = f64>;

    /// Computes and returns the k-truncated SVD of all rows observed so far.
    ///
    /// # Type Parameters
    /// * `K` - Number of singular values/vectors to keep in the truncated SVD
    ///
    /// # Returns
    /// A TruncatedSvdResult containing the U, S, and V^T matrices with enforced dimensions
    fn compute_truncated_svd<const K: usize>(&self) -> TruncatedSvdResult<M, N, K>;

    /// Returns the current number of rows that have been observed.
    fn rows_observed(&self) -> usize;

    /// Compile-time assertion to ensure S >= L
    const _ASSERT: () = assert!(
        S >= L,
        "Sketch parameter S must be greater than or equal to L"
    );
}
