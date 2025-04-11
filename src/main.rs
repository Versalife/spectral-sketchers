mod traits;
use ndarray::{Array2, ArrayBase, Data, Ix1,s, Array};
use ndarray_linalg::{QR, SVD};
use rand::Rng;
use rand_distr::{StandardNormal, Distribution};

use traits::{TruncatedSvdResult, SVDSketcher};

/// Implementation of a Randomized SVD sketcher using multiple sketches
pub struct RandomizedSvdSketcher<const M: usize, const N: usize, const S: usize, const L: usize> {
    // Random projection matrices
    omega: Array2<f64>,     // Ω ∈ F^{n×l}
    psi: Array2<f64>,       // Ψ ∈ F^{n×s}
    upsilon: Array2<f64>,   // Υ ∈ F^{l×m}
    phi: Array2<f64>,       // Φ ∈ F^{s×m}
    
    // Sketches
    y: Array2<f64>,         // Y = BΩ (range sketch)
    x: Array2<f64>,         // X = ΥB (co-range sketch)
    z: Array2<f64>,         // Z = ΦBΨ (core sketch)
    
    // Number of rows observed so far
    rows_seen: usize,
}

impl<const M: usize, const N: usize, const S: usize, const L: usize> SVDSketcher<M, N, S, L> 
    for RandomizedSvdSketcher<M, N, S, L> 
{
    fn new() -> Self {
        // 1. Draw and fix random matrices
        let mut rng = rand::rng();
        
        // Generate random matrices with standard normal distribution using a more ergonomic method
        let omega = generate_random_matrix::<N, L>(&mut rng);
        let psi = generate_random_matrix::<N, S>(&mut rng);
        let upsilon = generate_random_matrix::<L, M>(&mut rng);
        let phi = generate_random_matrix::<S, M>(&mut rng);
        
        // Initialize sketches with zeros
        let y = Array2::<f64>::zeros((M, L));
        let x = Array2::<f64>::zeros((L, N));
        let z = Array2::<f64>::zeros((S, L));
        
        Self {
            omega,
            psi,
            upsilon,
            phi,
            y,
            x,
            z,
            rows_seen: 0
        }
    }
    
    fn observe_row<D>(&mut self, row: &ArrayBase<D, Ix1>) 
    where
        D: Data<Elem = f64>
    {
        // Ensure the row has the correct length
        assert_eq!(row.len(), N, "Row must have N elements");
        
        // Increment the row counter
        self.rows_seen += 1;
        let row_idx = self.rows_seen - 1;
        
        if row_idx >= M {
            panic!("Cannot observe more than M rows");
        }
        
        // Update the sketches
        
        // 1. Update Y = BΩ (range sketch)
        for j in 0..L {
            let mut sum = 0.0;
            for k in 0..N {
                sum += row[k] * self.omega[[k, j]];
            }
            self.y[[row_idx, j]] = sum;
        }
        
        // 2. Update X = ΥB (co-range sketch)
        for i in 0..L {
            for j in 0..N {
                self.x[[i, j]] += self.upsilon[[i, row_idx]] * row[j];
            }
        }
        
        // 3. Update Z = ΦBΨ (core sketch)
        for i in 0..S {
            for j in 0..L {
                let mut sum = 0.0;
                for k in 0..N {
                    sum += row[k] * self.psi[[k, j]];
                }
                self.z[[i, j]] += self.phi[[i, row_idx]] * sum;
            }
        }
    }
    
    fn compute_truncated_svd<const K: usize>(&self) -> TruncatedSvdResult<M, N, K> {
        // 3. Perform two QR factorizations: Y = QR1 and X* = PR2
        let (q, _) = self.y.qr().unwrap();
        
        // For X* (transpose of X), we first transpose
        let x_transpose = self.x.t().to_owned();
        let (p, _) = x_transpose.qr().unwrap();
        
        // 4. Compute core approximation: Ĉ = (ΦQ)† Z (P*Ψ)†
        // First compute ΦQ and P*Ψ
        let phi_q = self.phi.dot(&q);
        let p_conj_psi = p.t().dot(&self.psi);
        
        // Compute pseudoinverses
        let phi_q_pinv = moore_penrose_pseudoinverse(&phi_q);
        let p_conj_psi_pinv = moore_penrose_pseudoinverse(&p_conj_psi);
        
        // Compute the core approximation
        let c_hat = phi_q_pinv.dot(&self.z).dot(&p_conj_psi_pinv);
        
        // 5. Form an SVD of the core: Ĉ = Û Σ̂ V̂† ∈ F^{s×s}
        let (u_hat, sigma_hat, vt_hat) = c_hat.svd(true, true).unwrap();
        
        // Extract the needed parts
        let u_hat = u_hat.unwrap();
        let vt_hat = vt_hat.unwrap();
        
        // 6. Set B̂ = (QÛ)Σ̂(PV̂)*
        let qu_hat = q.dot(&u_hat);
        let pv_hat = p.dot(&vt_hat.t());
        
        // 7. Truncate: Σ̂ -> ⟦Σ̂⟧_k with truncation rank k
        // Ensure K is not larger than available singular values
        let k_actual = K.min(sigma_hat.len());
        
        // Truncate the SVD components
        let u_truncated = qu_hat.slice(s![.., 0..k_actual]).to_owned();
        let sigma_truncated = sigma_hat.slice(s![0..k_actual]).to_owned();
        let vt_truncated = pv_hat.t().slice(s![0..k_actual, ..]).to_owned();
        
        // Return the truncated SVD
        TruncatedSvdResult::new(u_truncated, sigma_truncated, vt_truncated)
    }
    
    fn rows_observed(&self) -> usize {
        self.rows_seen
    }
}

/// Generate a random matrix of size R×C filled with normally distributed values
fn generate_random_matrix<const R: usize, const C: usize>(rng: &mut impl Rng) -> Array2<f64> {    
    // Create a vector of random samples, then reshape into a matrix
    let samples: Vec<f64> = (0..R*C)
        .map(|_| StandardNormal.sample(rng))
        .collect();
    
    Array::from_shape_vec((R, C), samples)
        .expect("Failed to create random matrix with the correct shape")
}

/// Compute the Moore-Penrose pseudoinverse of a matrix using SVD
/// A⁺ = V · S⁺ · U^T where A = U · S · V^T is the SVD of A
fn moore_penrose_pseudoinverse(matrix: &Array2<f64>) -> Array2<f64> {
    // Compute the SVD of the matrix
    let (u, s, vt) = matrix.svd(true, true).unwrap();
    let u = u.unwrap();
    let vt = vt.unwrap();
    
    // Determine numerical rank using a relative threshold
    let max_singular_value = s.iter().cloned().fold(0.0f64, f64::max);
    let rcond = 1e-15; // Relative condition number threshold
    let threshold = rcond * max_singular_value;
    
    // Compute reciprocal of singular values above threshold
    let s_reciprocal = s.mapv(|x| if x > threshold { 1.0 / x } else { 0.0 });
    
    // Construct diagonal matrix of reciprocal singular values
    let s_plus_diag = Array2::from_diag(&s_reciprocal);
    
    // Compute pseudoinverse: V · S⁺ · U^T
    vt.t().dot(&s_plus_diag).dot(&u.t())
}

fn main() {
    println!("Hello, world!");
}
