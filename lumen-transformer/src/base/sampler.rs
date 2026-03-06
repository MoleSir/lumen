use lumen_core::{FloatDType, NumDType, Tensor};
use rand::{rngs::StdRng, SeedableRng, Rng};
use std::cmp::Ordering;

pub struct Sampler {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: f64, top_p: f64, top_k: usize, seed: u64) -> Self { 
        Self {
            temperature,
            top_p,
            top_k,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn sample<T: FloatDType>(&mut self, logits: &Tensor<T>) -> usize {
        let mut logits: Vec<_> = logits.iter().expect("Meta Tensor").map(|v| <T as NumDType>::to_f64(v)).collect();
        
        // temperature = 0, greedy sample
        if self.temperature <= 0.0 {
            return self.argmax(&logits);
        }

        // temerature scaling
        for x in &mut logits {
            *x /= self.temperature;
        }
        // softmax
        let probs = self.softmax(&logits);

        // top-k & top-p sampling
        let filtered_probs = self.apply_top_k_top_p(probs);

        // multinomial sampling
        self.multinomial_sample(&filtered_probs)
    }

    fn argmax(&self, logits: &[f64]) -> usize {
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap() 
    }

    fn softmax(&self, x: &[f64]) -> Vec<f64> {
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&v| v / sum).collect()
    }

    fn apply_top_k_top_p(&self, probs: Vec<f64>) -> Vec<(usize, f64)> {
        let mut sorted_probs: Vec<(usize, f64)> = probs.into_iter().enumerate().collect();
        sorted_probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // top-k
        if self.top_k > 0 && self.top_k < sorted_probs.len() {
            sorted_probs.truncate(self.top_k);
        }

        // top-p
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut cutoff_index = sorted_probs.len();

            for (i, (_, prob)) in sorted_probs.iter().enumerate() {
                cumsum += prob;
                if cumsum > self.top_p {
                    cutoff_index = i + 1;
                    break;
                }
            }

            sorted_probs.truncate(cutoff_index);
        }

        sorted_probs
    }

    fn multinomial_sample(&mut self, sorted_probs: &[(usize, f64)]) -> usize {
        let total_prob: f64 = sorted_probs.iter().map(|(_, p)| p).sum();
        let mut rng_val = self.rng.random_range(0.0..1.0) * total_prob;

        for (index, prob) in sorted_probs {
            rng_val -= prob;
            if rng_val <= 0.0 {
                return *index;
            }
        }

        unreachable!()
    }
}