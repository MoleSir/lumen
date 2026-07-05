use lumen_core::{FloatDType, Tensor};
use lumen_nn::{functional::LossReduction, Module};

pub const IGNORE_ID: u32 = u32::MAX;

#[derive(Module)]
pub struct CrossEntropy {
    #[module(skip)]
    pub reduction: LossReduction
}

impl CrossEntropy {
    pub fn new(reduction: LossReduction) -> Self {
        Self { reduction }
    }

    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>, target: &Tensor<u32>) -> anyhow::Result<Tensor<T>> {
        let (safe_target, mask) = {
            let valid_mask = target.ne(IGNORE_ID)?;
            let safe_target = valid_mask.if_else(target, 0)?;
            let float_mask = valid_mask.cast::<T>()?; 
            (safe_target, float_mask)
        };
    
        let log_probs = lumen_nn::functional::log_softmax(input, 1)?;
        let gathered = log_probs.gather(safe_target, 1)?;
        let mut loss = gathered.neg()?;
    
        loss = loss.mul(&mask)?;
    
        match self.reduction {
            LossReduction::None => Ok(loss),
            LossReduction::Sum => Ok(loss.sum_all()?),
            LossReduction::Mean => {
                let sum_loss = loss.sum_all()?;
                let valid_count = mask.sum_all()?;
                Ok(sum_loss.div(&valid_count)?)
            }
        }
    }
}

