use lumen_core::{FloatDType, Tensor};
use crate::{utils, MlResult, PredictFitWithWeight, PredictModel};

pub struct AdaBoostRegressor<T, F> {
    pub fiter: F,
    pub n_estimators: usize,
    pub learning_rate: T,
}

pub struct AdaBoostModel<T, M> {
    pub estimators: Vec<(M, T)>,
}

impl<T: FloatDType, F> AdaBoostRegressor<T, F> 
where 
    F: PredictFitWithWeight<Input = Tensor<T>, Output = Tensor<T>, Weight = Tensor<T>>,
{
    pub fn fit(&self, x: &Tensor<T>, y: &Tensor<T>) -> MlResult<AdaBoostModel<T, F::Model>> {
        let (n_samples, _) = utils::validate_xy_shapes(x, y, None)?;

        // 初始化权重
        let mut weights = Tensor::<T>::ones((n_samples,))?.div_(T::from_usize(n_samples))?;
        let mut estimators = Vec::new();

        // 依次训练模型
        for _ in 0..self.n_estimators {
            // 使用当前 weight 训练模型
            let model = self.fiter.fit_with_weight(x, y, &weights)?;

            // 计算该模型的误差
            let y_pred = model.predict(x)?;
            let errors = (y_pred - y).abs()?;

            // 计算相对误差，将 errors -> [0, 1]
            let max_error = errors.max_all()?.to_scalar()?;
            let rel_errors = errors / max_error;

            // 计算加权平均误差率
            // weights 表示样本的重要程度，将重要性体现到误差上，接着求和
            // 如果模型在权重很高的样本上算错了，误差会很大
            let avg_error = (&rel_errors * &weights).sum_all()?.to_scalar()?;
            // 算法要求 L < 0.5，如果误差超过 0.5，说明这个模型比随机猜测还差，此时会停止迭代。
            if avg_error >= T::half() { 
                break; 
            } 

            // 计算模型权重
            /*

                beta 衡量模型 “好坏” 的系数
                           L
                \beta = -------
                         1 - L
                
                - L -> 0, \beta -> 0
                - L == 0.5, \beta == 1

                                  1              1 - L
                \alpha = \ln (---------) = \ln (-------)
                                \beta              L
                
            */
            let beta = avg_error / (T::ONE - avg_error);
            let alpha = (T::ONE / beta).ln();

            // 更新样本权重
            /*

                new_w = old_w * \beta^(1 - err)

                - 预测非常准确：err->0，更新因子为 \beta^1，而因为 \beta < 1，所以让权重减小
                - 预测非常糟糕：err->1,更新因子接近 1，样本权重保持不变

            */
            let new_weights = rel_errors.iter()?.zip(weights.iter()?)
                .map(|(e, old_w)| {
                    let factor = beta.powf(T::ONE - e); 
                    old_w * factor
                })
                .collect::<Vec<_>>();
            weights = Tensor::new(new_weights)?; 

            // // 归一化权重
            let sum_w = weights.sum_all()?.to_scalar()?; 
            weights.div_(sum_w)?;
        
            estimators.push((model, alpha));
        }   

        Ok(AdaBoostModel { estimators })
    }
}