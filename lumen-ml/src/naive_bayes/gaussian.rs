use lumen_core::{FloatDType, IndexOp, Tensor};

pub struct GaussianNBTrainer {
    pub var_smoothing: f64, 
}

impl Default for GaussianNBTrainer {
    fn default() -> Self {
        Self { var_smoothing: 1e-9 }
    }
}

pub struct GaussianNB<T: FloatDType> {
    pub class_log_prior: Tensor<T>, // (1, n_class)
    pub theta: Tensor<T>,           // 平均值 \mu: (n_class, n_features)
    pub var: Tensor<T>,             // 方差 \sigma^2: (n_class, n_features)
}

impl GaussianNBTrainer {
    /// ## Args
    /// - x: (n_samples, n_features): 连续型特征矩阵
    /// - y: (n_samples,)：标签 (0, 1, 2...)
    pub fn fit<T: FloatDType>(&self, x: &Tensor<T>, y: &Tensor<u32>) -> lumen_core::Result<GaussianNB<T>> {
        let (n_samples, _) = x.dims2()?;
        let n_samples_y = y.dims1()?; 
        if n_samples != n_samples_y {
            lumen_core::bail!("x samples {} != y samples {}", n_samples, n_samples_y);
        }

        let n_class = y.max_all()?.to_scalar()? + 1;
        

        let mut sample_counts = Vec::with_capacity(n_class as usize);
        let mut thetas = Vec::with_capacity(n_class as usize);
        let mut vars = Vec::with_capacity(n_class as usize);

        for c in 0..n_class {
            let mask = y.eq(c)?; // (n_samples,)

            // 1. 统计当前类别样本数
            let sample_count = mask.true_count()?;
            sample_counts.push(T::from_usize(sample_count));
            
            // 取出属于这个 class 的 X: (n_c, n_features)
            let cur_class_x = x.index(&mask)?; 
            
            // 2. 计算平均值 \mu (沿样本维度求平均)
            let mu = cur_class_x.mean(0)?; // (n_features,)
            thetas.push(mu.clone());

            // 3. 计算方差 \sigma^2 = E[(X - \mu)^2]
            // 注意：这里假设 lumen_core 支持 broadcast_sub 和 pow
            let diff = cur_class_x.broadcast_sub(&mu.unsqueeze(0)?)?;
            let sq_diff = diff.sqr()?;
            let variance = sq_diff.mean(0)?; // (n_features,)
            vars.push(variance);
        }

        // 4. 计算先验概率 \ln P(Y)
        let sample_counts = Tensor::new(sample_counts)?;
        let class_log_prior = (sample_counts / T::from_usize(n_samples)).ln()?.unsqueeze(0)?; // (1, n_class)

        // 5. 组合参数矩阵并加上平滑项 (Variance Smoothing)
        let theta = Tensor::stack(&thetas, 0)?; // (n_class, n_features)
        
        let var = Tensor::stack(&vars, 0)?; // (n_class, n_features)
        var.add_(T::from_f64(self.var_smoothing))?; // 防止方差为 0

        Ok(GaussianNB { class_log_prior, theta, var })
    }
}

impl<T: FloatDType> GaussianNB<T> {
    pub fn predict_log_proba(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // 实现公式: X.matmul(W1.T) - (X^2).matmul(W2.T) + Intercept
        
        // W1 = \mu / \sigma^2  (n_class, n_features)
        let w1 = self.theta.clone().div(&self.var)?; 
        
        // W2 = 1.0 / (2 * \sigma^2)  (n_class, n_features)
        let two = T::two();
        let half = T::half();

        let w2 = self.var.map(|v| T::ONE / (two * v))?;

        // 计算 Intercept 中的 \sum_i ( \ln(2\pi\sigma_i^2) + \mu_i^2 / \sigma_i^2 )
        let two_pi_var_ln = (self.var.clone() * (two * T::pi())).ln()?;
        let mu_sq_over_var = self.theta.clone().sqr()?.div(&self.var)?;
        let sum_term = (two_pi_var_ln + mu_sq_over_var).sum_keepdim(1)?; // (n_class, 1)

        // Intercept = \ln P(Y) - 0.5 * sum_term
        // 这里 class_log_prior 是 (1, n_class)，sum_term 是 (n_class, 1)
        // 需将其转置后相加，使其成为 (1, n_class) 用于后续 broadcast 
        let intercept = self.class_log_prior.clone() - (sum_term.transpose_last()? * half); // (1, n_class)

        // 计算预测矩阵部分
        // term1: X @ W1.T -> (n_samples, n_class)
        let term1 = x.matmul(&w1.transpose_last()?)?;
        
        // term2: X^2 @ W2.T -> (n_samples, n_class)
        let x_sq = x.sqr()?;
        let term2 = x_sq.matmul(&w2.transpose_last()?)?;

        // 最终 log probability: term1 - term2 + Intercept
        let log_prob = (term1 - term2).broadcast_add(&intercept)?; // (n_samples, n_class)
        
        Ok(log_prob)
    }

    pub fn predict(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<u32>> {
        let log_prob = self.predict_log_proba(x)?;
        let y = log_prob.argmax(1)?;
        Ok(y)
    }
}