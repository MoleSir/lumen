use lumen_core::{IndexOp, Tensor};

pub struct Client {
    /// client id
    pub id: usize,
    /// x local data: (n_samples, n_features)
    pub x_local: Tensor<f32>,
    /// y local data: (n_samples,)
    pub y_local: Tensor<bool>,
}

impl Client {
    pub fn new(id: usize, x_local: Tensor<f32>, y_local: Tensor<bool>) -> Self {
        Self { id, x_local, y_local }
    }

    /// mask: (n_samples,)，其中 False 表示设置某些样本无效
    pub fn get_counts(&self, mask: &Tensor<bool>) -> anyhow::Result<(usize, usize)> {
        // (n_valid_samples, )
        let valid_y = self.y_local.index(mask)?;
        // 计算输出 0 / 1 的数量
        let zero_count = valid_y.true_count()?;
        let false_count = valid_y.false_count()?;
        Ok((zero_count, false_count))
    }

    /// 评估这个划分的分布
    /// mask: (n_samples,)，其中 False 表示设置某些样本无效
    pub fn evaluate_split(&self, mask: &Tensor<bool>, feature_index: usize, threshold: f32) -> anyhow::Result<[(usize, usize); 2]> {
        // 用 mask 选出部分样本
        // valid_x: (n_valid_samples, n_features)
        // valid_y: (n_valid_samples, )
        let valid_x = self.x_local.index(mask)?;
        let valid_y = self.y_local.index(mask)?;
        
        /* 
            有效样本中判断是否满足 <= threshold
            - left_condition: (n,) Bool 数组，其中 k 个为 True，表示有 k 个样本在 feature_idx 的值 <= threshold，left_condition 的值表示这 k 个样本的索引
            - right_condition: (n,) Bool 数组，与 left 类似，但保存的是 > threshold
        */
        let left_condition = valid_x.index((.., feature_index))?.ge(threshold)?;
        let right_condition = left_condition.not()?;

        /*
            valid_y[left_condition] 表示样本中满足 <= threshold 的输出
            - 满足 <= threshold 的 0 / 1 数量 和 
            - 满足 >  threshold 的 0 / 1 数量 
        */     
        let left_condition_y = valid_y.index(left_condition)?;
        let right_confition_y = valid_y.index(right_condition)?;

        Ok([
            (left_condition_y.true_count()?, left_condition_y.false_count()?),
            (right_confition_y.true_count()?, right_confition_y.false_count()?)
        ])
    }

    /// 服务器确定需要在 feature_idx 维度使用 threshold 进行划分
    /// mask: (n_samples,)，其中 False 表示设置某些样本无效
    pub fn apply_split(&self, mask: &Tensor<bool>, feature_index: usize, threshold: f32) -> anyhow::Result<(Tensor<bool>, Tensor<bool>)> {
        // 选择出 x_local 数据在 feature_index <= threshold 的样本
        let condition = self.x_local.index((.., feature_index))?.ge(threshold)?;

        let left_mask = mask.and(&condition)?;
        let right_mask = mask.and(&condition.not()?)?;

        Ok((left_mask, right_mask))        
    }
}
