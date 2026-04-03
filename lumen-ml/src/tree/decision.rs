use std::collections::HashMap;
use lumen_core::{FloatDType, IndexOp, NumDType, Tensor, WithDType};

pub enum DecisionTree<V, T> {
    Leaf(V),
    Node {
        feature_id: usize,
        threshold: T,
        /// features[feature_id] <= threshold
        left: Box<DecisionTree<V, T>>,
        /// features[feature_id] > threshold
        right: Box<DecisionTree<V, T>>,
    }
}

impl<V: WithDType, T: WithDType> DecisionTree<V, T> {
    pub fn predict(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<V>> {
        let mut results = vec![];
        let (n_samples, _) = x.dims2()?;
        for n in 0..n_samples {
            results.push(self.predict_single(&x.index(n)?)?);
        }
        Ok(Tensor::new(results)?)
    }

    pub fn depth(&self) -> usize {
        self.get_depth(1)
    }

    fn get_depth(&self, d: usize) -> usize{
        match self {
            Self::Leaf(_) => d,
            Self::Node { feature_id: _, threshold: _, left, right } => {
                let left_d = left.get_depth(d+1);
                let right_d = right.get_depth(d+1);
                left_d.max(right_d)
            }
        }
    }

    fn predict_single(&self, x: &Tensor<T>) -> lumen_core::Result<V> {
        match self {
            Self::Leaf(v) => Ok(v.clone()),
            Self::Node { feature_id, threshold, left, right } => {
                let value = x.index(*feature_id)?.to_scalar()?;
                if value <= *threshold {
                    left.predict_single(x)
                } else {
                    right.predict_single(x)
                }
            }
        }
    }
}

// ==================================================================================== //
//                      DecisionTreeClassifier
// ==================================================================================== //

pub struct DecisionTreeClassifierTrainer {
    pub max_depth: usize,
}

impl DecisionTreeClassifierTrainer {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }
}

impl DecisionTreeClassifierTrainer {
    /// ## Args
    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples)
    pub fn fit<T: FloatDType>(&self, x: &Tensor<T>, y: &Tensor<u32>) -> lumen_core::Result<DecisionTreeClassifier<T>> {
        let (n_samples, _) = x.dims2()?;
        let n_samples_y = y.dims1()?; 
        if n_samples != n_samples_y {
            lumen_core::bail!("x samples {} != y samples {}", n_samples, n_samples_y);
        }

        let n_class = y.max(0)?.to_scalar()? as usize + 1;
        let root = self.build_tree(0, x, y, n_class)?;

        Ok(DecisionTreeClassifier { root})
    }

    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples)
    fn build_tree<T: FloatDType>(&self, depth: usize, x: &Tensor<T>, y: &Tensor<u32>, n_class: usize) -> lumen_core::Result<Box<DecisionTree<u32, T>>> {
        let (n_samples, n_features) = x.dims2()?;
        let mut counter = HashMap::new();
        for label in y.iter()? {
            *counter.entry(label).or_insert(0) += 1;
        }

        let majority_label = *counter.iter().max_by_key(|e| e.1).unwrap().0;
        /*
            1. is max depth?
            2. only one label(pure)
            3. no enough samples
        */
        if depth >= self.max_depth || counter.len() <= 1 || n_samples < 2 {
            return Ok(Box::new(DecisionTree::Leaf(majority_label)));
        }
        
        // find best split
        let mut best_record = (f64::MAX_VALUE, 0, T::ZERO);
        for feature_id in 0..n_features {
            let features = x.index((.., feature_id))?;
            let best_cur_feat = self.find_best_split_for_feature(&features, &y, n_class)?;
            if best_cur_feat.0 < best_record.0 {
                best_record.0 = best_cur_feat.0;
                best_record.1 = feature_id;
                best_record.2 = best_cur_feat.1;
            }
        }
        if best_record.0 == f64::MAX {
            return Ok(Box::new(DecisionTree::Leaf(majority_label)));
        }
        
        // split it !
        let (left_mask, right_mask) = Self::split_mask(x, best_record.1, best_record.2)?;
        let left_samples = left_mask.true_count()?;
        let right_samples = right_mask.true_count()?;
        if left_samples == 0 || right_samples == 0 {
            return Ok(Box::new(DecisionTree::Leaf(majority_label)));
        }

        let left_node = self.build_tree(depth + 1, &x.index(&left_mask)?, &y.index(&left_mask)?, n_class)?;
        let right_node = self.build_tree(depth + 1, &x.index(&right_mask)?, &y.index(&right_mask)?, n_class)?;

        Ok(Box::new(DecisionTree::Node { 
            feature_id: best_record.1, 
            threshold: best_record.2, 
            left: left_node, 
            right: right_node 
        }))
    }

    fn split_mask<T: FloatDType>(x: &Tensor<T>, feature_id: usize, threshold: T) -> lumen_core::Result<(Tensor<bool>, Tensor<bool>)> {
        let left_mask = x.index((.., feature_id))?.le(threshold)?;
        let right_mask = left_mask.not()?;

        Ok((left_mask, right_mask))
    }

    fn find_best_split_for_feature<T: FloatDType>(
        &self, 
        x_feature: &Tensor<T>, 
        y_labels: &Tensor<u32>,
        n_class: usize
    ) -> lumen_core::Result<(f64, T)> {
        let n_samples = x_feature.dims1()?;
        if n_samples < 2 { return Ok((f64::MAX, T::ZERO)); }
    
        // 1. 将特征和标签配对并排序
        let mut samples: Vec<(T, u32)> = x_feature.iter()?.zip(y_labels.iter()?).collect();
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
        // 2. 初始化统计信息
        // 初始状态：所有样本都在右侧
        let mut left_counts = vec![0usize; n_class];
        let mut right_counts = vec![0usize; n_class];
        for &(_, label) in &samples {
            right_counts[label as usize] += 1;
        }
    
        let mut best_gini = f64::MAX;
        let mut best_threshold = T::ZERO;
        let mut n_left = 0;
        let mut n_right = n_samples;
    
        // 3. 遍历所有可能的切分点
        for i in 0..(n_samples - 1) {
            let (val_curr, label_curr) = samples[i];
            let (val_next, _) = samples[i + 1];
    
            // 更新计数器：将当前样本从右边移到左边
            left_counts[label_curr as usize] += 1;
            right_counts[label_curr as usize] -= 1;
            n_left += 1;
            n_right -= 1;
    
            // 如果两个相邻特征值相等，不能在此处切分，否则无法区分左右
            if val_curr == val_next {
                continue;
            }
    
            // 计算当前切分点的 Gini
            // 阈值取两数中间
            let threshold = (val_curr + val_next) / T::from_usize(2);
            
            let gini_left = Self::calculate_gini(&left_counts, n_left);
            let gini_right = Self::calculate_gini(&right_counts, n_right);
            
            let weighted_gini = (n_left as f64 / n_samples as f64) * gini_left 
                              + (n_right as f64 / n_samples as f64) * gini_right;
    
            if weighted_gini < best_gini {
                best_gini = weighted_gini;
                best_threshold = threshold;
            }
        }
    
        Ok((best_gini, best_threshold))
    }
    
    fn calculate_gini(counts: &[usize], total: usize) -> f64 {
        if total == 0 { return 0.0; }
        let mut sum_sq = 0.0;
        let total_f = total as f64;
        for &count in counts {
            if count == 0 { continue; }
            let p = count as f64 / total_f;
            sum_sq += p * p;
        }
        1.0 - sum_sq
    }
}

pub struct DecisionTreeClassifier<T> {
    pub root: Box<DecisionTree<u32, T>>,
}

impl<T: WithDType> DecisionTreeClassifier<T> {
    pub fn predict(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<u32>> {
        self.root.predict(x)
    }

    pub fn depth(&self) -> usize {
        self.root.depth()
    }
}

// ==================================================================================== //
//                      DecisionTreeRegressor
// ==================================================================================== //

pub struct DecisionTreeRegressorTrainer {
    pub max_depth: usize,
}

impl DecisionTreeRegressorTrainer {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// ## Args
    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples) 注意：回归树的 y 现在是连续的浮点数 Tensor<T>
    pub fn fit<T: FloatDType>(&self, x: &Tensor<T>, y: &Tensor<T>) -> lumen_core::Result<DecisionTreeRegressor<T>> {
        let (n_samples, _) = x.dims2()?;
        let n_samples_y = y.dims1()?; 
        if n_samples != n_samples_y {
            lumen_core::bail!("x samples {} != y samples {}", n_samples, n_samples_y);
        }

        let root = self.build_tree(0, x, y)?;

        Ok(DecisionTreeRegressor { root })
    }

    /// 构建回归树
    fn build_tree<T: FloatDType>(&self, depth: usize, x: &Tensor<T>, y: &Tensor<T>) -> lumen_core::Result<Box<DecisionTree<T, T>>> {
        let (n_samples, n_features) = x.dims2()?;
        
        // 1. 计算当前节点的平均值
        let mut sum_y = T::ZERO;
        for val in y.iter()? {
            sum_y = sum_y + val;
        }
        let mean_y = sum_y / T::from_usize(n_samples);

        /*
            停止条件：
            1. 达到最大深度
            2. 样本数太少不足以切分
        */
        if depth >= self.max_depth || n_samples < 2 {
            return Ok(Box::new(DecisionTree::Leaf(mean_y)));
        }
        
        // 2. 寻找最佳切分点
        let mut best_record: Option<(T, usize, T)> = None;
        
        for feature_id in 0..n_features {
            let features = x.index((.., feature_id))?;
            
            if let Some((sse, threshold)) = self.find_best_split_for_feature(&features, &y)? {
                match best_record {
                    None => {
                        best_record = Some((sse, feature_id, threshold));
                    },
                    Some((min_sse, _, _)) if sse < min_sse => {
                        best_record = Some((sse, feature_id, threshold));
                    },
                    _ => {}
                }
            }
        }
        
        // 如果找不到可以降低误差的切分点
        let (_best_sse, best_feature_id, best_threshold) = match best_record {
            Some(record) => record,
            None => return Ok(Box::new(DecisionTree::Leaf(mean_y))),
        };
        
        // 3. 切分数据
        let (left_mask, right_mask) = Self::split_mask(x, best_feature_id, best_threshold)?;
        let left_samples = left_mask.true_count()?;
        let right_samples = right_mask.true_count()?;
        
        // 如果切分后有一边为空，说明无法有效切分，返回叶子节点
        if left_samples == 0 || right_samples == 0 {
            return Ok(Box::new(DecisionTree::Leaf(mean_y)));
        }

        // 4. 递归构建左右子树
        let left_node = self.build_tree(depth + 1, &x.index(&left_mask)?, &y.index(&left_mask)?)?;
        let right_node = self.build_tree(depth + 1, &x.index(&right_mask)?, &y.index(&right_mask)?)?;

        Ok(Box::new(DecisionTree::Node { 
            feature_id: best_feature_id, 
            threshold: best_threshold, 
            left: left_node, 
            right: right_node 
        }))
    }

    fn split_mask<T: FloatDType>(x: &Tensor<T>, feature_id: usize, threshold: T) -> lumen_core::Result<(Tensor<bool>, Tensor<bool>)> {
        let left_mask = x.index((.., feature_id))?.le(threshold)?;
        let right_mask = left_mask.not()?;
        Ok((left_mask, right_mask))
    }

    /// 为单个特征寻找最佳的 MSE (均方误差) 切分点
    fn find_best_split_for_feature<T: FloatDType>(
        &self, 
        x_feature: &Tensor<T>, 
        y_targets: &Tensor<T>
    ) -> lumen_core::Result<Option<(T, T)>> { 
        let n_samples = x_feature.dims1()?;
        if n_samples < 2 { return Ok(None); }
    
        // 1. 将特征和目标值配对并按特征值排序
        let mut samples: Vec<(T, T)> = x_feature.iter()?.zip(y_targets.iter()?).collect();
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
        // 2. 初始化统计信息：初始状态所有样本都在右侧
        let mut sum_right = T::ZERO;
        let mut sum_sq_right = T::ZERO;
        for &(_, y_val) in &samples {
            sum_right = sum_right + y_val;
            sum_sq_right = sum_sq_right + (y_val * y_val);
        }
        
        let mut sum_left = T::ZERO;
        let mut sum_sq_left = T::ZERO;
        
        let mut n_left = T::ZERO;
        let mut n_right = T::from_usize(n_samples);

        let mut best_sse: Option<T> = None;
        let mut best_threshold = T::ZERO;
    
        // 3. 遍历所有可能的切分点
        // 利用动态规划的思想，每次将一个样本从右边移到左边，O(N) 算出每一步的平方误差
        for i in 0..(n_samples - 1) {
            let (val_curr, y_curr) = samples[i];
            let (val_next, _) = samples[i + 1];
    
            // 更新统计：将当前样本从右侧移到左侧
            sum_left = sum_left + y_curr;
            sum_sq_left = sum_sq_left + (y_curr * y_curr);
            n_left = n_left + T::from_usize(1);

            sum_right = sum_right - y_curr;
            sum_sq_right = sum_sq_right - (y_curr * y_curr);
            n_right = n_right - T::from_usize(1);
    
            // 如果两个相邻特征值相等，不能在此处切分
            if val_curr == val_next {
                continue;
            }
            
            // 计算左右子集的 SSE (Sum of Squared Errors)
            // SSE = Sum(y^2) - (Sum(y))^2 / N
            let sse_left = sum_sq_left - (sum_left * sum_left) / n_left;
            let sse_right = sum_sq_right - (sum_right * sum_right) / n_right;
            let total_sse = sse_left + sse_right;

            let threshold = (val_curr + val_next) / T::from_usize(2);
    
            // 更新最佳 SSE
            match best_sse {
                None => {
                    best_sse = Some(total_sse);
                    best_threshold = threshold;
                },
                Some(min_sse) if total_sse < min_sse => {
                    best_sse = Some(total_sse);
                    best_threshold = threshold;
                },
                _ => {}
            }
        }
    
        Ok(best_sse.map(|sse| (sse, best_threshold)))
    }
}

pub struct DecisionTreeRegressor<T> {
    pub root: Box<DecisionTree<T, T>>,
}

impl<T: WithDType> DecisionTreeRegressor<T> {
    pub fn predict(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        self.root.predict(x)
    }

    pub fn depth(&self) -> usize {
        self.root.depth()
    }
}

// ==================================================================================== //
//                      test
// ==================================================================================== //

#[cfg(test)]
mod tests {
    use crate::{datasets::{load_diabetes, load_iris}, metrics::accuracy_score, model_selection::train_test_split, tree::{DecisionTreeClassifierTrainer, DecisionTreeRegressorTrainer}};

    #[test]
    fn test_class_iris() {
        let iris = load_iris::<f32>().unwrap();
        let x = iris.data;
        let y = iris.target;
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();

        let trainer = DecisionTreeClassifierTrainer::new(10);
        let model = trainer.fit(&x_train, &y_train).unwrap();

        let y_pred = model.predict(&x_test).unwrap();

        println!("Depth: {}", model.depth());
        println!("Acc: {}", accuracy_score(&y_test, &y_pred).unwrap())
    }

    #[test]
    fn test_diabetes() {
        let diabetes = load_diabetes::<f32>().unwrap();
        let x = diabetes.data;
        let y = diabetes.target;
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();
        
        let trainer = DecisionTreeRegressorTrainer::new(5);
        let model = trainer.fit(&x_train, &y_train).unwrap();
        let y_pred = model.predict(&x_test).unwrap();
        println!("{}", y_test);
        println!("{}", y_pred);
    }
}