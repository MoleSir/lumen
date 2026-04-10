use std::u32;

use lumen_core::{FloatDType, IndexOp, Tensor};
use crate::error::MlResult;

/// Density-Based Spatial Clustering of Applications with Noise
pub struct DBSCAN<T: FloatDType> {
    pub eps: T,
    pub min_pts: usize,
}   

pub struct DBSCANModel<T: FloatDType> {
    pub centers: Tensor<T>,
}

impl<T: FloatDType> DBSCAN<T> {
    pub fn fit(&self, x: &Tensor<T>) -> MlResult<Tensor<u32>> {
        let (n_samples, _) = x.dims2()?;
        // 预先计算两两结点的距离
        // (n_samples, 1, n_features) - (1, n_samples, n_features) => (n_samples, n_samples, n_features)
        let delta = x.unsqueeze(1)?.broadcast_sub(&x.unsqueeze(0)?)?;
        // (n_samples, n_samples, n_features) => (n_samples, n_samples,)
        let distances = delta.sqr()?.sum(2)?.sqrt()?;

        let mut labels = vec![u32::MAX; n_samples];
        let mut visited = vec![false; n_samples];
        let mut cluster_id = 0u32;

        // 遍历每个结点
        for sample in 0..n_samples {
            if visited[sample] {
                continue;
            }

            visited[sample] = true;
            // 获取聚类这个结点足够近的邻居
            let mut neighbors = self.region_query(sample, &distances)?;

            if neighbors.len() < self.min_pts {
                labels[sample] = u32::MAX; 
            } else {
                self.expand_cluster(cluster_id, sample, &mut labels, &mut neighbors, &mut visited, &distances)?;
                cluster_id += 1;
            }
        }

        let labels = Tensor::new(labels)?; 

        Ok(labels) 
    }

    fn expand_cluster(&self, cluster_id: u32, sample: usize, labels: &mut Vec<u32>, neighbors: &mut Vec<usize>, visited: &mut Vec<bool>, distances: &Tensor<T>) -> MlResult<()> {
        labels[sample] = cluster_id;

        let mut i = 0;
        while i < neighbors.len() {
            let neighbor = neighbors[i];
            if !visited[neighbor] {
                visited[neighbor] = true;
                let neighbor_neighbors = self.region_query(neighbor, distances)?;
                if neighbor_neighbors.len() > self.min_pts {
                    neighbors.extend(neighbor_neighbors);
                }
            }

            if labels[neighbor] == u32::MAX {
                labels[neighbor] = cluster_id;
            }
            i += 1;
        }

        Ok(())
    }

    fn region_query(&self, sample: usize, distances: &Tensor<T>) -> MlResult<Vec<usize>> {
        // 取出 sample 对应的位置
        // (n_samples, n_samples,) => (n_samples,)
        let sample_dis = distances.index(sample)?;
        // 过滤数量 (n_samples,)
        let mask = sample_dis.le(self.eps)?;
        
        Ok(
            mask.iter()?    
                .enumerate()
                .filter(|(_, m)| *m)
                .map(|(i, _)| i )
                .collect::<Vec<_>>()
        )
    }   
}
