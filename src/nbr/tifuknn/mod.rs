use std::collections::HashMap;

use crate::importance::RetrievalBasedModel;
use crate::nbr::caboose::sparse_topk_index::SparseTopKIndex;
use crate::nbr::caboose::types::{Score, SimilarRow};
use crate::nbr::tifuknn::types::{Basket, HyperParams, SparseItemVector, UserId};
use crate::sessrec::vmisknn::{Scored};
use itertools::Itertools;
use sprs::TriMat;
use crate::nbr::types::NextBasketDataset;

pub mod hyperparams;
pub mod io;
pub mod types;

pub struct TIFUIndex {
    index: SparseTopKIndex,
}

pub struct TIFUKNN {
    index: TIFUIndex,
    user_embeddings: HashMap<UserId, SparseItemVector>,
    hyper_params: HyperParams,
}

impl TIFUKNN {
    pub fn new(baskets_by_user: &NextBasketDataset, hyper_params: &HyperParams) -> Self {
        let hyper_params = *hyper_params;
        let qty_users = Self::num_users(baskets_by_user);
        let qty_items = Self::num_items(baskets_by_user);
        let mut interactions = TriMat::new((qty_users, qty_items));
        let mut user_embeddings = HashMap::new();

        for (&user_id, baskets) in baskets_by_user.user_baskets.iter() {
            let group_vectors = Self::compute_group_vectors(baskets, &hyper_params);

            let user_vector = Self::user_vector(&group_vectors, &hyper_params);

            for (item_index, _value) in &user_vector.entries {
                // We create a binary matrix, since we compute the jaccard similarity
                interactions.add_triplet(user_id as usize, *item_index, 1.0);
            }

            user_embeddings.insert(user_id, user_vector);
        }

        let interactions = interactions.to_csr();

        let topk_index = SparseTopKIndex::new(interactions.to_csr(), hyper_params.k);

        let tifu_index = TIFUIndex { index: topk_index };

        TIFUKNN {
            index: tifu_index,
            user_embeddings,
            hyper_params,
        }
    }

    pub fn compute_group_vectors(
        baskets: &Vec<Basket>,
        hyper_params: &HyperParams,
    ) -> Vec<SparseItemVector> {
        let t = baskets.len();

        let m = hyper_params.m as usize;

        let mut group_vectors: Vec<SparseItemVector> = Vec::new();

        if t < m {
            // When the number of baskets for this user is less than the group size `m`
            group_vectors.push(Self::group_vector(baskets, t, hyper_params));
        } else {
            let est_num_vec_each_block = t as f64 / m as f64;
            let base_num_vec_each_block = est_num_vec_each_block.floor() as usize;
            let residual = est_num_vec_each_block - base_num_vec_each_block as f64;
            let num_vec_has_extra_vec = (residual * m as f64).round() as usize;

            let mut offset = 0;
            if residual.abs() < f64::EPSILON {
                // Exact division case
                for _ in 0..m {
                    let basket_group = &baskets[offset..offset + base_num_vec_each_block];
                    group_vectors.push(Self::group_vector(
                        basket_group,
                        base_num_vec_each_block,
                        hyper_params,
                    ));
                    offset += base_num_vec_each_block;
                }
            } else {
                for _ in 0..(m - num_vec_has_extra_vec) {
                    let basket_group = &baskets[offset..offset + base_num_vec_each_block];
                    group_vectors.push(Self::group_vector(
                        basket_group,
                        base_num_vec_each_block,
                        hyper_params,
                    ));
                    offset += base_num_vec_each_block;
                }
                let est_num = est_num_vec_each_block.ceil() as usize;
                for _ in (m - num_vec_has_extra_vec)..m {
                    let basket_group = &baskets[offset..offset + est_num];
                    group_vectors.push(Self::group_vector(basket_group, est_num, hyper_params));
                    offset += est_num;
                }
            }
        }
        group_vectors
    }

    pub fn find_neighbors(&self, user_id: &UserId) -> Vec<SimilarRow> {
        
        self.index.index.neighbors(*user_id)
    }

    pub fn predict_for(
        &self,
        user_id: &UserId,
        neighbors: &Vec<SimilarRow>,
        how_many: usize,
    ) -> Vec<Scored> {
        let mut item_weights = SparseItemVector::new();
        let user_embedding = self
            .user_embeddings
            .get(user_id)
            .cloned()
            .unwrap_or_else(SparseItemVector::new);
        let num_neighbors = neighbors.len();
        let alpha = self.hyper_params.alpha;
        for similar_user in neighbors {
            let neighbor_embedding = self.user_embeddings.get(&similar_user.row).unwrap();

            let neighbor_weight = (1.0 - alpha) * (1.0 / num_neighbors as f64);

            item_weights.plus_mult(neighbor_weight, neighbor_embedding);
        }

        item_weights.plus_mult(alpha, &user_embedding);

        let recommended_items: Vec<_> = item_weights
            .entries
            .into_iter()
            .filter(|(_index, value)| *value > 0.0)
            .sorted_by_key(|(_index, value)| (value * 10000.0) as isize)
            .rev()
            .take(how_many)
            .map(|(item, score)| Scored::new(item as u32, score))
            .collect();

        recommended_items
    }

    pub fn get_all_user_embeddings(&self) -> HashMap<UserId, SparseItemVector> {
        self.user_embeddings.clone()
    }
    //
    // pub fn predict_for(&self, user_id: UserId, neighbors: Vec<SimilarRow>, how_many: usize,) -> Vec<(u64)>{
    //     self.predict_from_vectors(user_embedding, neighbors);
    //     let mut item_weights = SparseItemVector::new();
    //     let num_neighbors = neighbors.len();
    //     let alpha = self.hyper_params.alpha;
    //     for similar_user in neighbors {
    //         let neighbor_id = similar_user.row as usize;
    //         let neighbor_embedding = self.user_embeddings.get(&neighbor_id).unwrap();
    //
    //         let neighbor_weight = (1.0 - alpha) * (1.0 / num_neighbors as f64);
    //
    //         item_weights.plus_mult(neighbor_weight, neighbor_embedding);
    //     }
    //
    //
    //     if let Some(user_embedding) = self.user_embeddings.get(&user_id) {
    //         item_weights.plus_mult(alpha, user_embedding);
    //     }
    //
    //     let recommended_items: Vec<_> = item_weights.entries
    //         .into_iter()
    //         .filter(|(_index, value)| *value > 0.0)
    //         .sorted_by_key(|(_index, value)| (value * 10000.0) as isize)
    //         .rev()
    //         .take(how_many)
    //         .map(|(item, _)| item as u64)
    //         .collect();
    //
    //     recommended_items
    // }
    //

    pub fn predict(&self, user_id: &UserId, how_many: usize) -> Vec<Scored> {
        let neighbors = self.find_neighbors(user_id);
        

        self.predict_for(user_id, &neighbors, how_many)
    }

    fn num_items(all_baskets_by_user: &NextBasketDataset) -> usize {
        let num_items = all_baskets_by_user.user_baskets
            .values()
            .flat_map(|baskets| baskets.iter())
            .flat_map(|basket| basket.items.iter().max())
            .max()
            .copied()
            .unwrap_or(0)
            + 1;
        num_items
    }

    fn num_users(all_baskets_by_user: &NextBasketDataset) -> usize {
        let num_users = all_baskets_by_user.user_baskets.keys().max().copied().unwrap_or(0) + 1;
        num_users as usize
    }

    fn user_vector(group_vectors: &[SparseItemVector], params: &HyperParams) -> SparseItemVector {
        let mut user_vector = SparseItemVector::new();
        let m = params.m as usize;

        for (pos, group_vector) in group_vectors.iter().enumerate() {
            let i = pos + 1;
            let weight = (1.0 / m as f64) * params.r_g.powi((m - i) as i32);
            user_vector.plus_mult(weight, group_vector);
        }

        user_vector
    }

    fn group_vector(baskets: &[Basket], x: usize, params: &HyperParams) -> SparseItemVector {
        let mut group_vector = SparseItemVector::new();

        for (pos, basket) in baskets.iter().enumerate() {
            let j = pos + 1;
            let weight = (1.0 / x as f64) * params.r_b.powi((x - j) as i32);
            for item in &basket.items {
                group_vector.plus_at(*item, weight);
            }
        }

        group_vector
    }
}

impl RetrievalBasedModel for TIFUKNN {
    fn k(&self) -> usize {
        self.hyper_params.k
    }

    fn retrieve_k(&self, query_session: &Vec<Scored>) -> Vec<Scored> {
        if let Some(user_id_score) = query_session.first() {
            let neighbors = self.find_neighbors(&user_id_score.id);
            neighbors
                .iter()
                .map(Scored::from)
                .collect_vec()
        } else {
            vec![]
        }
    }

    fn retrieve_all(&self, query_session: &Vec<Scored>) -> Vec<Scored> {
        self.retrieve_k(query_session)
    }

    fn create_preaggregate(&self) -> HashMap<u64, f64> {
        todo!()
    }

    fn add_to_preaggregate<T: AsRef<Scored>>(
        &self,
        _agg: &mut HashMap<u64, f64>,
        _query_session: &Vec<Scored>,
        _neighbor: &T,
    ) {
        todo!()
    }

    fn remove_from_preaggregate<T: AsRef<Scored>>(
        &self,
        _agg: &mut HashMap<u64, f64>,
        _query_session: &Vec<Scored>,
        _neighbor: &T,
    ) {
        todo!()
    }

    fn generate_from_preaggregate(
        &self,
        _query_session: &Vec<Scored>,
        _agg: &HashMap<u64, f64>,
    ) -> Vec<Scored> {
        todo!()
    }

    fn predict(
        &self,
        query: &Vec<Scored>,
        neighbors: &Vec<Scored>,
        how_many: usize,
    ) -> Vec<Scored> {
        if let Some(user_id_score) = query.first() {
            let converted = neighbors
                .iter()
                .map(|scored| SimilarRow::new(scored.id, scored.score as Score))
                .collect();
            self.predict_for(&user_id_score.id, &converted, how_many)
        } else {
            vec![]
        }
    }
}

// Unit tests for the group_function
#[cfg(test)]
mod tests {
    use super::*;

    // Import everything from the outer module (including TIFUKNN and its dependencies)

    #[test]
    fn test_group_function_t_less_than_m() {
        // number of baskets t is less than m
        let m = 5;
        let baskets: Vec<Basket> = vec![
            Basket {
                id: 1,
                items: vec![101, 102, 103],
            },
            Basket {
                id: 2,
                items: vec![102, 104],
            },
            Basket {
                id: 3,
                items: vec![101, 103],
            },
        ];
        let hyper_params = HyperParams {
            m: m,
            r_b: 0.9,
            r_g: 0.7,
            alpha: 0.5,
            k: 50,
        };
        let result = TIFUKNN::compute_group_vectors(&baskets, &hyper_params);
        assert_eq!(result.len(), 1); // qty expected amount of vectors

        assert_eq!(result.get(0).unwrap().entries.len(), 4); // qty expected sparse entries in vector 0
    }

    #[test]
    fn test_group_function_t_equal_m() {
        let m = 4;
        let baskets: Vec<Basket> = vec![
            Basket {
                id: 1,
                items: vec![101],
            },
            Basket {
                id: 2,
                items: vec![102],
            },
            Basket {
                id: 3,
                items: vec![103],
            },
            Basket {
                id: 4,
                items: vec![104],
            },
        ];
        // Populate baskets_by_user with test data where t < m
        let hyper_params = HyperParams {
            m: m,
            r_b: 0.9,
            r_g: 0.7,
            alpha: 0.5,
            k: 50,
        };
        let result = TIFUKNN::compute_group_vectors(&baskets, &hyper_params);
        assert_eq!(result.len(), 4); // qty expected amount of vectors

        assert_eq!(result.get(0).unwrap().entries.len(), 1); // qty expected sparse entries in vector
        assert_eq!(result.get(1).unwrap().entries.len(), 1);
        assert_eq!(result.get(2).unwrap().entries.len(), 1);
        assert_eq!(result.get(3).unwrap().entries.len(), 1);
    }

    #[test]
    fn test_group_function_residual_case() {
        let m = 3;
        let baskets: Vec<Basket> = vec![
            Basket {
                id: 1,
                items: vec![101],
            },
            Basket {
                id: 2,
                items: vec![102],
            },
            Basket {
                id: 3,
                items: vec![103],
            },
            Basket {
                id: 4,
                items: vec![104],
            },
        ];
        // Populate baskets_by_user with test data where t < m
        let hyper_params = HyperParams {
            m: m,
            r_b: 0.9,
            r_g: 0.7,
            alpha: 0.5,
            k: 50,
        };
        let result = TIFUKNN::compute_group_vectors(&baskets, &hyper_params);
        assert_eq!(result.len(), 3);

        assert_eq!(result.get(0).unwrap().entries.len(), 1);
        assert_eq!(result.get(1).unwrap().entries.len(), 1);
        assert_eq!(result.get(2).unwrap().entries.len(), 2);
    }
}
