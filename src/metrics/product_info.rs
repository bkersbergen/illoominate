use crate::sessrec::types::ItemId;
use std::collections::HashSet;
#[derive(Debug, Clone)]
pub struct ProductInfo {
    pub sustainable_products: HashSet<ItemId>,
}

impl ProductInfo {
    pub fn new(sustainable_products: HashSet<ItemId>) -> Self {
        ProductInfo {
            sustainable_products,
        }
    }

    pub fn is_sustainable(&self, product_id: &ItemId) -> bool {
        self.sustainable_products.contains(product_id)
    }
}
