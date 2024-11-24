// use std::collections::HashMap;
//
// use illoominate::nbr::tifuknn::hyperparams::{
//     PARAMS_BOL, PARAMS_INSTACART, PARAMS_TAFANG, PARAMS_VALUEDSHOPPER,
// };
// use illoominate::nbr::tifuknn::io::read_baskets_file;
// use illoominate::nbr::tifuknn::types::SparseItemVector;
// use illoominate::nbr::tifuknn::TIFUKNN;
// use illoominate::nbr::types::NextBasketDataset;
// use illoominate::sessrec::metrics::hitrate::HitRate;
// use illoominate::sessrec::metrics::mrr::Mrr;
// use illoominate::sessrec::metrics::recall::Recall;
// use illoominate::sessrec::metrics::Metric;
// use illoominate::sessrec::vmisknn::Scored;
//
// fn main() {
//     let baskets_file = "data/nbr_eu1m/baskets.csv";
//
//     let all_baskets_by_user = read_baskets_file(baskets_file);
//
//     let mut train_baskets_by_user = HashMap::new();
//     let mut eval_baskets_by_user = HashMap::new();
//
//     let num_users = all_baskets_by_user.user_baskets.keys().max().copied().unwrap_or(0) + 1;
//     eprintln!("num_users: {}", num_users);
//     let num_items = all_baskets_by_user.user_baskets
//         .values()
//         .flat_map(|baskets| baskets.iter())
//         .flat_map(|basket| basket.items.iter().max())
//         .max()
//         .copied()
//         .unwrap_or(0)
//         + 1;
//     eprintln!("num_items: {}", num_items);
//
//     for (user_id, mut baskets) in all_baskets_by_user.user_baskets {
//         if baskets.len() > 1 {
//             let last_basket = baskets.pop().unwrap();
//             train_baskets_by_user.insert(user_id, baskets);
//             eval_baskets_by_user.insert(user_id, last_basket);
//         }
//     }
//
//     let train_baskets_by_user = NextBasketDataset::from(&train_baskets_by_user);
//
//     let num_recommendations = 20;
//     let model = TIFUKNN::new(&train_baskets_by_user, &PARAMS_BOL);
//
//     let mut mrr = Mrr::new(num_recommendations);
//     let mut hitrate = HitRate::new(num_recommendations);
//     let mut recall = Recall::new(num_recommendations);
//
//     eprintln!("Starting evaluation...");
//     for (user_id, eval_basket) in eval_baskets_by_user {
//         let neighbors = model.find_neighbors(&user_id);
//         let recommended_items = model.predict_for(&user_id, &neighbors, num_recommendations);
//
//         let actual_items: Vec<_> = eval_basket
//             .items
//             .iter()
//             .map(|&item| Scored::new(item as u32, 1.0))
//             .collect();
//
//         mrr.add(&recommended_items, &actual_items);
//         hitrate.add(&recommended_items, &actual_items);
//         recall.add(&recommended_items, &actual_items);
//     }
//
//     // assert!(mrr.result() >= 0.2507, "{} is out of the expected range: {}", mrr.get_name(), mrr.result());
//     // assert!(hitrate.result() >= 0.5826, "{} is out of the expected range: {}", hitrate.get_name(), hitrate.result());
//     // assert!(recall.result() >= 0.4293, "{} is out of the expected range: {}", recall.get_name(), recall.result());
//     println!("{}", baskets_file);
//     println!("{}: {:.4}", mrr.get_name(), mrr.result()); // 0.2507 - 0.2513
//     println!("{}: {:.4}", hitrate.get_name(), hitrate.result()); // 0.5826 - 0.5841
//     println!("{}: {:.4}", recall.get_name(), recall.result()); // 0.4297 - 0.4298
// }
fn main() {

}