use crate::nbr::tifuknn::types::HyperParams;

pub const PARAMS_VALUEDSHOPPER: HyperParams = HyperParams {
    m: 7,
    r_b: 1.0,
    r_g: 0.6,
    alpha: 0.5,
    k: 100,
};

pub const PARAMS_DUNNHUMBY: HyperParams = HyperParams {
    m: 3,
    r_b: 0.9,
    r_g: 0.6,
    alpha: 0.2,
    k: 900,
};

pub const PARAMS_INSTACART: HyperParams = HyperParams {
    m: 3,
    r_b: 0.9,
    r_g: 0.7,
    alpha: 0.2,
    k: 100,
};

pub const PARAMS_BOL: HyperParams = HyperParams {
    m: 3,
    r_b: 0.9,
    r_g: 0.7,
    alpha: 0.1,
    k: 75,
};

pub const PARAMS_TAFANG: HyperParams = HyperParams {
    m: 7,
    r_b: 0.9,
    r_g: 0.7,
    alpha: 0.5,
    k: 75,
};
